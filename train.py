"""
Auto-Simas training loop.

Minimizes L(p) = sum_i E(x_i, f(p(x_i) ⊕ x_i)) via REINFORCE.

  p(x; θ)  -- T5-small prefix model (trained)
  f(·)     -- frozen Claude (inference-only)
  E(·, ·)  -- exact-match evaluator on GSM8K final numeric answer
"""

import argparse
import os
import random

import torch
from tqdm import tqdm

from dataset import load_gsm8k, extract_gold_answer, extract_predicted_answer, score
from frozen_llm import FrozenLLM
from prefix_model import PrefixModel


def evaluate(prefix_model, llm, examples, temperature: float = 0.0) -> float:
    """Return mean accuracy on examples. Uses greedy decoding (temperature=0) for eval."""
    correct = 0
    for ex in tqdm(examples, desc="eval", leave=False):
        question = ex["question"]
        gold = extract_gold_answer(ex["answer"])
        if gold is None:
            continue

        prefix_model.model.eval()
        with torch.no_grad():
            enc = prefix_model._encode_question(question)
            out_ids = prefix_model.model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=prefix_model.max_prefix_tokens,
                do_sample=False,  # greedy
            )
        prefix_text = prefix_model.tokenizer.decode(out_ids[0, 1:], skip_special_tokens=True)

        prompt = f"{prefix_text}\n\n{question}" if prefix_text else question
        llm_output = llm(prompt)
        predicted = extract_predicted_answer(llm_output)
        correct += score(gold, predicted)

    return correct / len(examples) if examples else 0.0


def train(args):
    print("Loading GSM8K …")
    train_data = list(load_gsm8k("train"))
    test_data = list(load_gsm8k("test"))
    eval_examples = random.sample(test_data, min(args.eval_size, len(test_data)))

    print("Initialising prefix model (T5-small) …")
    prefix_model = PrefixModel(
        model_name=args.prefix_model,
        max_prefix_tokens=args.max_prefix_tokens,
    )

    print(f"Initialising frozen LLM ({args.llm_model}) …")
    llm = FrozenLLM(model=args.llm_model)

    # ------------------------------------------------------------------ warm-start
    if args.warmup_steps > 0:
        print(f"Warm-starting for {args.warmup_steps} steps with: "{args.warmup_prefix}"")
        warmup_qs = [ex["question"] for ex in random.sample(train_data, 64)]
        prefix_model.warm_start(
            fixed_prefix=args.warmup_prefix,
            questions=warmup_qs,
            num_steps=args.warmup_steps,
            lr=args.warmup_lr,
        )

    # ------------------------------------------------------------------ REINFORCE
    optimizer = torch.optim.Adam(prefix_model.parameters(), lr=args.lr)
    baseline = 0.0  # exponential moving average of recent mean rewards

    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0.0

    for step in range(args.num_steps):
        batch = random.sample(train_data, args.batch_size)

        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []
        prefixes: list[str] = []

        for ex in batch:
            question = ex["question"]
            gold = extract_gold_answer(ex["answer"])
            if gold is None:
                continue

            prefix_text, log_prob = prefix_model.generate(question, temperature=args.temperature)
            prompt = f"{prefix_text}\n\n{question}" if prefix_text else question
            llm_output = llm(prompt)
            predicted = extract_predicted_answer(llm_output)
            r = score(gold, predicted)

            log_probs.append(log_prob)
            rewards.append(r)
            prefixes.append(prefix_text)

        if not log_probs:
            continue

        mean_reward = sum(rewards) / len(rewards)
        # EMA baseline to reduce variance without introducing bias asymptotically.
        baseline = 0.9 * baseline + 0.1 * mean_reward

        # REINFORCE loss: -E[(r - b) * log π_θ(p | x)]
        loss = torch.stack([
            -(r - baseline) * lp
            for r, lp in zip(rewards, log_probs)
        ]).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prefix_model.parameters(), max_norm=1.0)
        optimizer.step()

        print(
            f"step {step:4d}  reward={mean_reward:.3f}  "
            f"baseline={baseline:.3f}  loss={loss.item():.4f}  "
            f"prefix="{prefixes[0][:60]}""
        )

        # -------------------------------------------------------------- eval
        if (step + 1) % args.eval_every == 0:
            acc = evaluate(prefix_model, llm, eval_examples)
            print(f"  → eval accuracy: {acc:.3f}")
            if acc > best_acc:
                best_acc = acc
                prefix_model.save(os.path.join(args.save_dir, "best"))
                print(f"  → saved best checkpoint (acc={best_acc:.3f})")

    print(f"\nTraining complete. Best eval accuracy: {best_acc:.3f}")
    prefix_model.save(os.path.join(args.save_dir, "final"))


def main():
    p = argparse.ArgumentParser(description="Train the Auto-Simas prefix model on GSM8K.")
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature for prefix generation during training.")
    p.add_argument("--max-prefix-tokens", type=int, default=32)
    p.add_argument("--warmup-steps", type=int, default=20,
                   help="Number of supervised warm-start steps before REINFORCE.")
    p.add_argument("--warmup-prefix", type=str, default="Let's think step by step.",
                   help="Fixed prefix to imitate during warm-start.")
    p.add_argument("--warmup-lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-size", type=int, default=50)
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--prefix-model", type=str, default="t5-small")
    p.add_argument("--llm-model", type=str, default="anthropic.claude-haiku-4-5-20251001-v1:0")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()
