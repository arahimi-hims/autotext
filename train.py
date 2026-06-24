"""
Hidden-prefix recovery training loop.

Minimises L(θ) = -sum_i E(f(p(x_i; θ) ⊕ x_i), y_i*) via REINFORCE.

  p*           -- hidden prefix drawn from prefix_pool at startup
  y_i*         -- gold output f(p* ⊕ x_i), pre-computed once
  p(x; θ)     -- T5-small prefix model (trained)
  f(·)         -- frozen Claude via Bedrock (inference-only)
  E(y, y*)     -- cosine similarity between sentence embeddings, scaled to [0, 100]
"""

import argparse
import concurrent.futures
import json
import os
import random
import time

import torch
from tqdm import tqdm

from frozen_llm import FrozenLLM
from instructions import INSTRUCTIONS
from judge import score_batch
from prefix_model import PrefixModel
from prefix_pool import get_prefix, PREFIX_POOL


def call_llm(llm: FrozenLLM, prompt: str) -> str:
    try:
        return llm(prompt)
    except Exception:
        return ""


def build_prompt(prefix: str, instruction: str) -> str:
    return f"{prefix}\n\n{instruction}" if prefix else instruction


def precompute_gold(
    llm: FrozenLLM,
    p_star: str,
    instructions: list[str],
    cache_path: str | None,
) -> dict[str, str]:
    """Return {instruction: gold_output} for every instruction in the pool.

    If cache_path is given and the file exists, load from disk. Otherwise
    compute via the frozen LLM and optionally save to cache_path.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading gold outputs from cache: {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    print(f"Pre-computing {len(instructions)} gold outputs (p* = '{p_star[:60]}…')")
    prompts = [build_prompt(p_star, x) for x in instructions]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(prompts))) as pool:
        outputs = list(tqdm(pool.map(lambda pr: call_llm(llm, pr), prompts), total=len(prompts), desc="gold"))

    gold = {x: y for x, y in zip(instructions, outputs)}

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(gold, f, indent=2)
        print(f"Gold outputs cached to: {cache_path}")

    return gold


def evaluate(
    prefix_model: PrefixModel,
    llm: FrozenLLM,
    gold: dict[str, str],
    eval_instructions: list[str],
) -> float:
    """Return mean judge score on eval_instructions using greedy prefix decoding."""
    scores = []
    for x in tqdm(eval_instructions, desc="eval", leave=False):
        prefix_text, _ = prefix_model.generate(x, temperature=1.0)
        prompt = build_prompt(prefix_text, x)
        y_hat = call_llm(llm, prompt)
        y_star = gold[x]
        from judge import score_pair
        scores.append(score_pair(y_hat, y_star))
    return sum(scores) / len(scores) if scores else 0.0


def train(args):
    print(f"Hidden prefix (p*): '{get_prefix(args.prefix_id)[:80]}…'")
    p_star = get_prefix(args.prefix_id)

    instructions = INSTRUCTIONS
    eval_instructions = random.sample(instructions, min(args.eval_size, len(instructions)))

    print(f"Initialising frozen LLM ({args.llm_model}) …")
    llm = FrozenLLM(model=args.llm_model)

    cache_path = (
        os.path.join(args.cache_dir, f"gold_prefix{args.prefix_id}.json")
        if args.cache_dir else None
    )
    gold = precompute_gold(llm, p_star, instructions, cache_path)

    print(f"Initialising prefix model ({args.prefix_model}) …")
    prefix_model = PrefixModel(
        model_name=args.prefix_model,
        max_prefix_tokens=args.max_prefix_tokens,
        device=args.device,
    )
    print(f"  device: {prefix_model.device}")

    # ------------------------------------------------------------------ warm-start
    if args.warmup_steps > 0:
        print(f"Warm-starting for {args.warmup_steps} steps …")
        warmup_pool = random.sample(instructions, min(64, len(instructions)))
        prefix_model.warm_start(
            questions=warmup_pool,
            targets=["you are a ..."] * len(warmup_pool),
            num_steps=args.warmup_steps,
            lr=args.warmup_lr,
        )

    # ------------------------------------------------------------------ REINFORCE
    optimizer = torch.optim.Adam(prefix_model.parameters(), lr=args.lr)
    baseline = 50.0  # midpoint of [0, 100] score range

    os.makedirs(args.save_dir, exist_ok=True)
    best_score = 0.0

    for step in range(args.num_steps):
        t0 = time.perf_counter()
        batch_instructions = random.sample(instructions, args.batch_size)

        prefix_texts, log_probs = prefix_model.generate_batch(
            batch_instructions, temperature=args.temperature
        )

        pred_prompts = [build_prompt(p, x) for p, x in zip(prefix_texts, batch_instructions)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(pred_prompts)) as pool:
            y_hats = list(pool.map(lambda pr: call_llm(llm, pr), pred_prompts))

        y_stars = [gold[x] for x in batch_instructions]
        rewards = score_batch(y_hats, y_stars)

        mean_reward = sum(rewards) / len(rewards)
        baseline = 0.9 * baseline + 0.1 * mean_reward

        advantages = torch.tensor(
            [r - baseline for r in rewards],
            dtype=torch.float32,
            device=prefix_model.device,
        )
        loss = -(advantages * log_probs).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prefix_model.parameters(), max_norm=1.0)
        optimizer.step()

        elapsed = time.perf_counter() - t0
        print(
            f"step {step:4d}  reward={mean_reward:.1f}  "
            f"baseline={baseline:.1f}  loss={loss.item():.4f}  "
            f"step_time={elapsed:.1f}s  "
            f"prefix='{prefix_texts[0][:60]}'"
        )

        # -------------------------------------------------------------- eval
        if (step + 1) % args.eval_every == 0:
            score = evaluate(prefix_model, llm, gold, eval_instructions)
            print(f"  → eval mean judge score: {score:.1f}/100")
            if score > best_score:
                best_score = score
                prefix_model.save(os.path.join(args.save_dir, "best"))
                print(f"  → saved best checkpoint (score={best_score:.1f})")

    print(f"\nTraining complete. Best eval score: {best_score:.1f}/100")
    prefix_model.save(os.path.join(args.save_dir, "final"))


def main():
    p = argparse.ArgumentParser(
        description="Train the T5 prefix model to recover a hidden prefix p*."
    )
    p.add_argument(
        "--prefix-id", type=int, default=0,
        help=f"Index of p* in the prefix pool (0-{len(PREFIX_POOL)-1}).",
    )
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature for prefix generation during training.",
    )
    p.add_argument("--max-prefix-tokens", type=int, default=32)
    p.add_argument(
        "--warmup-steps", type=int, default=3,
        help="Supervised warm-start steps that train T5 to directly output p*.",
    )
    p.add_argument("--warmup-lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-size", type=int, default=20)
    p.add_argument("--save-dir", type=str, default="checkpoints2")
    p.add_argument("--prefix-model", type=str, default="t5-small")
    p.add_argument("--llm-model", type=str, default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    p.add_argument(
        "--device", type=str, default=None,
        help="Device for the prefix model (cpu, mps, cuda). Defaults to auto-detect.",
    )
    p.add_argument(
        "--cache-dir", type=str, default="gold_cache",
        help="Directory to cache pre-computed gold outputs. Set to empty string to disable.",
    )
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    if args.cache_dir == "":
        args.cache_dir = None
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()
