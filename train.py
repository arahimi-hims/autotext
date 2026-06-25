"""
Hidden-prefix recovery via LM-judge supervision.

For each training step:
  1. Decode the current prefix p_i = p(x_i; θ) from T5 (greedy).
  2. Generate ŷ_i = f(p_i ⊕ x_i) using the frozen LLM f.
  3. Query the same LLM class E with (x_i, ŷ_i, y_i*) to infer a candidate
     prefix p_est_i.
  4. Minimise cross-entropy between T5's output distribution and p_est_i
     (teacher-forcing).

  p*           -- hidden prefix drawn from prefix_pool at startup
  y_i*         -- gold output f(p* ⊕ x_i), pre-computed once and cached
  p(x; θ)     -- T5-small seq2seq prefix model (trained)
  f(·)         -- frozen Claude via Bedrock (inference-only)
  E(x, ŷ, y*) -- same LLM class as f, prompted to infer the prefix
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
from judge import score_pair
from prefix_model import PrefixModel
from prefix_pool import get_prefix, PREFIX_POOL


def call_llm(llm: FrozenLLM, prompt: str) -> str:
    try:
        return llm(prompt)
    except Exception:
        return ""


def build_prompt(prefix: str, instruction: str) -> str:
    return f"{prefix}\n\n{instruction}" if prefix else instruction


def infer_prefix(llm: FrozenLLM, x: str, y_hat: str, y_star: str) -> str:
    """Ask E to infer the hidden prefix from one (x, ŷ, y*) triple."""
    prompt = (
        "You are given an instruction and two responses to it. "
        "Response A was produced by a language model conditioned on a hidden system prompt. "
        "Response B was produced by the same model conditioned on a different prefix.\n\n"
        f"Instruction:\n{x}\n\n"
        f"Response A (from the hidden system prompt):\n{y_star}\n\n"
        f"Response B (from a different prefix):\n{y_hat}\n\n"
        "Based on the style, tone, and approach of Response A, infer the hidden system prompt. "
        "Output only the system prompt text, with no explanation or preamble. "
        "Keep it concise (1–3 sentences)."
    )
    try:
        return llm(prompt)
    except Exception:
        return ""


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
        y_hat = call_llm(llm, build_prompt(prefix_text, x))
        scores.append(score_pair(y_hat, gold[x]))
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

    # ------------------------------------------------------------------ main loop
    optimizer = torch.optim.Adam(prefix_model.parameters(), lr=args.lr)
    os.makedirs(args.save_dir, exist_ok=True)
    best_score = 0.0

    for step in range(args.num_steps):
        t0 = time.perf_counter()
        batch_instructions = random.sample(instructions, args.batch_size)

        # Step 1: greedy-decode the current prefix for each instruction.
        prefix_model.model.eval()
        with torch.no_grad():
            enc = prefix_model._encode(batch_instructions)
            out_ids = prefix_model.model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=args.max_prefix_tokens,
                do_sample=False,
            )
        prefix_texts = prefix_model.tokenizer.batch_decode(out_ids[:, 1:], skip_special_tokens=True)

        # Step 2: generate ŷ_i = f(p_i ⊕ x_i) for each example.
        pred_prompts = [build_prompt(p, x) for p, x in zip(prefix_texts, batch_instructions)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(pred_prompts)) as pool:
            y_hats = list(pool.map(lambda pr: call_llm(llm, pr), pred_prompts))

        # Step 3: query E(x_i, ŷ_i, y_i*) to infer p_est_i for each example.
        y_stars = [gold[x] for x in batch_instructions]
        triples = list(zip(batch_instructions, y_hats, y_stars))
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(triples)) as pool:
            p_ests = list(pool.map(lambda t: infer_prefix(llm, *t), triples))

        # Step 4: teacher-forcing cross-entropy toward p_est_i.
        prefix_model.model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        n_valid = 0
        for x, p_est in zip(batch_instructions, p_ests):
            if not p_est.strip():
                continue
            enc_x = prefix_model._encode([x])
            target_ids = prefix_model.tokenizer(
                p_est,
                return_tensors="pt",
                max_length=args.max_prefix_tokens,
                truncation=True,
            ).input_ids.to(prefix_model.device)
            out = prefix_model.model(
                input_ids=enc_x.input_ids,
                attention_mask=enc_x.attention_mask,
                labels=target_ids,
            )
            out.loss.backward()
            total_loss += out.loss.item()
            n_valid += 1

        if n_valid > 0:
            torch.nn.utils.clip_grad_norm_(prefix_model.parameters(), max_norm=1.0)
            optimizer.step()

        elapsed = time.perf_counter() - t0
        print(
            f"step {step:4d}  loss={total_loss / max(n_valid, 1):.4f}  "
            f"valid={n_valid}/{args.batch_size}  step_time={elapsed:.1f}s\n"
            f"  p=    '{prefix_texts[0][:80]}'\n"
            f"  p_est='{p_ests[0][:80]}'"
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
        description="Train the T5 prefix model to recover a hidden prefix p* via LM-judge supervision."
    )
    p.add_argument(
        "--prefix-id", type=int, default=0,
        help=f"Index of p* in the prefix pool (0-{len(PREFIX_POOL)-1}).",
    )
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-prefix-tokens", type=int, default=64)
    p.add_argument(
        "--warmup-steps", type=int, default=3,
        help="Supervised steps that warm-start T5 to output 'you are a ...' before the main loop.",
    )
    p.add_argument("--warmup-lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-size", type=int, default=20)
    p.add_argument("--save-dir", type=str, default="checkpoints_lm_judge")
    p.add_argument("--prefix-model", type=str, default="t5-small")
    p.add_argument("--llm-model", type=str, default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    p.add_argument(
        "--device", type=str, default=None,
        help="Device for the prefix model (cpu, mps, cuda). Defaults to auto-detect.",
    )
    p.add_argument(
        "--cache-dir", type=str, default="gold_cache",
        help="Directory for cached gold outputs. Set to empty string to disable.",
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
