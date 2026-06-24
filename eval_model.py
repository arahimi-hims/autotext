"""
Model evaluation on GSM8K using a trained prefix model.

Usage:
    python eval_model.py --checkpoint checkpoints/best [--split test] [--n 100]
"""

import argparse
import random
import textwrap

from dataset import load_gsm8k, extract_gold_answer, extract_predicted_answer, score
from frozen_llm import FrozenLLM
from prefix_model import PrefixModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to a saved PrefixModel checkpoint.")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--n", type=int, default=None,
                   help="Number of examples to evaluate. Defaults to the full split.")
    p.add_argument("--llm-model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    p.add_argument("--max-prefix-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    data = list(load_gsm8k(args.split))
    if args.n is not None:
        data = random.sample(data, min(args.n, len(data)))

    prefix_model = PrefixModel(max_prefix_tokens=args.max_prefix_tokens)
    prefix_model.load(args.checkpoint)
    prefix_model.model.eval()

    llm = FrozenLLM(model=args.llm_model)

    correct = 0
    sep = "-" * 72
    for i, ex in enumerate(data):
        gold = extract_gold_answer(ex["answer"])
        if gold is None:
            continue

        prefix_text, _ = prefix_model.generate(ex["question"], temperature=1.0)
        prompt = f"{prefix_text}\n\n{ex['question']}" if prefix_text else ex["question"]
        output = llm(prompt)
        hit = score(gold, extract_predicted_answer(output))
        correct += hit

        print(sep)
        print(f"[{i+1}] {'CORRECT' if hit else 'WRONG'}  gold={gold}")
        print(f"  Q : {textwrap.shorten(ex['question'], 120)}")
        print(f"  P : {prefix_text!r}")
        print(f"  A : {textwrap.shorten(output.strip(), 120)}")

    print(sep)
    acc = correct / len(data)
    print(f"\nAuto-Simas ({args.checkpoint}, n={len(data)}): accuracy = {acc:.3f} ({correct}/{len(data)})")


if __name__ == "__main__":
    main()
