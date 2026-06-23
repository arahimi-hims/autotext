"""
Baseline evaluation on GSM8K with no prefix.

Usage:
    python eval_baseline.py [--split test] [--n 100] [--model claude-haiku-4-5-20251001]
"""

import argparse
import random

from tqdm import tqdm

from dataset import load_gsm8k, extract_gold_answer, extract_predicted_answer, score
from frozen_llm import FrozenLLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--n", type=int, default=None,
                   help="Number of examples to evaluate. Defaults to the full split.")
    p.add_argument("--model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    data = list(load_gsm8k(args.split))
    if args.n is not None:
        data = random.sample(data, min(args.n, len(data)))

    llm = FrozenLLM(model=args.model)

    correct = 0
    for ex in tqdm(data, desc="baseline eval"):
        gold = extract_gold_answer(ex["answer"])
        if gold is None:
            continue
        output = llm(ex["question"])
        correct += score(gold, extract_predicted_answer(output))

    acc = correct / len(data)
    print(f"\nBaseline ({args.model}, n={len(data)}): accuracy = {acc:.3f} ({correct}/{len(data)})")


if __name__ == "__main__":
    main()
