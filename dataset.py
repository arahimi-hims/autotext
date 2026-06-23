import re
from datasets import load_dataset


def load_gsm8k(split: str = "train"):
    return load_dataset("gsm8k", "main")[split]


def extract_gold_answer(answer_str: str) -> float | None:
    """Parse the numeric answer that follows '####' in a GSM8K ground-truth string."""
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_str)
    if m:
        return float(m.group(1).replace(",", ""))
    return None


def extract_predicted_answer(text: str) -> float | None:
    """
    Extract a numeric answer from free-form LLM output.
    Tries the '####' convention first, then falls back to the last number in the text.
    """
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1).replace(",", ""))
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    for tok in reversed(numbers):
        try:
            return float(tok.replace(",", ""))
        except ValueError:
            continue
    return None


def score(gold: float, predicted: float | None) -> float:
    if predicted is None:
        return 0.0
    return 1.0 if abs(predicted - gold) < 1e-2 else 0.0
