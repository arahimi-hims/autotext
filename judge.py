"""
Sentence-embedding cosine similarity judge backed by T5 encoder mean-pooling.

Cosine similarity between two sentence vectors is scaled from [-1, 1] to [0, 100].
"""

import torch
import torch.nn.functional as F
from transformers import T5EncoderModel, AutoTokenizer

_MODEL_NAME = "t5-small"
_encoder: T5EncoderModel | None = None
_tokenizer = None


def _get_encoder() -> tuple:
    global _encoder, _tokenizer
    if _encoder is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _encoder = T5EncoderModel.from_pretrained(_MODEL_NAME)
        _encoder.eval()
    return _encoder, _tokenizer


def _embed(texts: list[str], device: str = "cpu") -> torch.Tensor:
    """Return unit-norm mean-pooled encoder embeddings, shape (len(texts), hidden_dim).

    T5's encoder is not fine-tuned for semantic similarity, so scores are a proxy,
    but they give REINFORCE a useful gradient direction when two texts differ in content.
    """
    encoder, tokenizer = _get_encoder()
    encoder = encoder.to(device)
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        hidden = encoder(**enc).last_hidden_state  # (B, T, H)
    mask = enc.attention_mask.unsqueeze(-1).float()  # (B, T, 1)
    summed = (hidden * mask).sum(dim=1)              # (B, H)
    counts = mask.sum(dim=1).clamp(min=1e-9)         # (B, 1)
    return F.normalize(summed / counts, dim=-1)       # (B, H), unit-norm


def score_pair(y: str, y_star: str) -> float:
    """Return a similarity score in [0, 100] between two response strings.

    A score of 100 means identical embeddings; 50 means orthogonal; 0 means
    maximally opposite.
    """
    embs = _embed([y, y_star])
    cos = F.cosine_similarity(embs[0].unsqueeze(0), embs[1].unsqueeze(0)).item()
    return (cos + 1.0) / 2.0 * 100.0


def score_batch(ys: list[str], y_stars: list[str]) -> list[float]:
    """Return similarity scores for a list of (predicted, gold) response pairs."""
    if len(ys) != len(y_stars):
        raise ValueError("ys and y_stars must have the same length.")
    texts = ys + y_stars
    embs = _embed(texts)
    n = len(ys)
    pred_embs = embs[:n]
    gold_embs = embs[n:]
    cos = F.cosine_similarity(pred_embs, gold_embs).tolist()
    return [(c + 1.0) / 2.0 * 100.0 for c in cos]
