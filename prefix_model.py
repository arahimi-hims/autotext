import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, AutoTokenizer


class PrefixModel:
    """Small T5-based seq2seq model that maps a question x to a text prefix p(x; θ).

    The encoder reads x; the decoder generates p.
    """

    ENCODER_PREFIX = "generate reasoning prefix: "

    def __init__(
        self,
        model_name: str = "t5-small",
        max_prefix_tokens: int = 32,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.max_prefix_tokens = max_prefix_tokens

    def parameters(self):
        return self.model.parameters()

    def _encode(self, questions: list[str]):
        return self.tokenizer(
            [self.ENCODER_PREFIX + q for q in questions],
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(self.device)

    def generate_batch(
        self, questions: list[str], temperature: float = 1.0
    ) -> tuple[list[str], torch.Tensor]:
        """Sample p_i ~ π_θ(· | question_i) for each question and return
        (prefix_texts, log_probs) where log_probs has shape (batch,) with
        gradients w.r.t. model parameters.
        """
        enc = self._encode(questions)

        self.model.eval()
        with torch.no_grad():
            out_ids = self.model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=self.max_prefix_tokens,
                do_sample=True,
                temperature=temperature,
            )

        # T5 prepends decoder_start_token_id (= pad_token_id = 0) at position 0.
        generated_ids = out_ids[:, 1:]
        prefix_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        self.model.train()
        labels = generated_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        out = self.model(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            labels=labels,
        )

        # out.loss is the batch mean — we need per-example sums for REINFORCE.
        # Recompute from logits using gather.
        safe_labels = labels.clone()
        safe_labels[safe_labels == -100] = 0
        token_log_probs = (
            F.log_softmax(out.logits, dim=-1)
            .gather(2, safe_labels.unsqueeze(-1))
            .squeeze(-1)
        )
        mask = (labels != -100).float()
        log_probs = (token_log_probs * mask).sum(dim=1)

        return prefix_texts, log_probs

    def generate(self, question: str, temperature: float = 1.0) -> tuple[str, torch.Tensor]:
        """Single-example wrapper around generate_batch, used by eval scripts."""
        texts, log_probs = self.generate_batch([question], temperature)
        return texts[0], log_probs[0]

    def warm_start(
        self,
        questions: list[str],
        targets: list[str],
        num_steps: int = 20,
        lr: float = 1e-3,
        batch_size: int = 8,
    ) -> None:
        """Supervised warm-start: train the model to map each question to its
        corresponding target string before REINFORCE takes over.
        """
        import random as _random
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        pairs = list(zip(questions, targets))
        for step in range(num_steps):
            batch = _random.sample(pairs, min(batch_size, len(pairs)))
            total_loss = 0.0
            for q, t in batch:
                enc = self._encode([q])
                target_ids = self.tokenizer(
                    t,
                    return_tensors="pt",
                    max_length=self.max_prefix_tokens,
                    truncation=True,
                ).input_ids.to(self.device)
                out = self.model(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    labels=target_ids,
                )
                out.loss.backward()
                total_loss += out.loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if (step + 1) % 5 == 0:
                print(f"  warm-start step {step + 1}/{num_steps}  loss={total_loss / batch_size:.4f}")

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str) -> None:
        self.model = T5ForConditionalGeneration.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
