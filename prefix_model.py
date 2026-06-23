import torch
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

    def _encode_question(self, question: str):
        return self.tokenizer(
            self.ENCODER_PREFIX + question,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

    def generate(self, question: str, temperature: float = 1.0) -> tuple[str, torch.Tensor]:
        """
        Sample p ~ π_θ(· | question) and return (prefix_text, log_prob).
        log_prob is a scalar tensor with gradient w.r.t. model parameters.
        """
        enc = self._encode_question(question)

        # Sampling pass: no grad needed, just get the discrete token sequence.
        self.model.eval()
        with torch.no_grad():
            out_ids = self.model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=self.max_prefix_tokens,
                do_sample=True,
                temperature=temperature,
            )

        # T5 generate() prepends decoder_start_token_id (= pad_token_id = 0) at position 0.
        # The actual generated tokens start at position 1.
        generated_ids = out_ids[:, 1:]
        prefix_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Forward pass: compute log π_θ(generated_ids | question) WITH gradients.
        self.model.train()
        labels = generated_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        out = self.model(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            labels=labels,
        )

        # out.loss is the mean NLL per non-masked token; multiply by count to get the sum.
        n_tokens = (labels != -100).sum()
        log_prob = -out.loss * n_tokens

        return prefix_text, log_prob

    def warm_start(
        self,
        fixed_prefix: str,
        questions: list[str],
        num_steps: int = 20,
        lr: float = 1e-3,
    ) -> None:
        """Supervised warm-start: teach the model to emit fixed_prefix for any question
        before REINFORCE takes over.
        """
        # Seeds the policy near a known-good point, reducing the sparse-reward
        # variance problem that comes from starting with a random policy.
        target_ids = self.tokenizer(
            fixed_prefix,
            return_tensors="pt",
            max_length=self.max_prefix_tokens,
            truncation=True,
        ).input_ids.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        for step in range(num_steps):
            total_loss = 0.0
            for q in questions[:8]:
                enc = self._encode_question(q)
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
                print(f"  warm-start step {step + 1}/{num_steps}  loss={total_loss / 8:.4f}")

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str) -> None:
        self.model = T5ForConditionalGeneration.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
