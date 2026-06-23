import anthropic


class FrozenLLM:
    """Thin wrapper around the Anthropic API that treats Claude as the frozen f(·)."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 512):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def __call__(self, prompt: str) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
