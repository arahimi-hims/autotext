import boto3


class FrozenLLM:
    """Bedrock wrapper that treats Claude as the frozen f(·)."""

    def __init__(
        self,
        model: str = "anthropic.claude-haiku-4-5-20251001-v1:0",
        region: str = "us-east-1",
        profile: str = "ai-privileged",
        max_tokens: int = 512,
    ):
        session = boto3.Session(profile_name=profile, region_name=region)
        self.client = session.client("bedrock-runtime")
        self.model = model
        self.max_tokens = max_tokens

    def __call__(self, prompt: str) -> str:
        response = self.client.converse(
            modelId=self.model,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": self.max_tokens},
        )
        return response["output"]["message"]["content"][0]["text"]
