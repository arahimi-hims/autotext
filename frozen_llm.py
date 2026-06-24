import boto3
from botocore.config import Config


class FrozenLLM:
    """Bedrock wrapper that treats Claude as the frozen f(·)."""

    def __init__(
        self,
        model: str = "anthropic.claude-haiku-4-5-20251001-v1:0",
        region: str = "us-east-1",
        profile: str = "ai-privileged",
        max_tokens: int = 512,
        read_timeout: int = 30,
    ):
        session = boto3.Session(profile_name=profile, region_name=region)
        self.client = session.client(
            "bedrock-runtime",
            config=Config(read_timeout=read_timeout, connect_timeout=10),
        )
        self.model = model
        self.max_tokens = max_tokens

    def __call__(self, prompt: str) -> str:
        response = self.client.converse(
            modelId=self.model,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": self.max_tokens},
        )
        return response["output"]["message"]["content"][0]["text"]
