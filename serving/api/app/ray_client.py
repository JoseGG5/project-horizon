import os

import httpx


RAY_SERVE_URL = os.getenv("RAY_SERVE_URL", "http://ray:8000")


class EmbeddingClient:
    """The client that interacts with the Ray service
    """
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Asynchronous function to embed text using
        the ray embed/ endpoint

        Args:
            texts (list[str]): A list of texts.

        Returns:
            list[list[float]]: A list of embeddings
        """
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/embed",
                json={"texts": texts},
            )
            response.raise_for_status()
        return response.json()["embeddings"]
