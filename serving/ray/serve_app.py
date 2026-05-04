import os
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field
import ray
from ray import serve
from sentence_transformers import SentenceTransformer


MODEL_PATH = os.getenv("MODEL_PATH", "/models/checkpoint-600")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))
NUM_REPLICAS = int(os.getenv("NUM_REPLICAS", "1"))
RAY_NUM_CPUS = float(os.getenv("RAY_NUM_CPUS", "1"))
RAY_NUM_GPUS = float(os.getenv("RAY_NUM_GPUS", "0"))
SERVE_HOST = os.getenv("SERVE_HOST", "0.0.0.0")
SERVE_PORT = int(os.getenv("SERVE_PORT", "8000"))
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8265"))
METRICS_EXPORT_PORT = int(os.getenv("METRICS_EXPORT_PORT", "8080"))


api = FastAPI(title="Ray Serve Embedding Service")


class EmbedRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)


@serve.deployment(
    num_replicas=NUM_REPLICAS,
    ray_actor_options={"num_cpus": RAY_NUM_CPUS, "num_gpus": RAY_NUM_GPUS},
    max_ongoing_requests=MAX_BATCH_SIZE,
)
@serve.ingress(api)
class EmbeddingService:
    def __init__(self) -> None:
        self.model = SentenceTransformer(MODEL_PATH)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()

    @api.get("/health")
    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/metadata")
    def metadata(self) -> dict[str, Any]:
        return {
            "model_path": MODEL_PATH,
            "embedding_dimension": self.embedding_dimension,
            "num_replicas": NUM_REPLICAS,
            "ray_num_cpus": RAY_NUM_CPUS,
            "ray_num_gpus": RAY_NUM_GPUS,
            "dashboard_host": DASHBOARD_HOST,
            "dashboard_port": DASHBOARD_PORT,
            "metrics_export_port": METRICS_EXPORT_PORT,
        }

    @api.post("/embed")
    def embed(self, request: EmbedRequest) -> dict[str, Any]:
        if not request.texts:
            return {"embeddings": [], "count": 0, "dimension": self.embedding_dimension}

        embeddings = self.model.encode(
            request.texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return {
            "embeddings": embeddings.tolist(),
            "count": len(request.texts),
            "dimension": self.embedding_dimension,
        }


app = EmbeddingService.bind()


if __name__ == "__main__":
    ray.init(
        include_dashboard=True,
        dashboard_host=DASHBOARD_HOST,
        dashboard_port=DASHBOARD_PORT,
        _metrics_export_port=METRICS_EXPORT_PORT,
    )
    serve.start(http_options={"host": SERVE_HOST, "port": SERVE_PORT})
    serve.run(app, blocking=True)
