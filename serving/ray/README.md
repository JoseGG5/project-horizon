# Ray Serve service

This folder contains a Dockerized Ray Serve application for serving a local
`SentenceTransformer` model as an HTTP embedding service.
The container extends the official Ray image and installs Python dependencies
with `uv`.
It is configured to use the official Ray GPU image by default.

## Files

- `serve_app.py`: Ray Serve application
- `Dockerfile`: image based on the official `rayproject/ray` image
- `requirements.txt`: extra Python dependencies on top of Ray

## Build

From the repository root:

```bash
docker build -f serving/ray/Dockerfile -t projects-ray-serve .
```

Or from `serving/` with Compose:

```bash
cd serving
cp .env.template .env
docker compose up --build
```

## Run

This example mounts the local `models/` directory into the container and serves
the checkpoint at `models/checkpoint-600`.

```bash
docker run --rm -p 8001:8000 --shm-size=2g ^
  -e MODEL_PATH=/models/checkpoint-600 ^
  -v ${PWD}/models:/models:ro ^
  projects-ray-serve
```

On Linux/macOS shells:

```bash
docker run --rm -p 8001:8000 --shm-size=2g \
  -e MODEL_PATH=/models/checkpoint-600 \
  -v "$(pwd)/models:/models:ro" \
  projects-ray-serve
```

## Endpoints

- `GET /health`
- `GET /metadata`
- `POST /embed`

With the default compose settings, these URLs are exposed on the host:

- `http://localhost:8001/health`
- `http://localhost:8001/metadata`
- `http://localhost:8001/embed`
- `http://localhost:8267` for the Ray Dashboard
- `http://localhost:8080` for the Ray Prometheus metrics endpoint
- `http://localhost:9090` for Prometheus
- `http://localhost:3000` for Grafana

Example request:

```bash
curl -X POST http://localhost:8001/embed \
  -H "Content-Type: application/json" \
  -d "{\"texts\":[\"graph neural networks for energy systems\"]}"
```

## Environment variables

- `MODEL_PATH`: path to the model inside the container
- `NORMALIZE_EMBEDDINGS`: `true` or `false`
- `MAX_BATCH_SIZE`: deployment concurrency cap
- `NUM_REPLICAS`: Ray Serve replicas
- `RAY_NUM_CPUS`: CPUs reserved per replica
- `RAY_NUM_GPUS`: GPUs reserved per replica
- `NVIDIA_GPU_COUNT`: number of GPUs requested from Docker
- `NVIDIA_VISIBLE_DEVICES`: visible NVIDIA devices inside the container
- `NVIDIA_DRIVER_CAPABILITIES`: NVIDIA runtime capabilities
- `SERVE_HOST`: HTTP bind host for Ray Serve
- `SERVE_PORT`: HTTP bind port for Ray Serve
- `DASHBOARD_HOST`: HTTP bind host for the Ray Dashboard
- `DASHBOARD_PORT`: HTTP bind port for the Ray Dashboard
- `METRICS_EXPORT_PORT`: Prometheus metrics export port for Ray
- `RAY_GRAFANA_HOST`: URL Ray uses internally to reach Grafana
- `RAY_GRAFANA_IFRAME_HOST`: URL the browser uses to load Grafana iframes
- `RAY_PROMETHEUS_HOST`: URL Ray uses internally to reach Prometheus
- `RAY_PROMETHEUS_NAME`: Grafana datasource name used by Ray
- `RAY_GRAFANA_ORG_ID`: Grafana organization id used by Ray

## Monitoring Notes

The monitoring stack is split into three pieces:

- Ray exports metrics on `:8080`
- Prometheus scrapes those metrics
- Grafana visualizes them and Ray Dashboard embeds some of those panels

Two implementation details are easy to miss:

1. Prometheus uses a simple static scrape config for `ray:8080`

This is enough for the current single-container demo and is easier to debug than
dynamic discovery.

2. Grafana dashboards are copied out of the Ray container at runtime

Ray generates its dashboard JSON files under
`/tmp/ray/session_latest/metrics/grafana/dashboards` only after the cluster has
started. Because `session_latest` is a symlink, we do not mount it directly
into Grafana. Instead, `sync_observability_assets.sh` copies those JSON files to
a shared Docker volume that Grafana watches.
