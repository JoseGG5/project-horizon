# Ray Serve Service

This folder contains the Ray-specific part of the `serving/` stack.

Use [serving/README.md](/c:/Users/Jose%20Antonio/Desktop/ScriptsProyectos/own/projects_emb/serving/README.md) as the main entrypoint for the full architecture, Docker Compose workflow, and observability overview.

## Purpose

This service loads a local `SentenceTransformer` checkpoint and exposes an HTTP
embedding API through Ray Serve.

It also starts:

- Ray Dashboard
- Ray metrics export endpoint for Prometheus

## Files

- `serve_app.py`: Ray Serve application and deployment definition
- `entrypoint.sh`: container startup wrapper
- `sync_observability_assets.sh`: copies Ray-generated Grafana dashboards into a shared volume
- `Dockerfile`: Ray-based image definition
- `requirements.txt`: extra Python dependencies on top of the Ray base image

## Endpoints

The service exposes:

- `GET /health`
- `GET /metadata`
- `POST /embed`

## Why the sync script exists

Ray generates Grafana dashboard JSON files at runtime under:

`/tmp/ray/session_latest/metrics/grafana/dashboards`

We do not mount that path directly into Grafana because:

- the files do not exist until Ray has started
- `session_latest` is a symlinked runtime path
- a stable shared volume is easier to reason about in Docker Compose

So the startup flow is:

1. `entrypoint.sh` prepares the shared volume permissions
2. `sync_observability_assets.sh` runs in the background
3. Ray starts normally
4. the sync script copies generated dashboard JSON files into the shared Docker volume
5. Grafana imports those dashboards from its side of the same volume

## Important Environment Variables

Core model and serving settings:

- `MODEL_PATH`
- `NORMALIZE_EMBEDDINGS`
- `MAX_BATCH_SIZE`
- `NUM_REPLICAS`
- `RAY_NUM_CPUS`
- `RAY_NUM_GPUS`
- `SERVE_HOST`
- `SERVE_PORT`

Dashboard and metrics integration:

- `DASHBOARD_HOST`
- `DASHBOARD_PORT`
- `METRICS_EXPORT_PORT`
- `RAY_GRAFANA_HOST`
- `RAY_GRAFANA_IFRAME_HOST`
- `RAY_PROMETHEUS_HOST`
- `RAY_PROMETHEUS_NAME`
- `RAY_GRAFANA_ORG_ID`

GPU runtime settings:

- `NVIDIA_GPU_COUNT`
- `NVIDIA_VISIBLE_DEVICES`
- `NVIDIA_DRIVER_CAPABILITIES`

See [serving/.env.template](/c:/Users/Jose%20Antonio/Desktop/ScriptsProyectos/own/projects_emb/serving/.env.template) for variable-by-variable explanations.

## Typical Dev Workflow

From the `serving/` folder:

```bash
cp .env.template .env
docker compose up --build ray grafana prometheus
```

Or start the full stack:

```bash
docker compose up --build
```

## Related Files Outside This Folder

- [docker-compose.yml](/c:/Users/Jose%20Antonio/Desktop/ScriptsProyectos/own/projects_emb/serving/docker-compose.yml)
- [Prometheus config](/c:/Users/Jose%20Antonio/Desktop/ScriptsProyectos/own/projects_emb/serving/monitoring/prometheus/prometheus.yml)
- [Grafana datasource](/c:/Users/Jose%20Antonio/Desktop/ScriptsProyectos/own/projects_emb/serving/monitoring/grafana/provisioning/datasources/prometheus.yml)
- [Grafana dashboards provider](/c:/Users/Jose%20Antonio/Desktop/ScriptsProyectos/own/projects_emb/serving/monitoring/grafana/provisioning/dashboards/ray.yml)
