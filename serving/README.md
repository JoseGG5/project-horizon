# Serving Stack

This folder contains the local demo stack for serving project embeddings with:

- `api`: FastAPI CRUD and semantic search endpoints
- `db`: PostgreSQL with `pgvector`
- `ray`: Ray Serve deployment hosting the embedding model
- `prometheus`: metrics collection
- `grafana`: dashboards used by Ray Dashboard `Metrics view`

## Architecture

The stack is orchestrated with Docker Compose.

```text
Browser
  -> FastAPI on localhost:${API_PORT}
  -> Ray Dashboard on localhost:${DASHBOARD_PORT}
  -> Grafana on localhost:${GRAFANA_PORT}
  -> Prometheus on localhost:${PROMETHEUS_PORT}

FastAPI
  -> PostgreSQL for project storage
  -> Ray Serve for embeddings

Ray
  -> exposes metrics on :8080
  -> generates Grafana dashboard JSON files at runtime

Prometheus
  -> scrapes Ray metrics from ray:8080
  -> stores time-series data in its own volume

Grafana
  -> queries Prometheus
  -> loads Ray-generated dashboards

Ray Dashboard
  -> embeds Grafana panels in /#/metrics
```

## Folder Layout

- `api/`: FastAPI application, ingestion script, schemas and database access
- `db/init/`: database initialization scripts
- `monitoring/`: Prometheus and Grafana provisioning
- `ray/`: Ray Serve application and startup scripts
- `.env.template`: environment variable template for the full stack
- `docker-compose.yml`: local orchestration entrypoint

## Services

### API

The API is the entrypoint for the application layer.

Main responsibilities:

- create projects
- search projects by semantic similarity
- talk to Ray Serve to generate embeddings
- persist projects and embeddings in PostgreSQL

Relevant files:

- [main.py](/c:/Users/Jose%20Antonio/Desktop/ScriptsProyectos/own/projects_emb/serving/api/app/main.py)
- [ingest.py](/c:/Users/Jose%20Antonio/Desktop/ScriptsProyectos/own/projects_emb/serving/api/app/ingest.py)
- [models.py](/c:/Users/Jose%20Antonio/Desktop/ScriptsProyectos/own/projects_emb/serving/api/app/models.py)

### Database

PostgreSQL stores:

- project metadata
- embedding vectors in a `pgvector` column

The schema is lightweight and is created directly by the app code for the demo.

### Ray Serve

Ray Serve loads the local `SentenceTransformer` checkpoint and exposes:

- `GET /health`
- `GET /metadata`
- `POST /embed`

It also starts:

- Ray Dashboard
- Ray Prometheus metrics endpoint

See [ray/README.md](/c:/Users/Jose%20Antonio/Desktop/ScriptsProyectos/own/projects_emb/serving/ray/README.md) for service-specific details.

### Monitoring

Prometheus and Grafana are used so Ray Dashboard can show rich metrics.

Prometheus:

- scrapes `ray:8080`
- stores time-series metrics in `prometheus_data`

Grafana:

- uses Prometheus as datasource
- imports Ray dashboards from a shared Docker volume
- allows iframe embedding so Ray Dashboard can display panels

## Environment Setup

Create a local env file from the template:

```bash
cd serving
cp .env.template .env
```

The template is intentionally comment-driven. Each variable explains what it is
for instead of shipping opinionated defaults.

Relevant file:

- [serving/.env.template](/c:/Users/Jose%20Antonio/Desktop/ScriptsProyectos/own/projects_emb/serving/.env.template)

## Run Locally

From `serving/`:

```bash
docker compose up --build
```

To stop the stack:

```bash
docker compose down
```

If you want to remove volumes as well:

```bash
docker compose down -v
```

## Host URLs

With the default local setup, these are the main endpoints:

- API: `http://localhost:8002`
- Ray Serve: `http://localhost:8001`
- Ray Dashboard: `http://localhost:8267`
- Ray Dashboard Metrics view: `http://localhost:8267/#/metrics`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- PostgreSQL: `localhost:5432`

## Monitoring Flow

This is the part that usually causes the most confusion:

1. Ray exports metrics in Prometheus format on port `8080`.
2. Prometheus scrapes `ray:8080` and stores the samples in its own TSDB.
3. Grafana uses Prometheus as datasource and runs PromQL queries.
4. Ray generates dashboard JSON files that describe which panels Grafana should render.
5. A small sync script copies those JSON files from Ray's runtime directory into
   a shared Docker volume.
6. Grafana watches that shared folder and imports the dashboards.
7. Ray Dashboard embeds Grafana panels in its `Metrics view`.

Two details matter here:

- Ray and the browser need different Grafana URLs.
  `RAY_GRAFANA_HOST` is for container-to-container traffic, while
  `RAY_GRAFANA_IFRAME_HOST` is for the browser.

- We do not mount `/tmp/ray/session_latest/...` directly into Grafana.
  Ray writes dashboards under a symlinked session path, so we copy them into a
  stable shared volume instead.

Relevant files:

- [docker-compose.yml](/c:/Users/Jose%20Antonio/Desktop/ScriptsProyectos/own/projects_emb/serving/docker-compose.yml)
- [serving/.env](/c:/Users/Jose%20Antonio/Desktop/ScriptsProyectos/own/projects_emb/serving/.env)
- [prometheus.yml](/c:/Users/Jose%20Antonio/Desktop/ScriptsProyectos/own/projects_emb/serving/monitoring/prometheus/prometheus.yml)
- [Grafana datasource](/c:/Users/Jose%20Antonio/Desktop/ScriptsProyectos/own/projects_emb/serving/monitoring/grafana/provisioning/datasources/prometheus.yml)
- [Grafana dashboards provider](/c:/Users/Jose%20Antonio/Desktop/ScriptsProyectos/own/projects_emb/serving/monitoring/grafana/provisioning/dashboards/ray.yml)
- [Ray entrypoint](/c:/Users/Jose%20Antonio/Desktop/ScriptsProyectos/own/projects_emb/serving/ray/entrypoint.sh)
- [Ray dashboard sync script](/c:/Users/Jose%20Antonio/Desktop/ScriptsProyectos/own/projects_emb/serving/ray/sync_observability_assets.sh)

## Useful Commands

Start everything:

```bash
docker compose up --build
```

See running services:

```bash
docker compose ps
```

Tail Grafana logs:

```bash
docker compose logs -f grafana
```

Tail Ray logs:

```bash
docker compose logs -f ray
```

Check Prometheus targets:

```bash
curl http://localhost:9090/api/v1/targets
```

Search Grafana dashboards:

```bash
curl http://localhost:3000/api/search
```

## Troubleshooting

### Ray Dashboard shows empty or broken metrics panels

Check these first:

- `http://localhost:3000/api/search` returns Ray dashboards such as
  `rayDefaultDashboard`
- `http://localhost:9090/targets` shows the Ray target as `UP`
- `RAY_GRAFANA_IFRAME_HOST` points to a browser-visible URL such as
  `http://localhost:3000`

### Grafana says a panel or dashboard UID is missing

That usually means one of these:

- Ray dashboards were not copied into the shared volume yet
- Grafana is watching the wrong directory
- Ray started before generating dashboards and the sync script has not copied them yet

### Prometheus is up but graphs are empty

Check:

- `prometheus.yml` still targets `ray:8080`
- the Ray container is healthy
- you have waited long enough for one or two scrape intervals

## Next Documentation Layer

This README explains the whole stack.

Service-specific deep dives should live closer to the service itself:

- [ray/README.md](/c:/Users/Jose%20Antonio/Desktop/ScriptsProyectos/own/projects_emb/serving/ray/README.md) for Ray Serve internals
