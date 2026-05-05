#!/bin/sh
set -eu

PYTHON_BIN="${PYTHON_BIN:-/home/ray/anaconda3/bin/python}"
TARGET_DASHBOARDS_DIR="${TARGET_DASHBOARDS_DIR:-/ray-observability/grafana/dashboards}"

# Run the sync loop in the same container because Ray generates dashboard JSONs
# only after the local cluster starts. Grafana then reads those copied files
# from a shared Docker volume.
mkdir -p "$TARGET_DASHBOARDS_DIR"
# Docker named volumes are typically owned by root on first mount. We fix the
# ownership here so the non-root "ray" user can copy generated dashboard JSONs
# into the shared folder that Grafana watches.
chown -R ray:users /ray-observability

exec su -s /bin/sh ray -c "/serve_app/sync_observability_assets.sh & exec $PYTHON_BIN /serve_app/serve_app.py"
