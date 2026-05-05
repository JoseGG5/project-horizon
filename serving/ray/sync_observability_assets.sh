#!/bin/sh
set -eu

SOURCE_DASHBOARDS_DIR="${SOURCE_DASHBOARDS_DIR:-/tmp/ray/session_latest/metrics/grafana/dashboards}"
TARGET_DASHBOARDS_DIR="${TARGET_DASHBOARDS_DIR:-/ray-observability/grafana/dashboards}"
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-5}"

mkdir -p "$TARGET_DASHBOARDS_DIR"

while true; do
    if [ -d "$SOURCE_DASHBOARDS_DIR" ]; then
        # Ray writes dashboard JSONs under /tmp/ray/session_latest/... at
        # runtime. We copy them into a real Docker volume because mounting the
        # symlinked Ray session directory directly is unreliable in Compose.
        find "$SOURCE_DASHBOARDS_DIR" -maxdepth 1 -name "*.json" -exec cp -f {} "$TARGET_DASHBOARDS_DIR" \;
    fi

    sleep "$SYNC_INTERVAL_SECONDS"
done
