#!/bin/sh
set -eu

PYTHON_BIN="${PYTHON_BIN:-/home/ray/anaconda3/bin/python}"

exec su -s /bin/sh ray -c "$PYTHON_BIN /serve_app/serve_app.py"
