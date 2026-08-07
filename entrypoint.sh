#!/bin/sh
set -e
mkdir -p "${DATA_DIR:-/app}"
exec uvicorn dashboard.api:app --host 0.0.0.0 --port 8000
