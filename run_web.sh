#!/bin/sh
# Launch the localhost dashboard backend (127.0.0.1:8000 only).
# Usage: ./run_web.sh [--camera N]
# Stop: Ctrl+C (camera is released on shutdown).
cd "$(dirname "$0")" || exit 1
if [ ! -x ./myenv/bin/python ]; then
  echo "repo venv not found at ./myenv (see README Install)" >&2
  exit 1
fi
exec ./myenv/bin/python webapp.py "$@"
