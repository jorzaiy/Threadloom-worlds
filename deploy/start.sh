#!/bin/sh
# Container entrypoint: seed Nemotron config + demo card, then serve.
# Render/most PaaS inject PORT; HF Spaces uses THREADLOOM_PORT=7860.
set -e
if [ -n "$PORT" ]; then export THREADLOOM_PORT="$PORT"; fi
python deploy/seed_demo.py || true
exec python backend/server.py
