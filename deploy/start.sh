#!/bin/sh
# Container entrypoint: seed Nemotron config + demo card, then serve.
set -e
python deploy/seed_demo.py || true
exec python backend/server.py
