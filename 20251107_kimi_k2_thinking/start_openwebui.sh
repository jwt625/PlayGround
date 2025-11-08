#!/bin/bash
set -e

[ ! -f .env.openwebui ] && echo "Error: .env.openwebui not found" && exit 1

docker rm -f open-webui 2>/dev/null || true

docker run -d --name open-webui -p 8088:8080 \
  --env-file .env.openwebui -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main

echo "Open WebUI running at http://localhost:8080"

