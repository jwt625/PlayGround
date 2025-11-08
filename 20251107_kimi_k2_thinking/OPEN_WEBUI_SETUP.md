# Open WebUI Setup

## Initial Setup

The container is pulled automatically by the launch script from:
```
ghcr.io/open-webui/open-webui:main
```

## Quick Start

```bash
./start_openwebui.sh
```

Access at: **http://localhost:8080**

## Configuration

- **Environment:** `.env.openwebui` (API URL and key)
- **Model:** kimi-k2-thinking
- **Port:** 8080
- **Data:** Docker volume `open-webui`

## First-Time Setup

1. Open http://localhost:8080
2. Create admin account (first user)
3. Select model `kimi-k2-thinking` from dropdown
4. Start chatting

## Management

```bash
# View logs
docker logs -f open-webui

# Restart
docker restart open-webui

# Stop
docker stop open-webui

# Remove (keeps data)
docker rm -f open-webui

# Reset everything
docker rm -f open-webui && docker volume rm open-webui
```

## Troubleshooting

**Container won't start:**
```bash
docker logs open-webui
```

**Can't access UI:**
```bash
docker ps | grep open-webui
lsof -i :8080
```

**Model not responding:**
- Check inference service: `curl $LLM_URL/v1/models -H "Authorization: Bearer $LLM_TOKEN"`
- Verify `.env.openwebui` settings
- Check logs: `docker logs open-webui`
