# Claude Code Logging Proxy

A lightweight HTTP proxy server that logs all communication between Claude Code and the inference endpoint.

## Setup

1. Create `.env` file from template:
```bash
cp .env.example .env
```

2. Edit `.env` and set your upstream URL:
```
UPSTREAM_URL=https://your-inference-endpoint.com
```

3. Install dependencies:
```bash
uv sync
```


or
```bash
uv pip install flask requests python-dotenv
```

## Running

Start the proxy server:
```bash
uv run python proxy_server.py
```

The server will start on `http://127.0.0.1:58734` by default.

## Claude Code Configuration

Update your Claude Code settings to use the proxy:

**Option 1: Environment variable**
```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:58734
```

**Option 2: Settings file** (`.claude/settings.json`)
```json
{
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:58734",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "zai-org/GLM-4.6-FP8"
}
```

## Logs

Logs are written to `./logs/requests_YYYYMMDD.jsonl` in JSON Lines format.

Each log entry contains:
- Timestamp (UTC)
- Request method, path, headers, body
- Response status, headers, body
- Sensitive headers (authorization, api-key) are redacted

## Health Check

Visit `http://127.0.0.1:58734/` to verify the proxy is running.

