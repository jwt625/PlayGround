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
uv pip install flask requests python-dotenv flask_cors
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

## Analysis Tools

### Entity Extraction with Agent Tracking

Extract all entities from logs with agent instance tracking and deduplication:

```bash
python3 -m analysis.extract_all_entities \
    proxy/logs/requests_20260110.jsonl \
    -o proxy/logs/entities_with_tracking.json
```

**Features**:
- **Agent Instance Tracking**: Identifies individual agent instances across requests
- **Entity Deduplication**: Removes duplicate tool definitions, system prompts, etc.
- **Conversation Evolution**: Tracks how conversations evolve
- **Parent-Child Relationships**: Links spawned subagents to parent agents

**Output includes**:
- Agent instances with conversation fingerprints
- Deduplicated entities (tools, prompts, tasks)
- Agent hierarchy and relationships
- Workflow DAG with spawn and tool result edges
- Comprehensive statistics

See [docs/AGENT_WORKFLOW_TRACKING.md](../docs/AGENT_WORKFLOW_TRACKING.md) for detailed documentation.

### React-Based Viewer

Start the viewer to explore logs interactively:

```bash
# Terminal 1: Start log API server
uv run python log_api.py

# Terminal 2: Start React viewer
cd viewer
pnpm dev
```

Open `http://localhost:58735` to access:
- **Timeline Panel**: Chronological request visualization
- **Stats Panel**: Workflow metrics overview
- **Workflow Panel**: D3.js force-directed graph with spawn/tool edges
- **Agent Gantt Panel**: Timeline view with spawn arrows

### Standalone HTML Viewer

For offline viewing, open `viewer/workflow_tree_viz.html` and load an entities JSON file.

### Other Analysis Scripts

- `../scripts/analyze_system_prompts.py`: Analyze system prompt patterns
- `../scripts/extract_all_tools.py`: Extract tool definitions

