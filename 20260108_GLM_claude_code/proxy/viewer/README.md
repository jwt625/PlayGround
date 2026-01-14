# Claude Log Viewer

React-based web UI for browsing and analyzing Claude Code inference logs.

## Features

### Log Browsing
- Real-time log updates (auto-refresh every 2 seconds, toggleable)
- Collapsible log entries with request/response details
- Search with regex support and field selection
- Duration filtering (min/max)
- Windowing system for performance (configurable window size)
- Agent type badges with color coding
- Tool usage categorization (read, write, execute, orchestration, interaction)

### Visualization Panels

Four bottom panels for workflow analysis:

- **Timeline Panel**: Chronological request visualization
- **Stats Panel**: Workflow metrics and statistics overview
- **Workflow Panel**: D3.js force-directed graph with:
  - Three edge types: tool_result (grey), subagent_spawn (orange), content_reuse (purple)
  - Interactive zoom/pan (0.1x to 4x scale)
  - Click nodes to select corresponding log entry
  - Drag nodes to reposition
- **Agent Gantt Panel**: Timeline view showing:
  - Agent instances as horizontal bars with request segments
  - Spawn arrows using adaptive Bezier curves
  - Hover tooltips for spawn relationships

## Prerequisites

Before running the viewer, you need entities data. Generate it with:

```bash
# From project root
python3 -m analysis.extract_all_entities \
    proxy/logs/requests_YYYYMMDD.jsonl \
    -o proxy/logs/entities_YYYYMMDD.json
```

## Running

1. Start the log API server (from proxy directory):
```bash
cd ..
uv run python log_api.py
```

2. Start the viewer (from viewer directory):
```bash
pnpm dev
```

3. Open browser to `http://localhost:58735`

## API Endpoints

The log API server provides:

- `GET /api/logs` - Returns enriched logs with agent type and tool info
- `GET /api/workflow` - Returns logs with full workflow graph (cached)
- `GET /api/entities` - Returns entities JSON file
- `GET /api/health` - Health check with cache status

## Usage

- Click on any log entry's metadata bar to collapse/expand it
- Use the search bar to filter logs by content
- Toggle panels using the bottom panel buttons
- Click nodes in Workflow panel to highlight corresponding log
- Hover over Gantt chart spawn arrows to see relationships

