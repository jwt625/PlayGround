# Claude Log Viewer

Web UI for browsing Claude Code inference logs.

## Features

- Real-time log updates (auto-refresh every 2 seconds)
- Collapsible log entries (click on metadata bar to toggle)
- Minimap sidebar for quick navigation
- Filter by success/error status
- Shows three message types:
  - User → CC (user messages)
  - CC → Inference (requests sent to endpoint)
  - Inference → CC (responses from endpoint)

## Running

1. Start the log API server (from proxy directory):
```bash
python log_api.py
```

2. Start the viewer (from viewer directory):
```bash
pnpm dev
```

3. Open browser to `http://localhost:58735`

## Usage

- Click on any log entry's metadata bar to collapse/expand it
- Click on minimap items (left sidebar) to jump to specific logs
- Use filter buttons to show all/success/errors only
- Click "Request body" or "Response body" details to expand

