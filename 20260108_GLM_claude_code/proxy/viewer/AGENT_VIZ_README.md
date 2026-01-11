# Agent Tracker Visualization

Compact Gantt-chart style visualization for agent instance tracking.

## Features

### Layout
- **Sidebar**: List of all agent instances with color coding
- **Gantt Chart**: Timeline view showing request distribution
- **Compact Design**: Minimal spacing, monospace font, dark theme

### Visualization Modes

**Group by Agent** (default):
- Each agent's requests shown as connected points with lines
- Visualizes conversation evolution for each agent
- Lines connect sequential requests from the same agent

**Chronological** (ungrouped):
- All requests shown in time order
- One row per request
- Color-coded by agent type

### Interactions

**Click on Agent** (sidebar or point):
- Opens modal with full conversation history
- Shows all messages (user/assistant)
- Displays tool uses and results
- Formatted with syntax highlighting

**Hover on Point**:
- Shows compact tooltip with:
  - Agent ID
  - Request ID
  - Turn number
  - Message count
  - Timestamp

**Select Agent** (sidebar):
- Highlights agent's rows in timeline
- Keeps agent selected for easy tracking

### Controls

- **File Upload**: Load any entities JSON file
- **Sort By**: 
  - First Request (chronological)
  - Total Requests (most active first)
  - Agent Type (grouped by type)
- **Group by Agent**: Toggle between grouped/chronological view

## Design Principles

1. **Compact**: Maximum information density
2. **Fast**: Minimal DOM manipulation, efficient rendering
3. **Clean**: No emojis, no unnecessary decorations
4. **Functional**: Click to see details, hover for quick info

## Color Coding

Each agent type gets a unique color:
- Blue (#60a5fa)
- Green (#34d399)
- Yellow (#fbbf24)
- Red (#f87171)
- Purple (#a78bfa)
- Orange (#fb923c)
- Cyan (#22d3ee)
- Pink (#f472b6)
- Lime (#4ade80)
- Gold (#facc15)

## Modal Details

When clicking an agent, the modal shows:

**Header**:
- Agent ID with color coding
- Close button

**Metadata**:
- Agent type hash
- Request count
- Conversation turns
- First/last request IDs
- Conversation fingerprint

**Messages**:
- Chronologically ordered
- User messages (blue border)
- Assistant messages (green border)
- Tool uses (orange border, collapsed)
- Tool results (purple border, truncated to 500 chars)

## Performance

- Handles 100+ agents smoothly
- Lazy rendering for large datasets
- Efficient tooltip positioning
- Minimal reflows

## Usage

```bash
# Generate entities file
python3 -m analysis.extract_all_entities \
    proxy/logs/requests_20260110.jsonl \
    -o proxy/logs/entities_with_tracking.json

# Open visualization
open proxy/viewer/agent_tracker_viz.html
```

The visualization auto-loads `../logs/entities_with_tracking.json` if available.

