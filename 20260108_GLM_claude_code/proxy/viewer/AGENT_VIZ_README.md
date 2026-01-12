# Agent Gantt Panel

Gantt-chart style visualization for agent instance tracking, implemented as AgentGanttPanel.jsx.

## Features

### Layout
- **Agent Labels**: Left sidebar with agent IDs and type indicators
- **Gantt Chart**: Timeline view showing request distribution as horizontal bars
- **Spawn Arrows**: SVG overlay showing parent-child agent relationships

### Visualization

- Each agent displayed as a row with request segments
- Request segments colored by agent type
- Spawn arrows connect parent request to child agent's first request
- Adaptive Bezier curves for spawn arrows:
  - Small horizontal distance (<200px): forward-backward-forward curve
  - Large horizontal distance: standard S-curve

### Interactions

**Click on Request Segment**:
- Opens modal with full conversation history
- Shows all messages (user/assistant)
- Displays tool uses and results

**Hover on Request**:
- Shows tooltip with request metadata

**Hover on Spawn Arrow**:
- Highlights the spawn relationship
- Shows spawn method (task spawn or tool spawn)

### Controls

- **Zoom**: Adjust time axis scale
- **Pan**: Scroll through timeline

## Color Coding

Agent types are color-coded based on system prompt hash:
- file_path_extractor: Green (#10b981)
- file_search: Blue (#3b82f6)
- bash_processor: Orange (#f59e0b)
- summarizer: Purple (#8b5cf6)
- architect: Pink (#ec4899)
- topic_detector: Cyan (#06b6d4)
- main_agent: Red (#ef4444)
- unknown: Gray (#6b7280)

## Implementation

The panel is part of the React viewer application:

```
src/AgentGanttPanel.jsx  - Main component
src/AgentGanttPanel.css  - Styles
src/MessagesModal.jsx    - Conversation modal
```

## Usage

Access via the React viewer:

```bash
# Start API server
cd proxy
uv run python log_api.py

# Start viewer
cd proxy/viewer
pnpm dev
```

Open `http://localhost:58735` and click the "Agent Gantt" panel button.

