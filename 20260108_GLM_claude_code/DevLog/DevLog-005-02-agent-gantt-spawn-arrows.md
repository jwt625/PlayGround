# DevLog-005-02: Agent Gantt Visualization with Spawn Arrows

**Date**: 2026-01-11
**Status**: COMPLETE
**Parent**: DevLog-005 (Entity Extraction and Agent Architecture Analysis)
**Related Files**:
- `proxy/viewer/src/AgentGanttPanel.jsx` (enhanced)
- `proxy/viewer/src/AgentGanttPanel.css` (enhanced)

## Overview

Enhanced the Agent Gantt Chart visualization to display spawn relationships between parent and child agents using SVG arrows. This provides visual insight into the agent workflow DAG structure and helps understand the temporal relationships between agent spawning events.

## Problem Statement

The existing Gantt chart displayed agent instances and their request timelines, but did not visualize the parent-child spawn relationships. This made it difficult to:

1. Understand which agent spawned which child agent
2. Identify the exact request that triggered a spawn
3. Visualize the temporal relationship between spawn events
4. Trace the workflow execution path

## Implementation

### 1. Data Processing

**Task-to-Request Mapping** (lines 88-98):
- Built task index mapping task IDs to their `first_seen_request`
- Handles duplicate task entries by taking the first occurrence
- Used to identify the exact parent request that spawned each child

**Spawn Edge Construction** (lines 100-141):
- Extracted spawn edges from `workflow_dag.edges` with type `subagent_spawn`
- Looked up the spawning request using the task ID
- Calculated source position (right end of spawning request bar)
- Calculated target position (left start of child's first request bar)
- Stored edge metadata: source/target agent IDs, task ID, confidence

### 2. Arrow Path Calculation

**Adaptive Bezier Curve Algorithm** (lines 271-338):

The arrow rendering uses an adaptive S-curve that adjusts based on horizontal distance:

**Case 1: Small Horizontal Distance** (dx < 200px):
- Problem: Simple S-curves become too vertical when spawn happens immediately
- Solution: Forward-backward-forward curve with explicit control points
  - First control point: extends forward (right) from parent by 100px
  - Second control point: extends backward (left) before child by -100px
  - Creates a horizontal S-shape that maintains visual clarity

**Case 2: Large Horizontal Distance** (dx >= 200px):
- Uses standard S-curve with control points at 50% of horizontal distance
- Provides smooth transition between vertically separated agents

### 3. Coordinate System Alignment

**Critical Fix for Position Accuracy** (line 649):
- Initial implementation used `timelineRef.current.clientWidth` for arrow positioning
- Gantt bars use percentages relative to `.gantt-rows` container
- `.gantt-rows` has `width: calc(100% - 20px)` due to padding
- This created a scaling mismatch causing fractional offset
- **Solution**: Changed to use `rowsRef.current.clientWidth` for consistent coordinate system

### 4. SVG Rendering

**Arrow Overlay** (lines 621-668):
- Positioned as absolute overlay on `.gantt-rows` container
- Arrow marker definition with grey fill (#888)
- Each spawn edge rendered as bezier path
- Hover effects: stroke width increases from 1.5px to 2.5px, color lightens to #bbb
- Tooltips display: source agent, target agent, task ID, confidence

### 5. Layout Fixes

**Y-Axis Clipping Resolution**:
- Removed `marginBottom` from last row to prevent unnecessary space (line 567)
- Fixed `minHeight` calculation (line 551):
  - Formula: `N * rowHeight + (N-1) * rowHeight * 0.2 + rowHeight * 1.0`
  - Accounts for: N row heights, N-1 margins between rows, top+bottom padding
- Increased padding from 0.3 to 0.5 rowHeight for better visibility

## Technical Details

### Coordinate Transformations

**Time to X Position**:
```javascript
const sourceXFull = (edge.sourceX - minTime) / duration
const visibleDuration = zoomX.end - zoomX.start
const sourceXRel = (sourceXFull - zoomX.start) / visibleDuration
const x1 = sourceXRel * containerWidth
```

**Agent Index to Y Position**:
```javascript
const sourceAgentVisibleIdx = visibleAgents.findIndex(a => a.agent_id === edge.sourceAgentId)
const y1 = (sourceAgentVisibleIdx + 0.5) * (rowHeight * 1.2) + rowHeight * 0.3
```

### Bezier Curve Parameters

**Small Distance** (forward-backward-forward):
- Forward offset: 100px
- Backward offset: -100px
- Vertical control: ±25% of dy at control points

**Large Distance** (standard S-curve):
- Control point offset: 50% of horizontal distance
- Horizontal control points, no vertical offset

## Results

1. **Visual Clarity**: Spawn relationships are clearly visible with adaptive curve rendering
2. **Accurate Positioning**: Arrows align precisely with request bar endpoints
3. **Temporal Insight**: Easy to identify immediate spawns vs delayed spawns
4. **Interactive**: Hover effects and tooltips provide detailed spawn information
5. **Scalable**: Works correctly at all zoom levels with proper coordinate transformation

## Future Enhancements

1. Color-code arrows by confidence level
2. Add filtering to show/hide specific spawn relationships
3. Highlight spawn path on hover (parent and all descendants)
4. Add animation to trace spawn sequence chronologically
5. Support for other edge types in workflow DAG (e.g., data dependencies)

