import { useState, useMemo, useEffect, useRef } from 'react'
import { Tooltip } from 'react-tooltip'
import MessagesModal from './MessagesModal'
import './AgentGanttPanel.css'

function AgentGanttPanel({ entitiesData, logs }) {
  const [selectedAgentId, setSelectedAgentId] = useState(null)
  const [zoomX, setZoomX] = useState({ start: 0, end: 1 }) // X-axis zoom (time)
  const [zoomY, setZoomY] = useState({ start: 0, end: 1 }) // Y-axis zoom (agents)
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState(null)
  const [dragCurrent, setDragCurrent] = useState(null)
  const [modalOpen, setModalOpen] = useState(false)
  const [selectedRequest, setSelectedRequest] = useState(null)
  const [selectedAgent, setSelectedAgent] = useState(null)
  const [containerHeight, setContainerHeight] = useState(0)
  const timelineRef = useRef(null)
  const rowsRef = useRef(null)
  const containerRef = useRef(null)
  const labelsRef = useRef(null)

  // Process agent data
  const agentData = useMemo(() => {
    if (!entitiesData || !logs || logs.length === 0) {
      return { agents: [], minTime: 0, maxTime: 0, duration: 0, spawnEdges: [], requestSequenceEdges: [], contentReuseEdges: [] }
    }

    const agentInstances = entitiesData.entities?.agent_instances || []
    if (agentInstances.length === 0) {
      return { agents: [], minTime: 0, maxTime: 0, duration: 0, spawnEdges: [], requestSequenceEdges: [], contentReuseEdges: [] }
    }

    // Build request map from enriched logs (already have agent_type from /api/logs)
    const requestMap = new Map()
    logs.forEach((log, idx) => {
      requestMap.set(idx, log)
    })

    // Build agent data with request details
    const agents = agentInstances.map(agent => {
      const requests = agent.requests.map((reqId, idx) => {
        const log = requestMap.get(reqId)
        if (!log) return null

        const timestamp = new Date(log.timestamp).getTime()
        const duration = log.response?.duration_ms || 0
        const status = log.response?.status || 'pending'
        const isError = status >= 400

        // Get agent type from enriched log (should be present from /api/raw-logs which returns enriched data)
        const agentType = log.agent_type || { name: 'unknown', label: 'Unknown', color: '#6b7280' }

        return {
          reqId,
          idx,
          timestamp,
          duration,
          endTime: timestamp + duration,
          status,
          isError,
          agentType,
          log
        }
      }).filter(r => r !== null)

      if (requests.length === 0) return null

      return {
        agent_id: agent.agent_id,
        requests,
        parent_agent_id: agent.parent_agent_id,
        child_agent_ids: agent.child_agent_ids || [],
        firstTimestamp: requests[0].timestamp,
        lastTimestamp: requests[requests.length - 1].endTime,
        agentType: requests[0].agentType
      }
    }).filter(a => a !== null)

    // Sort by first timestamp (earliest first for proper Gantt chart)
    agents.sort((a, b) => a.firstTimestamp - b.firstTimestamp)

    // Build agent index for quick lookup
    const agentIndex = new Map()
    agents.forEach((agent, idx) => {
      agentIndex.set(agent.agent_id, idx)
    })

    // Build task index to find which request spawned each task
    const taskIndex = new Map()
    const tasks = entitiesData.entities?.tasks || []
    tasks.forEach(task => {
      if (task.first_seen_request !== undefined) {
        // Only set if not already set (take the first occurrence)
        if (!taskIndex.has(task.id)) {
          taskIndex.set(task.id, task.first_seen_request)
        }
      }
    })

    // Build tool_use index to find which request spawned each tool call
    const toolUseIndex = new Map()
    const toolUses = entitiesData.entities?.tool_uses || []
    toolUses.forEach(toolUse => {
      if (toolUse.request_id !== undefined && toolUse.id) {
        if (!toolUseIndex.has(toolUse.id)) {
          toolUseIndex.set(toolUse.id, toolUse.request_id)
        }
      }
    })

    // Build edges from workflow_dag
    const spawnEdges = []
    const requestSequenceEdges = []
    const contentReuseEdges = []
    const workflowEdges = entitiesData.workflow_dag?.edges || []

    workflowEdges.forEach(edge => {
      if (edge.type === 'subagent_spawn') {
        const sourceIdx = agentIndex.get(edge.source_agent_id)
        const targetIdx = agentIndex.get(edge.target_agent_id)

        if (sourceIdx !== undefined && targetIdx !== undefined) {
          const sourceAgent = agents[sourceIdx]
          const targetAgent = agents[targetIdx]

          // Find the exact request that spawned this child
          // Use source_request_id from the edge (computed by extractor) as primary source
          let sourceRequest = null
          if (edge.source_request_id !== undefined && edge.source_request_id !== null) {
            sourceRequest = sourceAgent.requests.find(r => r.reqId === edge.source_request_id)
          }

          // Fallback to last request if we can't find the exact one
          if (!sourceRequest) {
            sourceRequest = sourceAgent.requests[sourceAgent.requests.length - 1]
          }

          const targetRequest = targetAgent.requests[0]

          spawnEdges.push({
            sourceAgentId: edge.source_agent_id,
            targetAgentId: edge.target_agent_id,
            sourceIdx,
            targetIdx,
            sourceRequestId: edge.source_request_id,  // Include for debugging
            sourceX: sourceRequest.endTime, // Right end of parent's spawning request
            targetX: targetRequest.timestamp, // Left start of child's first request
            spawned_by_task_id: edge.spawned_by_task_id,
            spawned_by_tool_use_id: edge.spawned_by_tool_use_id,
            spawn_method: edge.spawn_method,
            tool_name: edge.tool_name,
            confidence: edge.confidence
          })
        }
      } else if (edge.type === 'request_sequence') {
        const agentIdx = agentIndex.get(edge.source_agent_id)
        if (agentIdx !== undefined) {
          const agent = agents[agentIdx]
          const sourceRequest = agent.requests.find(r => r.reqId === edge.source_request_id)
          const targetRequest = agent.requests.find(r => r.reqId === edge.target_request_id)

          if (sourceRequest && targetRequest) {
            requestSequenceEdges.push({
              agentId: edge.source_agent_id,
              agentIdx,
              sourceRequestId: edge.source_request_id,
              targetRequestId: edge.target_request_id,
              sourceX: sourceRequest.endTime,
              targetX: targetRequest.timestamp,
              time_gap_ms: edge.time_gap_ms,
              confidence: edge.confidence
            })
          }
        }
      } else if (edge.type === 'content_reuse') {
        const sourceIdx = agentIndex.get(edge.source_agent_id)
        const targetIdx = agentIndex.get(edge.target_agent_id)

        if (sourceIdx !== undefined && targetIdx !== undefined) {
          const sourceAgent = agents[sourceIdx]
          const targetAgent = agents[targetIdx]
          const sourceRequest = sourceAgent.requests.find(r => r.reqId === edge.source_request_id)
          const targetRequest = targetAgent.requests.find(r => r.reqId === edge.target_request_id)

          if (sourceRequest && targetRequest) {
            contentReuseEdges.push({
              sourceAgentId: edge.source_agent_id,
              targetAgentId: edge.target_agent_id,
              sourceIdx,
              targetIdx,
              sourceX: sourceRequest.endTime,
              targetX: targetRequest.timestamp,
              content_hash: edge.content_hash,
              confidence: edge.confidence
            })
          }
        }
      }
    })

    // Calculate time bounds
    const allTimestamps = agents.flatMap(a => a.requests.map(r => r.timestamp))
    const allEndTimes = agents.flatMap(a => a.requests.map(r => r.endTime))
    const minTime = Math.min(...allTimestamps)
    const maxTime = Math.max(...allEndTimes)
    const duration = maxTime - minTime

    return { agents, minTime, maxTime, duration, spawnEdges, requestSequenceEdges, contentReuseEdges }
  }, [entitiesData, logs])

  const { agents, minTime, duration, spawnEdges, requestSequenceEdges, contentReuseEdges } = agentData

  // Format duration
  const formatDuration = (ms) => {
    if (ms < 1000) return `${ms.toFixed(0)}ms`
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
    return `${(ms / 60000).toFixed(1)}m`
  }

  // Calculate visible agents and row height based on Y-axis zoom and container height
  const { visibleAgents, rowHeight, needsScroll } = useMemo(() => {
    if (agents.length === 0) return { visibleAgents: [], rowHeight: 2.7, needsScroll: false }

    const startIdx = Math.floor(zoomY.start * agents.length)
    const endIdx = Math.ceil(zoomY.end * agents.length)
    const visible = agents.slice(startIdx, endIdx)

    if (visible.length === 0 || containerHeight === 0) {
      return { visibleAgents: visible, rowHeight: 2.7, needsScroll: false }
    }

    // Calculate row height based on zoom level
    // Base height is 2.7px, scale inversely with the zoom range
    const zoomFactor = 1 / (zoomY.end - zoomY.start)
    const baseRowHeight = 2.7 * zoomFactor

    // Minimum row height (with 20% spacing)
    const MIN_ROW_HEIGHT = 2.7
    const ROW_SPACING_FACTOR = 1.2 // 20% spacing between rows

    // Calculate total height needed with minimum row height
    const minTotalHeight = visible.length * MIN_ROW_HEIGHT * ROW_SPACING_FACTOR

    // If min height fits in container, scale up to fill the space
    let finalRowHeight = baseRowHeight
    let scroll = false

    if (minTotalHeight < containerHeight) {
      // Scale up to fill available space
      finalRowHeight = (containerHeight / visible.length) / ROW_SPACING_FACTOR
    } else if (baseRowHeight < MIN_ROW_HEIGHT) {
      // Use minimum height and enable scrolling
      finalRowHeight = MIN_ROW_HEIGHT
      scroll = true
    } else {
      // Use calculated height and check if scrolling is needed
      const totalHeight = visible.length * baseRowHeight * ROW_SPACING_FACTOR
      scroll = totalHeight > containerHeight
    }

    return { visibleAgents: visible, rowHeight: finalRowHeight, needsScroll: scroll }
  }, [agents, zoomY, containerHeight])

  // Generate time markers based on X-axis zoom
  const timeMarkers = useMemo(() => {
    if (duration === 0) return []
    const markers = []
    const visibleDuration = (zoomX.end - zoomX.start) * duration

    for (let i = 0; i <= 10; i++) {
      const fraction = i / 10
      const time = minTime + (zoomX.start * duration) + (visibleDuration * fraction)
      markers.push({ fraction, time })
    }
    return markers
  }, [minTime, duration, zoomX])

  // Measure container height (use ResizeObserver for better detection)
  useEffect(() => {
    const measureHeight = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect()
        const newHeight = rect.height
        console.log('Container height measured:', newHeight)
        setContainerHeight(newHeight)
      }
    }

    measureHeight()

    // Use ResizeObserver for better height tracking
    const resizeObserver = new ResizeObserver(() => {
      measureHeight()
    })

    if (containerRef.current) {
      resizeObserver.observe(containerRef.current)
    }

    window.addEventListener('resize', measureHeight)

    return () => {
      resizeObserver.disconnect()
      window.removeEventListener('resize', measureHeight)
    }
  }, [])

  // Debug: log when height or agents change
  useEffect(() => {
    console.log('Container height:', containerHeight, 'Visible agents:', visibleAgents.length, 'Row height:', rowHeight, 'Needs scroll:', needsScroll)
  }, [containerHeight, visibleAgents.length, rowHeight, needsScroll])

  // Calculate position and width for bars based on X-axis zoom
  const getBarStyle = (request) => {
    const fullLeft = (request.timestamp - minTime) / duration
    const fullWidth = request.duration / duration

    const visibleDuration = zoomX.end - zoomX.start
    const relativeLeft = (fullLeft - zoomX.start) / visibleDuration
    const relativeWidth = fullWidth / visibleDuration

    return {
      left: `${relativeLeft * 100}%`,
      width: `${Math.max(relativeWidth * 100, 0.5)}%`,
      background: request.agentType.color
    }
  }

  // Helper: Calculate Y position for an agent, even if outside visible range
  const getAgentYPosition = (agentId, fullAgentIdx) => {
    // First check if agent is in visible range
    const visibleIdx = visibleAgents.findIndex(a => a.agent_id === agentId)

    if (visibleIdx !== -1) {
      // Agent is visible - return its Y position
      return {
        y: (visibleIdx + 0.5) * (rowHeight * 1.2) + rowHeight * 0.3,
        isVisible: true
      }
    }

    // Agent is not visible - calculate position relative to visible range
    const totalAgents = agents.length
    const visibleStartIdx = Math.floor(zoomY.start * totalAgents)
    const visibleHeight = visibleAgents.length * rowHeight * 1.2

    if (fullAgentIdx < visibleStartIdx) {
      // Agent is above visible range - position at top edge
      return {
        y: -rowHeight * 0.5,
        isVisible: false
      }
    } else {
      // Agent is below visible range - position at bottom edge
      return {
        y: visibleHeight + rowHeight * 0.5,
        isVisible: false
      }
    }
  }

  // Calculate SVG path for spawn arrow (bezier S-curve)
  const getSpawnArrowPath = (edge, containerWidth) => {
    // Convert time to X position (0-1 range based on zoom)
    const sourceXFull = (edge.sourceX - minTime) / duration
    const targetXFull = (edge.targetX - minTime) / duration

    const visibleDuration = zoomX.end - zoomX.start
    const sourceXRel = (sourceXFull - zoomX.start) / visibleDuration
    const targetXRel = (targetXFull - zoomX.start) / visibleDuration

    const x1 = sourceXRel * containerWidth
    const x2 = targetXRel * containerWidth

    // Get Y positions (works for both visible and non-visible agents)
    const sourcePos = getAgentYPosition(edge.sourceAgentId, edge.sourceIdx)
    const targetPos = getAgentYPosition(edge.targetAgentId, edge.targetIdx)

    const y1 = sourcePos.y
    const y2 = targetPos.y

    // Bezier S-curve control points
    const dx = x2 - x1
    const dy = y2 - y1

    // Minimum horizontal offset for control points (in pixels)
    const minHorizontalOffset = 100

    // When dx is small, use a forward-backward-forward curve
    // Otherwise use a simple S-curve
    let path
    if (Math.abs(dx) < minHorizontalOffset * 2) {
      // Small horizontal distance: use forward-backward-forward curve
      // This creates a more horizontal S-shape that looks better
      const forwardOffset = minHorizontalOffset
      const backwardOffset = -minHorizontalOffset

      // First control point: go forward (right) from source
      const cx1 = x1 + forwardOffset
      const cy1 = y1 + dy * 0.25

      // Second control point: go backward (left) before target
      const cx2 = x2 + backwardOffset
      const cy2 = y2 - dy * 0.25

      path = `M ${x1} ${y1} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${x2} ${y2}`
    } else {
      // Normal case: simple S-curve with control points at 50% of horizontal distance
      const controlPointOffset = Math.abs(dx) * 0.5

      const cx1 = x1 + controlPointOffset
      const cy1 = y1
      const cx2 = x2 - controlPointOffset
      const cy2 = y2

      path = `M ${x1} ${y1} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${x2} ${y2}`
    }

    return {
      path,
      x1, y1, x2, y2,
      sourceVisible: sourcePos.isVisible,
      targetVisible: targetPos.isVisible
    }
  }

  // Calculate SVG path for request sequence arrow (horizontal line within same agent)
  const getRequestSequenceArrowPath = (edge, containerWidth) => {
    // Convert time to X position (0-1 range based on zoom)
    const sourceXFull = (edge.sourceX - minTime) / duration
    const targetXFull = (edge.targetX - minTime) / duration

    const visibleDuration = zoomX.end - zoomX.start
    const sourceXRel = (sourceXFull - zoomX.start) / visibleDuration
    const targetXRel = (targetXFull - zoomX.start) / visibleDuration

    const x1 = sourceXRel * containerWidth
    const x2 = targetXRel * containerWidth

    // Get Y position (works for both visible and non-visible agents)
    const agentPos = getAgentYPosition(edge.agentId, edge.agentIdx)
    const y = agentPos.y + rowHeight * 0.2  // Slightly below center

    // Simple horizontal line with slight curve
    const midX = (x1 + x2) / 2
    const curveOffset = 5 // Small vertical offset for curve
    const path = `M ${x1} ${y} Q ${midX} ${y + curveOffset}, ${x2} ${y}`

    return {
      path,
      x1, y, x2, y,
      isVisible: agentPos.isVisible
    }
  }

  // Build tooltip content for a request
  const buildTooltip = (agent, request, reqIdx) => {
    const lines = [
      `Agent: ${agent.agent_id}`,
      `Request ${reqIdx + 1}/${agent.requests.length}`,
      `Type: ${request.agentType.label}`,
      `Time: ${new Date(request.timestamp).toLocaleString()}`,
      `Duration: ${request.duration.toFixed(0)}ms`,
      `Status: ${request.status}`
    ]

    if (request.log.tool_info?.count > 0) {
      lines.push(`Tools: ${request.log.tool_info.tool_names.join(', ')}`)
    }

    if (request.log.has_subagent_spawns) {
      lines.push(`Subagents: ${request.log.subagent_count}`)
    }

    if (request.log.has_errors) {
      lines.push(`Errors: ${request.log.tool_errors}`)
    }

    if (request.log.stop_reason) {
      lines.push(`Stop: ${request.log.stop_reason}`)
    }

    const usage = request.log.response?.body?.usage
    if (usage) {
      if (usage.input_tokens !== undefined && usage.output_tokens !== undefined) {
        lines.push(`Tokens: ${usage.input_tokens + usage.output_tokens} (${usage.input_tokens} in + ${usage.output_tokens} out)`)
      } else if (usage.total_tokens !== undefined) {
        lines.push(`Tokens: ${usage.total_tokens}`)
      }
    }

    return lines.join('\n')
  }

  // Handle mouse down to start drag selection
  const handleMouseDown = (e) => {
    if (e.button !== 0) return // Only left click
    const rect = timelineRef.current.getBoundingClientRect()
    const x = (e.clientX - rect.left) / rect.width
    const y = (e.clientY - rect.top) / rect.height
    setDragStart({ x, y })
    setDragCurrent({ x, y })
    setIsDragging(true)
  }

  // Handle mouse move during drag
  const handleMouseMove = (e) => {
    if (!isDragging || !dragStart) return

    const rect = timelineRef.current.getBoundingClientRect()
    const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height))
    setDragCurrent({ x, y })
  }

  // Handle mouse up to end drag
  const handleMouseUp = () => {
    if (!isDragging || !dragStart || !dragCurrent) {
      setIsDragging(false)
      setDragStart(null)
      setDragCurrent(null)
      return
    }

    // Calculate new zoom ranges from the drag selection
    const dragStartX = Math.min(dragStart.x, dragCurrent.x)
    const dragEndX = Math.max(dragStart.x, dragCurrent.x)
    const dragStartY = Math.min(dragStart.y, dragCurrent.y)
    const dragEndY = Math.max(dragStart.y, dragCurrent.y)

    // Only zoom if there's a meaningful selection (at least 1% of visible range)
    if (dragEndX - dragStartX > 0.01 || dragEndY - dragStartY > 0.01) {
      // X-axis zoom (time)
      if (dragEndX - dragStartX > 0.01) {
        const currentVisibleDurationX = zoomX.end - zoomX.start
        const newStartX = zoomX.start + (dragStartX * currentVisibleDurationX)
        const newEndX = zoomX.start + (dragEndX * currentVisibleDurationX)
        setZoomX({ start: newStartX, end: newEndX })
      }

      // Y-axis zoom (agents)
      if (dragEndY - dragStartY > 0.01) {
        const currentVisibleDurationY = zoomY.end - zoomY.start
        const newStartY = zoomY.start + (dragStartY * currentVisibleDurationY)
        const newEndY = zoomY.start + (dragEndY * currentVisibleDurationY)
        setZoomY({ start: newStartY, end: newEndY })
      }
    }

    setIsDragging(false)
    setDragStart(null)
    setDragCurrent(null)
  }

  // Handle double click to reset zoom
  const handleDoubleClick = () => {
    setZoomX({ start: 0, end: 1 })
    setZoomY({ start: 0, end: 1 })
  }

  // Handle mouse wheel to zoom in/out centered on cursor
  const handleWheel = (e) => {
    e.preventDefault()

    const rect = timelineRef.current.getBoundingClientRect()

    // Get cursor position as fraction (0-1) of visible area
    const cursorXFraction = (e.clientX - rect.left) / rect.width
    const cursorYFraction = (e.clientY - rect.top) / rect.height

    // Zoom factor: scroll up = zoom in, scroll down = zoom out
    const zoomFactor = e.deltaY > 0 ? 1.15 : 0.85

    // Check if shift is held for X-only zoom, or ctrl for Y-only zoom
    const zoomXAxis = !e.ctrlKey
    const zoomYAxis = !e.shiftKey

    // Zoom X-axis (time)
    if (zoomXAxis) {
      const currentRange = zoomX.end - zoomX.start
      const newRange = Math.min(1, currentRange * zoomFactor)

      // Calculate cursor position in full data space
      const cursorInData = zoomX.start + cursorXFraction * currentRange

      // Calculate new start/end keeping cursor position fixed
      let newStart = cursorInData - cursorXFraction * newRange
      let newEnd = cursorInData + (1 - cursorXFraction) * newRange

      // Clamp to valid range [0, 1]
      if (newStart < 0) {
        newEnd = Math.min(1, newEnd - newStart)
        newStart = 0
      }
      if (newEnd > 1) {
        newStart = Math.max(0, newStart - (newEnd - 1))
        newEnd = 1
      }

      setZoomX({ start: newStart, end: newEnd })
    }

    // Zoom Y-axis (agents)
    if (zoomYAxis) {
      const currentRange = zoomY.end - zoomY.start
      const newRange = Math.min(1, currentRange * zoomFactor)

      // Calculate cursor position in full data space
      const cursorInData = zoomY.start + cursorYFraction * currentRange

      // Calculate new start/end keeping cursor position fixed
      let newStart = cursorInData - cursorYFraction * newRange
      let newEnd = cursorInData + (1 - cursorYFraction) * newRange

      // Clamp to valid range [0, 1]
      if (newStart < 0) {
        newEnd = Math.min(1, newEnd - newStart)
        newStart = 0
      }
      if (newEnd > 1) {
        newStart = Math.max(0, newStart - (newEnd - 1))
        newEnd = 1
      }

      setZoomY({ start: newStart, end: newEnd })
    }
  }

  // Handle bar click to show messages
  const handleBarClick = (e, agent, request) => {
    e.stopPropagation()
    setSelectedAgent(agent)
    setSelectedRequest(request)
    setModalOpen(true)
  }

  // Close modal
  const closeModal = () => {
    setModalOpen(false)
    setSelectedRequest(null)
    setSelectedAgent(null)
  }

  // Sync scroll between labels and timeline when scrolling is enabled
  const handleTimelineScroll = (e) => {
    if (labelsRef.current && needsScroll) {
      labelsRef.current.scrollTop = e.target.scrollTop
    }
  }

  const handleLabelsScroll = (e) => {
    if (timelineRef.current && needsScroll) {
      timelineRef.current.scrollTop = e.target.scrollTop
    }
  }

  // Add/remove event listeners for drag
  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove)
      window.addEventListener('mouseup', handleMouseUp)
      return () => {
        window.removeEventListener('mousemove', handleMouseMove)
        window.removeEventListener('mouseup', handleMouseUp)
      }
    }
  }, [isDragging, dragStart, dragCurrent, zoomX, zoomY])

  // Calculate selection rectangle for visual feedback
  const selectionRect = useMemo(() => {
    if (!isDragging || !dragStart || !dragCurrent) return null

    const startX = Math.min(dragStart.x, dragCurrent.x)
    const endX = Math.max(dragStart.x, dragCurrent.x)
    const startY = Math.min(dragStart.y, dragCurrent.y)
    const endY = Math.max(dragStart.y, dragCurrent.y)

    return {
      left: `${startX * 100}%`,
      width: `${(endX - startX) * 100}%`,
      top: `${startY * 100}%`,
      height: `${(endY - startY) * 100}%`
    }
  }, [isDragging, dragStart, dragCurrent])

  if (agents.length === 0) {
    return (
      <div className="agent-gantt-panel">
        <div className="gantt-empty">No agent data available</div>
      </div>
    )
  }

  const totalRequests = agents.reduce((sum, a) => sum + a.requests.length, 0)

  return (
    <div className="agent-gantt-panel">
      <div className="gantt-header">
        <div className="gantt-title">Agent Instance Timeline</div>
        <div className="gantt-stats">
          {agents.length} agents | {totalRequests} requests | {formatDuration(duration)} total
        </div>
        <div className="gantt-hint">
          Scroll to zoom | Shift: X-only | Ctrl: Y-only | Drag to select | Double-click to reset
        </div>
      </div>

      <div className="gantt-container" ref={containerRef}>
        <div
          className="gantt-labels"
          ref={labelsRef}
          onScroll={handleLabelsScroll}
          style={{
            overflowY: needsScroll ? 'auto' : 'hidden',
            height: '100%',
            paddingTop: `${rowHeight * 0.3}px`,
            paddingBottom: `${rowHeight * 0.3}px`
          }}
        >
          {visibleAgents.map((agent) => (
            <div
              key={agent.agent_id}
              className={`gantt-label ${selectedAgentId === agent.agent_id ? 'selected' : ''}`}
              onClick={() => setSelectedAgentId(agent.agent_id)}
              style={{
                borderLeftColor: agent.agentType.color,
                height: `${rowHeight}px`,
                minHeight: `${rowHeight}px`,
                marginBottom: `${rowHeight * 0.2}px`
              }}
            >
              <span className="label-id">{agent.agent_id}</span>
              <span className="label-count">{agent.requests.length}</span>
            </div>
          ))}
        </div>

        <div
          className="gantt-timeline"
          ref={timelineRef}
          onMouseDown={handleMouseDown}
          onDoubleClick={handleDoubleClick}
          onWheel={handleWheel}
          onScroll={handleTimelineScroll}
          style={{
            overflowY: needsScroll ? 'auto' : 'hidden',
            height: '100%'
          }}
        >
          <div className="gantt-axis">
            {timeMarkers.map((marker, idx) => (
              <div key={idx} className="axis-marker" style={{ left: `${marker.fraction * 100}%` }}>
                <span className="axis-label">{new Date(marker.time).toLocaleTimeString()}</span>
              </div>
            ))}
          </div>

          <div
            className="gantt-rows"
            ref={rowsRef}
            style={{
              paddingTop: `${rowHeight * 0.5}px`,
              paddingBottom: `${rowHeight * 0.5}px`,
              paddingLeft: '10px',
              paddingRight: '10px',
              minHeight: `${visibleAgents.length > 0 ? (visibleAgents.length * rowHeight + (visibleAgents.length - 1) * rowHeight * 0.2 + rowHeight * 1.0) : 0}px`,
              width: 'calc(100% - 20px)'
            }}
          >
            <div className="gantt-grid">
              {timeMarkers.map((marker, idx) => (
                <div key={idx} className="grid-line" style={{ left: `${marker.fraction * 100}%` }} />
              ))}
            </div>

            {visibleAgents.map((agent, agentIdx) => (
              <div
                key={agent.agent_id}
                className="gantt-row"
                style={{
                  height: `${rowHeight}px`,
                  marginBottom: agentIdx < visibleAgents.length - 1 ? `${rowHeight * 0.2}px` : '0'
                }}
              >
                {agent.requests.map((request, reqIdx) => {
                  const barStyle = getBarStyle(request)

                  return (
                    <div
                      key={reqIdx}
                      className="gantt-bar"
                      style={{
                        ...barStyle,
                        height: `${rowHeight * 0.8}px`,
                        top: `${rowHeight * 0.1}px`
                      }}
                      onClick={(e) => handleBarClick(e, agent, request)}
                      data-tooltip-id="gantt-tooltip"
                      data-tooltip-content={buildTooltip(agent, request, reqIdx)}
                    />
                  )
                })}
              </div>
            ))}

            {/* Spawn arrows overlay */}
            {timelineRef.current && rowsRef.current && (
              <svg
                className="gantt-arrows"
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: `${visibleAgents.length * rowHeight * 1.2 + rowHeight * 0.6}px`,
                  pointerEvents: 'none',
                  zIndex: 5
                }}
              >
                <defs>
                  <marker
                    id="arrow-spawn"
                    viewBox="0 0 10 10"
                    refX="9"
                    refY="5"
                    markerWidth="6"
                    markerHeight="6"
                    orient="auto"
                  >
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#888" />
                  </marker>
                  <marker
                    id="arrow-sequence"
                    viewBox="0 0 10 10"
                    refX="9"
                    refY="5"
                    markerWidth="5"
                    markerHeight="5"
                    orient="auto"
                  >
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#3b82f6" />
                  </marker>
                  <marker
                    id="arrow-content"
                    viewBox="0 0 10 10"
                    refX="9"
                    refY="5"
                    markerWidth="6"
                    markerHeight="6"
                    orient="auto"
                  >
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#8b5cf6" />
                  </marker>
                </defs>
                {/* Spawn edges */}
                {spawnEdges.map((edge, idx) => {
                  const containerWidth = rowsRef.current.clientWidth
                  const pathData = getSpawnArrowPath(edge, containerWidth)

                  if (!pathData) return null

                  // Build tooltip content based on spawn method
                  let tooltipContent = `Spawn: ${edge.sourceAgentId} → ${edge.targetAgentId}\n`
                  if (edge.spawn_method === 'task') {
                    tooltipContent += `Method: Task\nTask ID: ${edge.spawned_by_task_id || 'N/A'}\n`
                  } else if (edge.spawn_method === 'tool_call') {
                    tooltipContent += `Method: Tool Call\nTool: ${edge.tool_name || 'N/A'}\nTool Use ID: ${edge.spawned_by_tool_use_id || 'N/A'}\n`
                  } else {
                    tooltipContent += `Task/Tool ID: ${edge.spawned_by_task_id || edge.spawned_by_tool_use_id || 'N/A'}\n`
                  }
                  tooltipContent += `Confidence: ${edge.confidence || 'N/A'}`

                  return (
                    <path
                      key={`spawn-${idx}`}
                      d={pathData.path}
                      stroke="#888"
                      strokeWidth="1.5"
                      fill="none"
                      markerEnd="url(#arrow-spawn)"
                      style={{ pointerEvents: 'stroke' }}
                      data-tooltip-id="gantt-tooltip"
                      data-tooltip-content={tooltipContent}
                    />
                  )
                })}

                {/* Request sequence edges */}
                {requestSequenceEdges.map((edge, idx) => {
                  const containerWidth = rowsRef.current.clientWidth
                  const pathData = getRequestSequenceArrowPath(edge, containerWidth)

                  if (!pathData) return null

                  return (
                    <path
                      key={`seq-${idx}`}
                      d={pathData.path}
                      stroke="#3b82f6"
                      strokeWidth="1.0"
                      fill="none"
                      markerEnd="url(#arrow-sequence)"
                      style={{ pointerEvents: 'stroke', opacity: 0.6 }}
                      data-tooltip-id="gantt-tooltip"
                      data-tooltip-content={`Sequence: req ${edge.sourceRequestId} → ${edge.targetRequestId}\nGap: ${edge.time_gap_ms}ms\nAgent: ${edge.agentId}`}
                    />
                  )
                })}

                {/* Content reuse edges */}
                {contentReuseEdges.map((edge, idx) => {
                  const containerWidth = rowsRef.current.clientWidth
                  const pathData = getSpawnArrowPath(edge, containerWidth)

                  if (!pathData) return null

                  return (
                    <path
                      key={`content-${idx}`}
                      d={pathData.path}
                      stroke="#8b5cf6"
                      strokeWidth="1.5"
                      strokeDasharray="5,5"
                      fill="none"
                      markerEnd="url(#arrow-content)"
                      style={{ pointerEvents: 'stroke', opacity: 0.7 }}
                      data-tooltip-id="gantt-tooltip"
                      data-tooltip-content={`Content Reuse: ${edge.sourceAgentId} → ${edge.targetAgentId}\nHash: ${edge.content_hash}\nConfidence: ${edge.confidence}`}
                    />
                  )
                })}
              </svg>
            )}

            {/* Selection rectangle during drag */}
            {selectionRect && (
              <div className="gantt-selection" style={selectionRect} />
            )}
          </div>
        </div>
      </div>

      {/* Tooltip */}
      <Tooltip
        id="gantt-tooltip"
        place="top"
        style={{
          backgroundColor: '#1a1a1a',
          color: '#e0e0e0',
          fontSize: '11px',
          padding: '8px 12px',
          borderRadius: '4px',
          border: '1px solid #333',
          whiteSpace: 'pre-line',
          zIndex: 1000
        }}
      />

      {/* Messages Modal */}
      <MessagesModal
        isOpen={modalOpen}
        onClose={closeModal}
        request={selectedRequest}
        agent={selectedAgent}
        entitiesData={entitiesData}
      />
    </div>
  )
}

export default AgentGanttPanel

