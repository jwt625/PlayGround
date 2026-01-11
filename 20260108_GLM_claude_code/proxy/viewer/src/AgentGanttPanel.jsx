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
  const timelineRef = useRef(null)
  const rowsRef = useRef(null)

  // Process agent data
  const agentData = useMemo(() => {
    if (!entitiesData || !logs || logs.length === 0) {
      return { agents: [], minTime: 0, maxTime: 0, duration: 0 }
    }

    const agentInstances = entitiesData.entities?.agent_instances || []
    if (agentInstances.length === 0) {
      return { agents: [], minTime: 0, maxTime: 0, duration: 0 }
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
        firstTimestamp: requests[0].timestamp,
        agentType: requests[0].agentType
      }
    }).filter(a => a !== null)

    // Sort by first timestamp (earliest first for proper Gantt chart)
    agents.sort((a, b) => a.firstTimestamp - b.firstTimestamp)

    // Calculate time bounds
    const allTimestamps = agents.flatMap(a => a.requests.map(r => r.timestamp))
    const allEndTimes = agents.flatMap(a => a.requests.map(r => r.endTime))
    const minTime = Math.min(...allTimestamps)
    const maxTime = Math.max(...allEndTimes)
    const duration = maxTime - minTime

    return { agents, minTime, maxTime, duration }
  }, [entitiesData, logs])

  const { agents, minTime, duration } = agentData

  // Format duration
  const formatDuration = (ms) => {
    if (ms < 1000) return `${ms.toFixed(0)}ms`
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
    return `${(ms / 60000).toFixed(1)}m`
  }

  // Calculate visible agents and row height based on Y-axis zoom
  const { visibleAgents, rowHeight } = useMemo(() => {
    if (agents.length === 0) return { visibleAgents: [], rowHeight: 2.7 }

    const startIdx = Math.floor(zoomY.start * agents.length)
    const endIdx = Math.ceil(zoomY.end * agents.length)
    const visible = agents.slice(startIdx, endIdx)

    // Calculate row height based on zoom level
    // Base height is 2.7px, scale inversely with the zoom range
    const zoomFactor = 1 / (zoomY.end - zoomY.start)
    const calculatedHeight = 2.7 * zoomFactor

    return { visibleAgents: visible, rowHeight: calculatedHeight }
  }, [agents, zoomY])

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
          Drag to zoom | Double-click to reset
        </div>
      </div>

      <div className="gantt-container">
        <div className="gantt-labels">
          {visibleAgents.map((agent) => (
            <div
              key={agent.agent_id}
              className={`gantt-label ${selectedAgentId === agent.agent_id ? 'selected' : ''}`}
              onClick={() => setSelectedAgentId(agent.agent_id)}
              style={{
                borderLeftColor: agent.agentType.color,
                height: `${rowHeight}px`,
                minHeight: `${rowHeight}px`
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
        >
          <div className="gantt-axis">
            {timeMarkers.map((marker, idx) => (
              <div key={idx} className="axis-marker" style={{ left: `${marker.fraction * 100}%` }}>
                <span className="axis-label">{new Date(marker.time).toLocaleTimeString()}</span>
              </div>
            ))}
          </div>

          <div className="gantt-rows" ref={rowsRef}>
            <div className="gantt-grid">
              {timeMarkers.map((marker, idx) => (
                <div key={idx} className="grid-line" style={{ left: `${marker.fraction * 100}%` }} />
              ))}
            </div>

            {visibleAgents.map((agent) => (
              <div
                key={agent.agent_id}
                className="gantt-row"
                style={{
                  height: `${rowHeight}px`,
                  marginBottom: `${rowHeight * 0.2}px`
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

