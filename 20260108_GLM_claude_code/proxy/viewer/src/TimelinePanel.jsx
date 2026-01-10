import { useState, useMemo, useRef, useEffect } from 'react'
import { Tooltip } from 'react-tooltip'
import './TimelinePanel.css'

// Agent type definitions (matching log_classifier.py)
const AGENT_TYPES = {
  "file_path_extractor": {
    "label": "File Path Extractor",
    "color": "#10b981",
  },
  "file_search": {
    "label": "File Search Specialist",
    "color": "#3b82f6",
  },
  "bash_processor": {
    "label": "Bash Command Processor",
    "color": "#f59e0b",
  },
  "summarizer": {
    "label": "Conversation Summarizer",
    "color": "#8b5cf6",
  },
  "architect": {
    "label": "Software Architect",
    "color": "#ec4899",
  },
  "topic_detector": {
    "label": "Topic Change Detector",
    "color": "#06b6d4",
  },
  "main_agent": {
    "label": "Main Interactive Agent",
    "color": "#ef4444",
  },
  "unknown": {
    "label": "Unknown Agent",
    "color": "#6b7280",
  }
}

function TimelinePanel({ logs, onSelectLog, selectedLogIndex, colorByAgent = true }) {
  const [zoomRange, setZoomRange] = useState({ start: 0, end: 1 }) // 0 to 1 representing the visible portion
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState(null)
  const [dragCurrent, setDragCurrent] = useState(null)
  const containerRef = useRef(null)
  const timelineContentRef = useRef(null)
  const timelineRowsRef = useRef(null)

  // Calculate timeline data
  const timelineData = useMemo(() => {
    if (logs.length === 0) return { rows: [], minTime: 0, maxTime: 0, duration: 0 }

    // Parse timestamps and calculate start/end times
    // Logs come in newest-first from parent, which is correct for top-to-bottom rendering
    // X-axis positioning uses (startTime - minTime), so newer timestamps automatically appear on the right
    const items = logs.map((log, idx) => {
      const startTime = new Date(log.timestamp).getTime()
      const duration = log.response?.duration_ms || 0
      const endTime = startTime + duration
      const model = log.body?.model || 'unknown'
      const status = log.response?.status || 'pending'
      const isError = status >= 400
      const usage = log.response?.body?.usage
      const agentType = log.agent_type
      const toolInfo = log.tool_info || {}
      const hasSubagents = log.has_subagent_spawns || false
      const subagentCount = log.subagent_count || 0
      const hasErrors = log.has_errors || false
      const toolErrors = log.tool_errors || 0
      const stopReason = log.stop_reason

      let totalTokens = 'no tokens'
      if (usage) {
        if (usage.input_tokens !== undefined && usage.output_tokens !== undefined) {
          totalTokens = `${usage.input_tokens + usage.output_tokens} tokens (${usage.input_tokens} in + ${usage.output_tokens} out)`
        } else if (usage.total_tokens !== undefined) {
          totalTokens = `${usage.total_tokens} tokens`
        }
      }
      const durationText = duration !== undefined ? `${duration.toFixed(0)}ms` : 'no duration'
      const timestamp = new Date(log.timestamp).toLocaleString()

      // Build enhanced tooltip
      let tooltipLines = [
        timestamp,
        model,
        `Status: ${status}`
      ]
      if (agentType) {
        tooltipLines.push(`Agent: ${agentType.label}`)
      }
      if (stopReason) {
        tooltipLines.push(`Stop: ${stopReason}`)
      }
      if (toolInfo.count > 0) {
        tooltipLines.push(`Tools: ${toolInfo.tool_names.join(', ')}`)
      }
      if (hasSubagents) {
        tooltipLines.push(`Subagents: ${subagentCount}`)
      }
      if (hasErrors) {
        tooltipLines.push(`Errors: ${toolErrors}`)
      }
      tooltipLines.push(totalTokens)
      tooltipLines.push(`Duration: ${durationText}`)

      return {
        idx,
        startTime,
        endTime,
        duration,
        model,
        status,
        isError,
        totalTokens,
        durationText,
        timestamp,
        tooltipContent: tooltipLines.join('\n'),
        agentType,
        log,
        key: `${model}-${log.method}-${log.path}` // Group key
      }
    })

    // Find time bounds
    const minTime = Math.min(...items.map(i => i.startTime))
    const maxTime = Math.max(...items.map(i => i.endTime))
    const duration = maxTime - minTime

    // Each item gets its own row (no grouping)
    // Items are already in newest-first order, which renders top-to-bottom correctly
    const rows = items.map(item => [item])

    return { rows, minTime, maxTime, duration }
  }, [logs])

  const { rows, minTime, maxTime, duration } = timelineData

  // Calculate position and width for each item based on zoom range
  const getItemStyle = (item) => {
    // Position in the full timeline (0 to 1)
    const fullLeft = (item.startTime - minTime) / duration
    const fullWidth = item.duration / duration

    // Visible range
    const visibleDuration = zoomRange.end - zoomRange.start

    // Position relative to the visible range
    const relativeLeft = (fullLeft - zoomRange.start) / visibleDuration
    const relativeWidth = fullWidth / visibleDuration

    return {
      left: `${relativeLeft * 100}%`,
      width: `${Math.max(relativeWidth * 100, 0.5)}%` // Minimum 0.5% width for visibility
    }
  }

  // Format time for display
  const formatTime = (ms) => {
    if (ms < 1000) return `${ms.toFixed(0)}ms`
    return `${(ms / 1000).toFixed(2)}s`
  }

  // Format timestamp for axis
  const formatAxisTime = (timestamp) => {
    const elapsed = timestamp - minTime
    return formatTime(elapsed)
  }

  // Generate time axis markers based on visible range
  const timeMarkers = useMemo(() => {
    if (duration === 0) return []

    const markerCount = 10
    const markers = []
    const visibleDuration = (zoomRange.end - zoomRange.start) * duration

    for (let i = 0; i <= markerCount; i++) {
      const position = (i / markerCount) * 100
      const timestamp = minTime + (zoomRange.start * duration) + (visibleDuration * i / markerCount)
      markers.push({ position, timestamp })
    }

    return markers
  }, [minTime, duration, zoomRange])

  // Handle mouse down to start drag selection
  const handleMouseDown = (e) => {
    if (e.button !== 0) return // Only left click
    const rect = timelineContentRef.current.getBoundingClientRect()
    const x = (e.clientX - rect.left) / rect.width
    setDragStart(x)
    setDragCurrent(x)
    setIsDragging(true)
  }

  // Handle mouse move during drag
  const handleMouseMove = (e) => {
    if (!isDragging || dragStart === null) return

    const rect = timelineContentRef.current.getBoundingClientRect()
    const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    setDragCurrent(x)
  }

  // Handle mouse up to end drag
  const handleMouseUp = () => {
    if (!isDragging || dragStart === null || dragCurrent === null) {
      setIsDragging(false)
      setDragStart(null)
      setDragCurrent(null)
      return
    }

    // Calculate new zoom range from the drag selection
    // The drag coordinates are relative to the current visible range (0 to 1 on screen)
    // We need to map them to the actual timeline coordinates
    const dragStartNormalized = Math.min(dragStart, dragCurrent)
    const dragEndNormalized = Math.max(dragStart, dragCurrent)

    // Only zoom if there's a meaningful selection (at least 1% of visible range)
    if (dragEndNormalized - dragStartNormalized > 0.01) {
      // Map the screen coordinates to the actual timeline coordinates
      const currentVisibleDuration = zoomRange.end - zoomRange.start
      const newStart = zoomRange.start + (dragStartNormalized * currentVisibleDuration)
      const newEnd = zoomRange.start + (dragEndNormalized * currentVisibleDuration)

      setZoomRange({ start: newStart, end: newEnd })
    }

    setIsDragging(false)
    setDragStart(null)
    setDragCurrent(null)
  }

  // Handle double click to reset zoom
  const handleDoubleClick = () => {
    setZoomRange({ start: 0, end: 1 })
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
  }, [isDragging, dragStart, dragCurrent])

  // Calculate selection rectangle for visual feedback
  const selectionRect = useMemo(() => {
    if (!isDragging || dragStart === null || dragCurrent === null) return null

    const start = Math.min(dragStart, dragCurrent)
    const end = Math.max(dragStart, dragCurrent)

    // Get the full height of the timeline rows
    const rowsHeight = timelineRowsRef.current?.scrollHeight || 0

    return {
      left: `${start * 100}%`,
      width: `${(end - start) * 100}%`,
      height: rowsHeight > 0 ? `${rowsHeight}px` : '100%'
    }
  }, [isDragging, dragStart, dragCurrent])

  if (logs.length === 0) {
    return (
      <div className="timeline-panel">
        <div className="timeline-empty">No requests to display</div>
      </div>
    )
  }

  // Compute unique agent types present in the logs
  const uniqueAgentTypes = useMemo(() => {
    const agentTypeSet = new Set()
    logs.forEach(log => {
      if (log.agent_type && log.agent_type.name) {
        agentTypeSet.add(log.agent_type.name)
      }
    })
    return Array.from(agentTypeSet).sort()
  }, [logs])

  return (
    <div className="timeline-panel" ref={containerRef}>
      <div className="timeline-header">
        <div className="timeline-title">Request Timeline</div>
        <div className="timeline-hint">
          Drag to zoom | Double-click to reset
        </div>
      </div>

      {/* Color Legend */}
      <div className="timeline-legend">
        {colorByAgent ? (
          // Agent type legend
          <div className="legend-items">
            {uniqueAgentTypes.map(agentName => {
              const agentType = AGENT_TYPES[agentName] || AGENT_TYPES.unknown
              return (
                <div key={agentName} className="legend-item">
                  <div
                    className="legend-color-box"
                    style={{ backgroundColor: agentType.color }}
                  />
                  <span className="legend-label">{agentType.label}</span>
                </div>
              )
            })}
          </div>
        ) : (
          // Success/Error legend
          <div className="legend-items">
            <div className="legend-item">
              <div
                className="legend-color-box"
                style={{ background: 'linear-gradient(90deg, #10b981 0%, #059669 100%)' }}
              />
              <span className="legend-label">Success</span>
            </div>
            <div className="legend-item">
              <div
                className="legend-color-box"
                style={{ background: 'linear-gradient(90deg, #ef4444 0%, #dc2626 100%)' }}
              />
              <span className="legend-label">Error</span>
            </div>
          </div>
        )}
      </div>

      <div className="timeline-axis-container">
        <div className="timeline-axis-pinned">
          {timeMarkers.map((marker, idx) => (
            <div
              key={idx}
              className="time-marker"
              style={{ left: `${marker.position}%` }}
            >
              <div className="marker-label">{formatAxisTime(marker.timestamp)}</div>
            </div>
          ))}
        </div>
      </div>

      <div
        className="timeline-content"
        ref={timelineContentRef}
        onMouseDown={handleMouseDown}
        onDoubleClick={handleDoubleClick}
        style={{ cursor: isDragging ? 'col-resize' : 'crosshair' }}
      >
        <div className="timeline-rows" ref={timelineRowsRef}>
          {/* Grid lines */}
          <div className="timeline-grid">
            {timeMarkers.map((marker, idx) => (
              <div
                key={idx}
                className="grid-line"
                style={{ left: `${marker.position}%` }}
              />
            ))}
          </div>
          {selectionRect && (
            <div
              className="selection-rect"
              style={{
                position: 'absolute',
                top: 0,
                left: selectionRect.left,
                width: selectionRect.width,
                height: selectionRect.height,
                background: 'rgba(59, 130, 246, 0.2)',
                border: '1px solid rgba(59, 130, 246, 0.5)',
                pointerEvents: 'none',
                zIndex: 100
              }}
            />
          )}
          {rows.map((row, rowIdx) => (
            <div key={rowIdx} className="timeline-row">
              {row.map((item) => {
                const itemStyle = getItemStyle(item)

                // Apply agent color if enabled
                if (colorByAgent && item.agentType) {
                  itemStyle.background = item.agentType.color
                } else {
                  // Use default gradient based on error status
                  itemStyle.background = item.isError
                    ? 'linear-gradient(90deg, #ef4444 0%, #dc2626 100%)'
                    : 'linear-gradient(90deg, #10b981 0%, #059669 100%)'
                }

                return (
                  <div
                    key={item.idx}
                    className={`timeline-item ${selectedLogIndex === item.idx ? 'selected' : ''}`}
                    style={itemStyle}
                    onClick={() => onSelectLog(item.idx)}
                    data-tooltip-id="timeline-tooltip"
                    data-tooltip-content={item.tooltipContent}
                  />
                )
              })}
            </div>
          ))}
        </div>
      </div>

      <Tooltip
        id="timeline-tooltip"
        place="top"
        style={{
          backgroundColor: '#1a1a1a',
          color: '#e0e0e0',
          border: '1px solid #333',
          borderRadius: '4px',
          padding: '8px 12px',
          fontSize: '12px',
          whiteSpace: 'pre-line',
          zIndex: 9999
        }}
      />
    </div>
  )
}

export default TimelinePanel

