import { useState, useMemo, useRef, useEffect } from 'react'
import { Tooltip } from 'react-tooltip'
import './TimelinePanel.css'

function TimelinePanel({ logs, onSelectLog, selectedLogIndex }) {
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
    const items = logs.map((log, idx) => {
      const startTime = new Date(log.timestamp).getTime()
      const duration = log.response?.duration_ms || 0
      const endTime = startTime + duration
      const model = log.body?.model || 'unknown'
      const status = log.response?.status || 'pending'
      const isError = status >= 400

      return {
        idx,
        startTime,
        endTime,
        duration,
        model,
        status,
        isError,
        log,
        key: `${model}-${log.method}-${log.path}` // Group key
      }
    })

    // Find time bounds
    const minTime = Math.min(...items.map(i => i.startTime))
    const maxTime = Math.max(...items.map(i => i.endTime))
    const duration = maxTime - minTime

    // Group items into rows (same key, non-overlapping)
    const rows = []
    const rowsByKey = new Map()

    items.forEach(item => {
      // Try to find an existing row with the same key where this item fits
      let placed = false
      
      if (rowsByKey.has(item.key)) {
        const candidateRows = rowsByKey.get(item.key)
        
        for (const rowIdx of candidateRows) {
          const row = rows[rowIdx]
          const lastItem = row[row.length - 1]
          
          // Check if this item doesn't overlap with the last item in the row
          if (item.startTime >= lastItem.endTime) {
            row.push(item)
            placed = true
            break
          }
        }
      }

      // If not placed, create a new row
      if (!placed) {
        const newRowIdx = rows.length
        rows.push([item])
        
        if (!rowsByKey.has(item.key)) {
          rowsByKey.set(item.key, [])
        }
        rowsByKey.get(item.key).push(newRowIdx)
      }
    })

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

  return (
    <div className="timeline-panel" ref={containerRef}>
      <div className="timeline-header">
        <div className="timeline-title">Request Timeline</div>
        <div className="timeline-hint">
          Drag to zoom | Double-click to reset
        </div>
      </div>

      <div
        className="timeline-content"
        ref={timelineContentRef}
        onMouseDown={handleMouseDown}
        onDoubleClick={handleDoubleClick}
        style={{ cursor: isDragging ? 'col-resize' : 'crosshair' }}
      >
        <div className="timeline-axis">
          {timeMarkers.map((marker, idx) => (
            <div
              key={idx}
              className="time-marker"
              style={{ left: `${marker.position}%` }}
            >
              <div className="marker-line" />
              <div className="marker-label">{formatAxisTime(marker.timestamp)}</div>
            </div>
          ))}
        </div>

        <div className="timeline-rows" ref={timelineRowsRef}>
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
              {row.map((item) => (
                <div
                  key={item.idx}
                  className={`timeline-item ${item.isError ? 'error' : 'success'} ${selectedLogIndex === item.idx ? 'selected' : ''}`}
                  style={getItemStyle(item)}
                  onClick={() => onSelectLog(item.idx)}
                  data-tooltip-id="timeline-tooltip"
                  data-tooltip-content={`${item.model}\nStatus: ${item.status}\nDuration: ${formatTime(item.duration)}\nStart: ${new Date(item.startTime).toLocaleTimeString()}`}
                />
              ))}
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

