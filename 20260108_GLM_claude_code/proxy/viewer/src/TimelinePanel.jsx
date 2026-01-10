import { useState, useMemo, useRef, useEffect } from 'react'
import { Tooltip } from 'react-tooltip'
import './TimelinePanel.css'

function TimelinePanel({ logs, onSelectLog, selectedLogIndex }) {
  const [zoom, setZoom] = useState(1)
  const containerRef = useRef(null)

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

  // Calculate position and width for each item
  const getItemStyle = (item) => {
    const left = ((item.startTime - minTime) / duration) * 100
    const width = (item.duration / duration) * 100
    
    return {
      left: `${left}%`,
      width: `${Math.max(width, 0.5)}%` // Minimum 0.5% width for visibility
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

  // Generate time axis markers
  const timeMarkers = useMemo(() => {
    if (duration === 0) return []
    
    const markerCount = 10
    const markers = []
    
    for (let i = 0; i <= markerCount; i++) {
      const position = (i / markerCount) * 100
      const timestamp = minTime + (duration * i / markerCount)
      markers.push({ position, timestamp })
    }
    
    return markers
  }, [minTime, duration])

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
        <div className="timeline-controls">
          <button onClick={() => setZoom(z => Math.max(0.5, z - 0.25))}>-</button>
          <span className="zoom-level">{(zoom * 100).toFixed(0)}%</span>
          <button onClick={() => setZoom(z => Math.min(3, z + 0.25))}>+</button>
          <button onClick={() => setZoom(1)}>Reset</button>
        </div>
      </div>

      <div className="timeline-content" style={{ transform: `scaleX(${zoom})`, transformOrigin: 'left' }}>
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

        <div className="timeline-rows">
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

