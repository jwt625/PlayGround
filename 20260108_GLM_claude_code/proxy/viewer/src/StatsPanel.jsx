import { useMemo, useState } from 'react'
import './StatsPanel.css'

// Format milliseconds to hh:mm:ss (omit zero hours)
function formatTime(ms) {
  const totalSeconds = Math.floor(ms / 1000)
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const seconds = totalSeconds % 60

  const parts = []
  if (hours > 0) {
    parts.push(`${hours}h`)
  }
  if (minutes > 0 || hours > 0) {
    parts.push(`${minutes}m`)
  }
  parts.push(`${seconds}s`)

  return parts.join(' ')
}

// Agent type definitions (matching log_classifier.py)
const AGENT_TYPES = {
  "file_path_extractor": { label: "File Path Extractor", color: "#10b981" },
  "file_search": { label: "File Search Specialist", color: "#3b82f6" },
  "bash_processor": { label: "Bash Command Processor", color: "#f59e0b" },
  "summarizer": { label: "Conversation Summarizer", color: "#8b5cf6" },
  "architect": { label: "Software Architect", color: "#ec4899" },
  "topic_detector": { label: "Topic Change Detector", color: "#06b6d4" },
  "main_agent": { label: "Main Interactive Agent", color: "#ef4444" },
  "unknown": { label: "Unknown Agent", color: "#6b7280" }
}

// Color palettes for pie charts
const TOOL_COLORS = [
  "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
  "#ec4899", "#06b6d4", "#14b8a6", "#f97316", "#84cc16"
]

const STOP_REASON_COLORS = [
  "#8b5cf6", "#ec4899", "#f59e0b", "#10b981", "#3b82f6"
]

// Simple SVG Pie Chart Component with tooltip
function PieChart({ data, showLegend = true, valueFormatter = (v) => v.toLocaleString() }) {
  const [tooltip, setTooltip] = useState(null)

  if (!data || data.length === 0) {
    return <div className="pie-empty">No data</div>
  }

  const total = data.reduce((sum, item) => sum + item.value, 0)
  if (total === 0) {
    return <div className="pie-empty">No data</div>
  }

  const size = 300
  const center = size / 2
  const radius = size / 2 - 10

  let currentAngle = -90 // Start from top

  const slices = data.map((item, idx) => {
    const percentage = (item.value / total) * 100
    const angle = (item.value / total) * 360
    const startAngle = currentAngle
    const endAngle = currentAngle + angle
    currentAngle = endAngle

    // Convert angles to radians
    const startRad = (startAngle * Math.PI) / 180
    const endRad = (endAngle * Math.PI) / 180

    // Calculate arc path
    const x1 = center + radius * Math.cos(startRad)
    const y1 = center + radius * Math.sin(startRad)
    const x2 = center + radius * Math.cos(endRad)
    const y2 = center + radius * Math.sin(endRad)

    const largeArc = angle > 180 ? 1 : 0

    const pathData = [
      `M ${center} ${center}`,
      `L ${x1} ${y1}`,
      `A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2}`,
      'Z'
    ].join(' ')

    return {
      path: pathData,
      color: item.color,
      label: item.label,
      value: item.value,
      percentage: percentage.toFixed(1)
    }
  })

  const handleMouseEnter = (slice, event) => {
    setTooltip({
      label: slice.label,
      value: valueFormatter(slice.value),
      percentage: slice.percentage,
      x: event.clientX,
      y: event.clientY
    })
  }

  const handleMouseMove = (slice, event) => {
    setTooltip({
      label: slice.label,
      value: valueFormatter(slice.value),
      percentage: slice.percentage,
      x: event.clientX,
      y: event.clientY
    })
  }

  const handleMouseLeave = () => {
    setTooltip(null)
  }

  return (
    <div className="pie-chart-wrapper">
      <svg className="pie-chart-svg" viewBox={`0 0 ${size} ${size}`}>
        {slices.map((slice, idx) => (
          <path
            key={idx}
            d={slice.path}
            fill={slice.color}
            stroke="#0a0a0a"
            strokeWidth="2"
            className="pie-slice"
            onMouseEnter={(e) => handleMouseEnter(slice, e)}
            onMouseMove={(e) => handleMouseMove(slice, e)}
            onMouseLeave={handleMouseLeave}
          />
        ))}
      </svg>
      {showLegend && (
        <div className="pie-legend">
          {slices.map((slice, idx) => (
            <div key={idx} className="legend-item">
              <div className="legend-color" style={{ backgroundColor: slice.color }} />
              <div className="legend-label">
                <span className="legend-name">{slice.label}</span>
              </div>
            </div>
          ))}
        </div>
      )}
      {tooltip && (
        <div
          className="pie-tooltip"
          style={{
            left: `${tooltip.x + 10}px`,
            top: `${tooltip.y + 10}px`
          }}
        >
          <div className="tooltip-label">{tooltip.label}</div>
          <div className="tooltip-value">{tooltip.value}</div>
          <div className="tooltip-percentage">{tooltip.percentage}%</div>
        </div>
      )}
    </div>
  )
}

// Shared legend component for matrix layout
function SharedLegend({ data }) {
  if (!data || data.length === 0) {
    return null
  }

  return (
    <div className="shared-legend">
      {data.map((item, idx) => (
        <div key={idx} className="legend-item">
          <div className="legend-color" style={{ backgroundColor: item.color }} />
          <div className="legend-label">
            <span className="legend-name">{item.label}</span>
          </div>
        </div>
      ))}
    </div>
  )
}

function StatsPanel({ logs }) {
  const stats = useMemo(() => {
    if (logs.length === 0) {
      return {
        totalRequests: 0,
        successCount: 0,
        errorCount: 0,
        agentTypeCounts: {},
        toolCounts: {},
        stopReasonCounts: {},
        tokensByAgent: {},
        timeByAgent: {},
        tokensByTool: {},
        timeByTool: {},
        totalTokens: 0,
        durations: [],
        timeRange: null
      }
    }

    const agentTypeCounts = {}
    const toolCounts = {}
    const stopReasonCounts = {}
    const tokensByAgent = {}
    const timeByAgent = {}
    const tokensByTool = {}
    const timeByTool = {}
    let totalTokens = 0
    let successCount = 0
    let errorCount = 0
    const durations = []
    const timestamps = []

    logs.forEach(log => {
      // Agent type counts
      const agentType = log.agent_type?.name || 'unknown'
      agentTypeCounts[agentType] = (agentTypeCounts[agentType] || 0) + 1

      // Duration for this request
      const duration = log.response?.duration_ms

      // Time by agent
      if (duration !== undefined && duration !== null) {
        timeByAgent[agentType] = (timeByAgent[agentType] || 0) + duration
      }

      // Tool counts and metrics
      const toolInfo = log.tool_info || {}
      if (toolInfo.tool_names && Array.isArray(toolInfo.tool_names)) {
        toolInfo.tool_names.forEach(tool => {
          if (tool) {
            toolCounts[tool] = (toolCounts[tool] || 0) + 1

            // Time by tool (distribute duration equally among tools in this request)
            if (duration !== undefined && duration !== null) {
              const timePerTool = duration / toolInfo.tool_names.length
              timeByTool[tool] = (timeByTool[tool] || 0) + timePerTool
            }
          }
        })
      }

      // Stop reason counts
      const stopReason = log.stop_reason || 'unknown'
      stopReasonCounts[stopReason] = (stopReasonCounts[stopReason] || 0) + 1

      // Tokens by agent and tool
      const usage = log.response?.body?.usage
      if (usage) {
        const tokens = (usage.input_tokens || 0) + (usage.output_tokens || 0) || usage.total_tokens || 0
        tokensByAgent[agentType] = (tokensByAgent[agentType] || 0) + tokens
        totalTokens += tokens

        // Tokens by tool (distribute tokens equally among tools in this request)
        if (toolInfo.tool_names && Array.isArray(toolInfo.tool_names) && toolInfo.tool_names.length > 0) {
          const tokensPerTool = tokens / toolInfo.tool_names.length
          toolInfo.tool_names.forEach(tool => {
            if (tool) {
              tokensByTool[tool] = (tokensByTool[tool] || 0) + tokensPerTool
            }
          })
        }
      }

      // Success/Error counts
      const status = log.response?.status || 0
      if (status >= 400) {
        errorCount++
      } else if (status >= 200 && status < 400) {
        successCount++
      }

      // Durations
      if (duration !== undefined && duration !== null) {
        durations.push(duration)
      }

      // Timestamps
      if (log.timestamp) {
        timestamps.push(new Date(log.timestamp).getTime())
      }
    })

    // Calculate time range
    let timeRange = null
    if (timestamps.length > 0) {
      const minTime = Math.min(...timestamps)
      const maxTime = Math.max(...timestamps)
      timeRange = {
        start: new Date(minTime),
        end: new Date(maxTime),
        durationMs: maxTime - minTime
      }
    }

    // Calculate duration stats
    durations.sort((a, b) => a - b)
    const durationStats = durations.length > 0 ? {
      min: durations[0],
      max: durations[durations.length - 1],
      avg: durations.reduce((a, b) => a + b, 0) / durations.length,
      median: durations[Math.floor(durations.length / 2)]
    } : null

    return {
      totalRequests: logs.length,
      successCount,
      errorCount,
      agentTypeCounts,
      toolCounts,
      stopReasonCounts,
      tokensByAgent,
      timeByAgent,
      tokensByTool,
      timeByTool,
      totalTokens,
      durationStats,
      timeRange
    }
  }, [logs])

  if (logs.length === 0) {
    return (
      <div className="stats-panel">
        <div className="stats-empty">No data to display</div>
      </div>
    )
  }

  return (
    <div className="stats-panel">
      <div className="stats-header">
        <div className="stats-title">Statistics Overview</div>
      </div>

      <div className="stats-content">
        {/* Summary Cards */}
        <div className="stats-section">
          <div className="stats-cards">
            <div className="stat-card">
              <div className="stat-label">Total Requests</div>
              <div className="stat-value">{stats.totalRequests}</div>
            </div>
            <div className="stat-card success">
              <div className="stat-label">Success</div>
              <div className="stat-value">{stats.successCount}</div>
            </div>
            <div className="stat-card error">
              <div className="stat-label">Errors</div>
              <div className="stat-value">{stats.errorCount}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Total Tokens</div>
              <div className="stat-value">{stats.totalTokens.toLocaleString()}</div>
            </div>
          </div>
        </div>

        {/* Duration Stats */}
        {stats.durationStats && (
          <div className="stats-section">
            <h3 className="section-title">Request Duration</h3>
            <div className="stats-grid">
              <div className="stat-item">
                <span className="stat-item-label">Min:</span>
                <span className="stat-item-value">{stats.durationStats.min.toFixed(0)}ms</span>
              </div>
              <div className="stat-item">
                <span className="stat-item-label">Max:</span>
                <span className="stat-item-value">{stats.durationStats.max.toFixed(0)}ms</span>
              </div>
              <div className="stat-item">
                <span className="stat-item-label">Avg:</span>
                <span className="stat-item-value">{stats.durationStats.avg.toFixed(0)}ms</span>
              </div>
              <div className="stat-item">
                <span className="stat-item-label">Median:</span>
                <span className="stat-item-value">{stats.durationStats.median.toFixed(0)}ms</span>
              </div>
            </div>
          </div>
        )}

        {/* Time Range */}
        {stats.timeRange && (
          <div className="stats-section">
            <h3 className="section-title">Time Range</h3>
            <div className="stats-grid">
              <div className="stat-item">
                <span className="stat-item-label">Start:</span>
                <span className="stat-item-value">{stats.timeRange.start.toLocaleString()}</span>
              </div>
              <div className="stat-item">
                <span className="stat-item-label">End:</span>
                <span className="stat-item-value">{stats.timeRange.end.toLocaleString()}</span>
              </div>
              <div className="stat-item">
                <span className="stat-item-label">Duration:</span>
                <span className="stat-item-value">{(stats.timeRange.durationMs / 1000 / 60).toFixed(1)} min</span>
              </div>
            </div>
          </div>
        )}

        {/* 2x3 Matrix: Agent/Tool x Usage/Time/Tokens */}
        <div className="stats-section">
          <h3 className="section-title">Agent and Tool Metrics</h3>

          {/* Prepare agent data */}
          {(() => {
            const agentLegendData = Object.entries(stats.agentTypeCounts)
              .sort((a, b) => b[1] - a[1])
              .map(([agentName]) => {
                const agentType = AGENT_TYPES[agentName] || AGENT_TYPES.unknown
                return {
                  label: agentType.label,
                  color: agentType.color
                }
              })

            const agentUsageData = Object.entries(stats.agentTypeCounts)
              .sort((a, b) => b[1] - a[1])
              .map(([agentName, count]) => {
                const agentType = AGENT_TYPES[agentName] || AGENT_TYPES.unknown
                return {
                  label: agentType.label,
                  value: count,
                  color: agentType.color
                }
              })

            const agentTimeData = Object.entries(stats.timeByAgent)
              .sort((a, b) => b[1] - a[1])
              .map(([agentName, time]) => {
                const agentType = AGENT_TYPES[agentName] || AGENT_TYPES.unknown
                return {
                  label: agentType.label,
                  value: time,
                  color: agentType.color
                }
              })

            const agentTokenData = Object.entries(stats.tokensByAgent)
              .sort((a, b) => b[1] - a[1])
              .map(([agentName, tokens]) => {
                const agentType = AGENT_TYPES[agentName] || AGENT_TYPES.unknown
                return {
                  label: agentType.label,
                  value: tokens,
                  color: agentType.color
                }
              })

            // Prepare tool data (top 10)
            const topTools = Object.entries(stats.toolCounts)
              .sort((a, b) => b[1] - a[1])
              .slice(0, 10)
              .map(([toolName], idx) => ({
                name: toolName,
                color: TOOL_COLORS[idx % TOOL_COLORS.length]
              }))

            const toolLegendData = topTools.map(tool => ({
              label: tool.name,
              color: tool.color
            }))

            const toolUsageData = topTools.map(tool => ({
              label: tool.name,
              value: stats.toolCounts[tool.name] || 0,
              color: tool.color
            }))

            const toolTimeData = topTools.map(tool => ({
              label: tool.name,
              value: stats.timeByTool[tool.name] || 0,
              color: tool.color
            }))

            const toolTokenData = topTools.map(tool => ({
              label: tool.name,
              value: stats.tokensByTool[tool.name] || 0,
              color: tool.color
            }))

            return (
              <div className="metrics-matrix">
                {/* Header row */}
                <div className="matrix-header">
                  <div className="matrix-cell"></div>
                  <div className="matrix-cell">Usage Count</div>
                  <div className="matrix-cell">Time Consumed (ms)</div>
                  <div className="matrix-cell">Tokens Consumed</div>
                </div>

                {/* Agent row */}
                <div className="matrix-row">
                  <div className="matrix-cell row-header">
                    <div className="row-title">Agent Types</div>
                    <SharedLegend data={agentLegendData} />
                  </div>
                  <div className="matrix-cell">
                    <PieChart
                      data={agentUsageData}
                      showLegend={false}
                      valueFormatter={(v) => v.toLocaleString()}
                    />
                  </div>
                  <div className="matrix-cell">
                    <PieChart
                      data={agentTimeData}
                      showLegend={false}
                      valueFormatter={(v) => formatTime(v)}
                    />
                  </div>
                  <div className="matrix-cell">
                    <PieChart
                      data={agentTokenData}
                      showLegend={false}
                      valueFormatter={(v) => v.toLocaleString()}
                    />
                  </div>
                </div>

                {/* Tool row */}
                <div className="matrix-row">
                  <div className="matrix-cell row-header">
                    <div className="row-title">Tool Types (Top 10)</div>
                    <SharedLegend data={toolLegendData} />
                  </div>
                  <div className="matrix-cell">
                    <PieChart
                      data={toolUsageData}
                      showLegend={false}
                      valueFormatter={(v) => v.toLocaleString()}
                    />
                  </div>
                  <div className="matrix-cell">
                    <PieChart
                      data={toolTimeData}
                      showLegend={false}
                      valueFormatter={(v) => formatTime(v)}
                    />
                  </div>
                  <div className="matrix-cell">
                    <PieChart
                      data={toolTokenData}
                      showLegend={false}
                      valueFormatter={(v) => v.toLocaleString()}
                    />
                  </div>
                </div>
              </div>
            )
          })()}
        </div>

        {/* Stop Reason Distribution */}
        <div className="stats-section">
          <h3 className="section-title">Stop Reason Distribution</h3>
          <div className="pie-chart-container">
            <PieChart data={Object.entries(stats.stopReasonCounts)
              .sort((a, b) => b[1] - a[1])
              .map(([reason, count], idx) => ({
                label: reason,
                value: count,
                color: STOP_REASON_COLORS[idx % STOP_REASON_COLORS.length]
              }))} />
          </div>
        </div>
      </div>
    </div>
  )
}

export default StatsPanel

