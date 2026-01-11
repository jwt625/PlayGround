import { useState, useMemo, useEffect } from 'react'
import './AgentGanttPanel.css'

function AgentGanttPanel({ entitiesData, logs }) {
  const [selectedAgentId, setSelectedAgentId] = useState(null)

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

  const { agents, minTime, maxTime, duration } = agentData

  // Format duration
  const formatDuration = (ms) => {
    if (ms < 1000) return `${ms.toFixed(0)}ms`
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
    return `${(ms / 60000).toFixed(1)}m`
  }

  // Generate time markers
  const timeMarkers = useMemo(() => {
    if (duration === 0) return []
    const markers = []
    for (let i = 0; i <= 10; i++) {
      const fraction = i / 10
      const time = minTime + (duration * fraction)
      markers.push({ fraction, time })
    }
    return markers
  }, [minTime, duration])

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
      </div>

      <div className="gantt-container">
        <div className="gantt-labels">
          {agents.map((agent, idx) => (
            <div
              key={agent.agent_id}
              className={`gantt-label ${selectedAgentId === agent.agent_id ? 'selected' : ''}`}
              onClick={() => setSelectedAgentId(agent.agent_id)}
              style={{ borderLeftColor: agent.agentType.color }}
            >
              <span className="label-id">{agent.agent_id}</span>
              <span className="label-count">{agent.requests.length}</span>
            </div>
          ))}
        </div>

        <div className="gantt-timeline">
          <div className="gantt-axis">
            {timeMarkers.map((marker, idx) => (
              <div key={idx} className="axis-marker" style={{ left: `${marker.fraction * 100}%` }}>
                <span className="axis-label">{new Date(marker.time).toLocaleTimeString()}</span>
              </div>
            ))}
          </div>

          <div className="gantt-rows">
            <div className="gantt-grid">
              {timeMarkers.map((marker, idx) => (
                <div key={idx} className="grid-line" style={{ left: `${marker.fraction * 100}%` }} />
              ))}
            </div>

            {agents.map((agent, agentIdx) => (
              <div key={agent.agent_id} className="gantt-row">
                {agent.requests.map((request, reqIdx) => {
                  const left = ((request.timestamp - minTime) / duration) * 100
                  const width = Math.max(0.5, (request.duration / duration) * 100)
                  
                  return (
                    <div
                      key={reqIdx}
                      className="gantt-bar"
                      style={{
                        left: `${left}%`,
                        width: `${width}%`,
                        background: request.agentType.color
                      }}
                      data-tooltip-id="gantt-tooltip"
                      data-tooltip-content={`${agent.agent_id} - Request ${reqIdx + 1}/${agent.requests.length}\n${new Date(request.timestamp).toLocaleString()}\n${request.duration.toFixed(0)}ms\nStatus: ${request.status}`}
                    />
                  )
                })}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default AgentGanttPanel

