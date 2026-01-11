import { useState, useEffect, useRef, useMemo } from 'react'
import { JsonView, darkStyles } from 'react-json-view-lite'
import { Tooltip } from 'react-tooltip'
import TimelinePanel from './TimelinePanel'
import StatsPanel from './StatsPanel'
import WorkflowPanel from './WorkflowPanel'
import AgentGanttPanel from './AgentGanttPanel'
import SearchBar from './SearchBar'
import 'react-json-view-lite/dist/index.css'
import 'react-tooltip/dist/react-tooltip.css'
import './App.css'

function App() {
  const [logs, setLogs] = useState([])
  const [entitiesData, setEntitiesData] = useState(null)
  const [workflowGraph, setWorkflowGraph] = useState(null)
  const [workflowLoading, setWorkflowLoading] = useState(false)
  const [filter, setFilter] = useState('all')
  const [loading, setLoading] = useState(true)
  const [collapsedItems, setCollapsedItems] = useState(new Set())
  const [collapsedPanels, setCollapsedPanels] = useState(new Set())
  const [selectedLogIndex, setSelectedLogIndex] = useState(null)
  const [windowSize, setWindowSize] = useState(10)
  const [headerCollapsed, setHeaderCollapsed] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [activeBottomPanel, setActiveBottomPanel] = useState(null) // 'timeline' | 'stats' | 'workflow' | 'gantt' | null
  const [bottomPanelHeight, setBottomPanelHeight] = useState(250)
  const [minDuration, setMinDuration] = useState(0)
  const [maxDuration, setMaxDuration] = useState(Infinity)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchType, setSearchType] = useState('all')
  const [searchFields, setSearchFields] = useState(['user_message', 'assistant_response'])
  const [timelineColorByAgent, setTimelineColorByAgent] = useState(true)
  const logRefs = useRef([])
  const mainRef = useRef(null)
  const resizerRef = useRef(null)
  const isDraggingRef = useRef(false)

  // Helper functions defined early to avoid temporal dead zone issues
  const extractUserMessage = (body) => {
    if (!body?.messages) return null
    const lastUserMsg = [...body.messages].reverse().find(m => m.role === 'user')
    if (!lastUserMsg?.content) return null

    if (Array.isArray(lastUserMsg.content)) {
      return lastUserMsg.content.map(c => c.text || c.type).join(' ')
    }
    return lastUserMsg.content
  }

  const extractAssistantResponse = (responseBody) => {
    if (!responseBody) return null

    if (responseBody.error) {
      return { error: responseBody.error.message }
    }

    if (responseBody.content) {
      if (Array.isArray(responseBody.content)) {
        return responseBody.content.map(c => c.text || JSON.stringify(c)).join('\n')
      }
      return responseBody.content
    }

    return JSON.stringify(responseBody, null, 2)
  }

  const handleSearch = (value) => setSearchQuery(value)

  const matchSearchQuery = (log, query, type, fields) => {
    if (!query.trim()) return true

    const searchText = buildSearchText(log, fields)

    if (type === 'regex') {
      try {
        const regex = new RegExp(query, 'i')
        return regex.test(searchText)
      } catch (e) {
        console.error('Invalid regex:', e)
        return false
      }
    }

    return searchText.toLowerCase().includes(query.toLowerCase())
  }

  const buildSearchText = (log, fields) => {
    const texts = []

    if (fields.includes('user_message')) {
      const userMsg = extractUserMessage(log.body)
      if (userMsg) texts.push(userMsg)
    }

    if (fields.includes('assistant_response')) {
      const assistantResp = extractAssistantResponse(log.response?.body)
      if (assistantResp) {
        if (typeof assistantResp === 'string') {
          texts.push(assistantResp)
        } else {
          texts.push(JSON.stringify(assistantResp))
        }
      }
    }

    if (fields.includes('request_body')) {
      texts.push(JSON.stringify(log.body))
    }

    if (fields.includes('response_body')) {
      texts.push(JSON.stringify(log.response?.body))
    }

    if (fields.includes('tools')) {
      const toolsAvailable = log.body?.tools?.map(t => t.name).join(' ') || ''
      const toolsUsed = extractToolNames(log.response?.body) || ''
      texts.push(toolsAvailable, toolsUsed)
    }

    return texts.join(' ')
  }

  const extractToolNames = (responseBody) => {
    if (!responseBody?.content) return ''
    const toolUses = responseBody.content.filter(c => c.type === 'tool_use')
    return toolUses.map(t => t.name).join(' ')
  }

  const fetchLogs = async () => {
    try {
      const response = await fetch('/api/logs')
      const data = await response.json()

      // New API returns just logs, no workflow graph
      if (data && typeof data === 'object' && 'logs' in data) {
        setLogs(data.logs || [])
      } else if (Array.isArray(data)) {
        // Fallback for old format
        setLogs(data)
      } else {
        console.error('Unexpected data format:', data)
        setLogs([])
      }

      setLoading(false)
    } catch (error) {
      console.error('Failed to fetch logs:', error)
      setLoading(false)
    }
  }

  const fetchEntities = async () => {
    try {
      // Fetch entities JSON
      const entitiesResponse = await fetch('/api/entities')
      if (entitiesResponse.ok) {
        const entities = await entitiesResponse.json()
        setEntitiesData(entities)
      }
    } catch (error) {
      console.error('Failed to fetch entities:', error)
    }
  }

  const fetchWorkflow = async () => {
    if (workflowLoading) {
      console.log('Workflow already loading, skipping...')
      return
    }

    setWorkflowLoading(true)
    try {
      console.log('Fetching workflow graph...')
      const response = await fetch('/api/workflow')
      const data = await response.json()

      if (data && typeof data === 'object') {
        setLogs(data.logs || logs) // Update logs if included
        setWorkflowGraph(data.workflow_graph || null)
        console.log('Workflow graph loaded:', data.workflow_graph)
      } else {
        console.error('Unexpected workflow data format:', data)
        setWorkflowGraph(null)
      }
    } catch (error) {
      console.error('Failed to fetch workflow:', error)
      setWorkflowGraph(null)
    } finally {
      setWorkflowLoading(false)
    }
  }

  const filteredLogs = useMemo(() => {
    const filtered = logs.filter(log => {
      // Status filter
      if (filter === 'errors' && log.response?.status < 400) return false
      if (filter === 'success' && log.response?.status >= 400) return false

      // Duration filter
      const duration = log.response?.duration_ms
      if (duration !== undefined) {
        if (duration < minDuration) return false
        if (duration > maxDuration) return false
      }

      // Search query filter
      if (searchQuery) {
        if (!matchSearchQuery(log, searchQuery, searchType, searchFields)) {
          return false
        }
      }

      return true
    })

    // Reverse so newest logs appear first (at top)
    return filtered.reverse()
  }, [logs, filter, minDuration, maxDuration, searchQuery, searchType, searchFields])

  useEffect(() => {
    fetchLogs()
    fetchEntities()
    if (autoRefresh) {
      const interval = setInterval(() => {
        fetchLogs()
        fetchEntities()
      }, 2000)
      return () => clearInterval(interval)
    }
  }, [autoRefresh])

  useEffect(() => {
    if (selectedLogIndex === null && filteredLogs.length > 0) {
      setSelectedLogIndex(Math.floor(filteredLogs.length / 2))
    }
  }, [filteredLogs, selectedLogIndex])

  // Handle timeline panel resize
  const handleResizerMouseDown = () => {
    isDraggingRef.current = true
    document.body.style.cursor = 'ns-resize'
    document.body.style.userSelect = 'none'

    const handleMouseMove = (e) => {
      if (!isDraggingRef.current) return

      const container = mainRef.current?.parentElement
      if (!container) return

      const containerRect = container.getBoundingClientRect()
      const newHeight = containerRect.bottom - e.clientY
      // Only enforce minimum height, no maximum limit
      const clampedHeight = Math.max(100, newHeight)
      setBottomPanelHeight(clampedHeight)
    }

    const handleMouseUp = () => {
      isDraggingRef.current = false
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
  }

  // Removed scroll-based selection - only timeline clicks update selection

  const windowedLogs = useMemo(() => {
    if (filteredLogs.length === 0) return []
    if (selectedLogIndex === null) return filteredLogs.slice(0, windowSize)

    const halfWindow = Math.floor(windowSize / 2)
    const startIdx = Math.max(0, selectedLogIndex - halfWindow)
    const endIdx = Math.min(filteredLogs.length, startIdx + windowSize)

    return filteredLogs.slice(startIdx, endIdx)
  }, [filteredLogs, selectedLogIndex, windowSize])

  const toggleCollapse = (idx) => {
    setCollapsedItems(prev => {
      const next = new Set(prev)
      if (next.has(idx)) {
        next.delete(idx)
      } else {
        next.add(idx)
      }
      return next
    })
  }

  const togglePanel = (logIdx, panelName) => {
    const key = `${logIdx}-${panelName}`
    setCollapsedPanels(prev => {
      const next = new Set(prev)
      if (next.has(key)) {
        next.delete(key)
      } else {
        next.add(key)
      }
      return next
    })
  }

  const scrollToLog = (absoluteIdx) => {
    setSelectedLogIndex(absoluteIdx)
    setTimeout(() => {
      logRefs.current[0]?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }, 100)
  }

  if (loading) {
    return <div className="loading">Loading logs...</div>
  }

  return (
    <div className="app">
      <aside className="minimap">
        <div className="minimap-title">Timeline</div>
        <div className="minimap-items">
          {filteredLogs.map((log, idx) => {
            const isError = log.response?.status >= 400
            const isSelected = selectedLogIndex === idx
            const timestamp = new Date(log.timestamp).toLocaleString()
            const model = log.body?.model || 'unknown'
            const status = log.response?.status || 'pending'
            const usage = log.response?.body?.usage
            const duration = log.response?.duration_ms
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

            const tooltipContent = tooltipLines.join('\n')

            // Determine color
            let itemStyle = {}
            if (timelineColorByAgent && agentType) {
              itemStyle.backgroundColor = agentType.color
            }

            return (
              <div
                key={idx}
                className={`minimap-item ${isError ? 'error' : 'success'} ${isSelected ? 'selected' : ''}`}
                onClick={() => scrollToLog(idx)}
                data-tooltip-id="minimap-tooltip"
                data-tooltip-content={tooltipContent}
                style={itemStyle}
              />
            )
          })}
        </div>
        <Tooltip
          id="minimap-tooltip"
          place="right"
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
      </aside>

      <div className="content">
        <header className={`header ${headerCollapsed ? 'collapsed' : ''}`}>
          <div className="header-top">
            <div className="header-title-row">
              <h1>Claude Log Viewer</h1>
              <button
                className="collapse-header-btn"
                onClick={() => setHeaderCollapsed(!headerCollapsed)}
                title={headerCollapsed ? 'Expand header' : 'Collapse header'}
              >
                {headerCollapsed ? '▼' : '▲'}
              </button>
            </div>
            {!headerCollapsed && (
              <>
                <SearchBar
                  searchQuery={searchQuery}
                  onSearchChange={handleSearch}
                  searchType={searchType}
                  onSearchTypeChange={setSearchType}
                  searchFields={searchFields}
                  onSearchFieldsChange={setSearchFields}
                />
                <div className="filters">
                  <button
                    className={filter === 'all' ? 'active' : ''}
                    onClick={() => setFilter('all')}
                  >
                    All ({logs.length})
                  </button>
                  <button
                    className={filter === 'success' ? 'active' : ''}
                    onClick={() => setFilter('success')}
                  >
                    Success
                  </button>
                  <button
                    className={filter === 'errors' ? 'active' : ''}
                    onClick={() => setFilter('errors')}
                  >
                    Errors
                  </button>
                  <button
                    className={autoRefresh ? 'active' : ''}
                    onClick={() => setAutoRefresh(!autoRefresh)}
                    title={autoRefresh ? 'Auto-refresh enabled' : 'Auto-refresh disabled'}
                  >
                    {autoRefresh ? 'Live' : 'Paused'}
                  </button>
                  <button
                    className={activeBottomPanel === 'timeline' ? 'active' : ''}
                    onClick={() => setActiveBottomPanel(activeBottomPanel === 'timeline' ? null : 'timeline')}
                    title={activeBottomPanel === 'timeline' ? 'Hide timeline' : 'Show timeline'}
                  >
                    Timeline
                  </button>
                  <button
                    className={activeBottomPanel === 'stats' ? 'active' : ''}
                    onClick={() => setActiveBottomPanel(activeBottomPanel === 'stats' ? null : 'stats')}
                    title={activeBottomPanel === 'stats' ? 'Hide stats' : 'Show stats'}
                  >
                    Stats
                  </button>
                  <button
                    className={activeBottomPanel === 'workflow' ? 'active' : ''}
                    onClick={() => {
                      const newPanel = activeBottomPanel === 'workflow' ? null : 'workflow'
                      setActiveBottomPanel(newPanel)
                      // Fetch workflow graph when opening the panel
                      if (newPanel === 'workflow' && !workflowGraph) {
                        fetchWorkflow()
                      }
                    }}
                    title={activeBottomPanel === 'workflow' ? 'Hide workflow graph' : 'Show workflow graph'}
                  >
                    Workflow {workflowLoading && '⏳'}
                  </button>
                  <button
                    className={activeBottomPanel === 'gantt' ? 'active' : ''}
                    onClick={() => setActiveBottomPanel(activeBottomPanel === 'gantt' ? null : 'gantt')}
                    title={activeBottomPanel === 'gantt' ? 'Hide agent gantt' : 'Show agent gantt'}
                  >
                    Agent Gantt
                  </button>
                  {activeBottomPanel === 'timeline' && (
                    <button
                      className={timelineColorByAgent ? 'active' : ''}
                      onClick={() => setTimelineColorByAgent(!timelineColorByAgent)}
                      title={timelineColorByAgent ? 'Disable agent color coding' : 'Enable agent color coding'}
                    >
                      Color by Agent
                    </button>
                  )}
                  <button
                    className="fold-toggle-btn"
                    onClick={() => {
                      const allCollapsed = collapsedItems.size === windowedLogs.length
                      if (allCollapsed) {
                        setCollapsedItems(new Set())
                      } else {
                        setCollapsedItems(new Set(windowedLogs.map((_, idx) => idx)))
                      }
                    }}
                    title={collapsedItems.size === windowedLogs.length ? 'Expand all log entries' : 'Collapse all log entries'}
                  >
                    {collapsedItems.size === windowedLogs.length ? 'Unfold All' : 'Fold All'}
                  </button>
                </div>
              </>
            )}
          </div>
          {!headerCollapsed && (
            <div className="range-controls">
              <div className="range-info">
                Showing {windowedLogs.length} of {filteredLogs.length} logs
                {selectedLogIndex !== null && ` (centered around #${selectedLogIndex + 1})`}
              </div>
              <div className="control-row">
                <div className="window-size-control">
                  <label>
                    Window size:
                    <input
                      type="number"
                      min="10"
                      max="200"
                      step="10"
                      value={windowSize}
                      onChange={(e) => setWindowSize(parseInt(e.target.value) || 10)}
                      className="window-size-input"
                    />
                  </label>
                </div>
                <div className="duration-filter-control">
                  <label>
                    Min duration (ms):
                    <input
                      type="number"
                      min="0"
                      step="100"
                      value={minDuration}
                      onChange={(e) => setMinDuration(parseInt(e.target.value) || 0)}
                      className="duration-input"
                      placeholder="0"
                    />
                  </label>
                  <label>
                    Max duration (ms):
                    <input
                      type="number"
                      min="0"
                      step="100"
                      value={maxDuration === Infinity ? '' : maxDuration}
                      onChange={(e) => setMaxDuration(e.target.value === '' ? Infinity : parseInt(e.target.value) || Infinity)}
                      className="duration-input"
                      placeholder="∞"
                    />
                  </label>
                  {(minDuration > 0 || maxDuration !== Infinity) && (
                    <button
                      onClick={() => {
                        setMinDuration(0)
                        setMaxDuration(Infinity)
                      }}
                      className="clear-duration-btn"
                      title="Clear duration filter"
                    >
                      Clear
                    </button>
                  )}
                </div>
              </div>
            </div>
          )}
        </header>

        <main className="main" ref={mainRef} style={activeBottomPanel ? { flex: `1 1 calc(100% - ${bottomPanelHeight}px)` } : {}}>
        {windowedLogs.length === 0 ? (
          <div className="empty">No logs found</div>
        ) : (
          windowedLogs.map((log, idx) => {
            const userMsg = extractUserMessage(log.body)
            const assistantResp = extractAssistantResponse(log.response?.body)
            const isError = log.response?.status >= 400
            const timestamp = new Date(log.timestamp).toLocaleString()
            const model = log.body?.model || 'unknown'
            const tokens = log.response?.body?.usage || log.response?.body?.input_tokens
            const duration = log.response?.duration_ms
            const isCollapsed = collapsedItems.has(idx)

            // Enriched metadata
            const agentType = log.agent_type
            const toolInfo = log.tool_info || {}
            const hasSubagents = log.has_subagent_spawns || false
            const subagentCount = log.subagent_count || 0
            const subagentSpawns = log.subagent_spawns || []
            const hasErrors = log.has_errors || false
            const toolErrors = log.tool_errors || 0
            const stopReason = log.stop_reason
            const logIndex = log.log_index

            return (
              <div
                key={idx}
                className={`log-entry ${agentType ? `agent-${agentType.name}` : ''}`}
                ref={el => logRefs.current[idx] = el}
                style={agentType ? { borderLeftColor: agentType.color } : {}}
              >
                <div className="metadata" onClick={() => toggleCollapse(idx)}>
                  <span className="collapse-icon">{isCollapsed ? '▶' : '▼'}</span>
                  {logIndex !== undefined && (
                    <span className="log-index" data-tooltip-id="tooltip" data-tooltip-content="Log index (matches workflow graph node number)">
                      #{logIndex}
                    </span>
                  )}
                  <span className="time">{timestamp}</span>

                  {agentType && (
                    <span
                      className="agent-badge"
                      style={{ backgroundColor: agentType.color }}
                      data-tooltip-id="tooltip"
                      data-tooltip-content={agentType.description}
                    >
                      {agentType.label}
                    </span>
                  )}

                  <span className="model">{model}</span>
                  <span className={`status ${isError ? 'error' : 'success'}`}>
                    {log.response?.status || 'pending'}
                  </span>

                  {stopReason && (
                    <span className={`stop-reason ${stopReason}`}>
                      {stopReason}
                    </span>
                  )}

                  {toolInfo.count > 0 && (
                    <span
                      className="tool-count"
                      data-tooltip-id="tooltip"
                      data-tooltip-content={toolInfo.tool_names.join(', ')}
                    >
                      {toolInfo.count} tool{toolInfo.count > 1 ? 's' : ''}
                    </span>
                  )}

                  {hasSubagents && (
                    <span
                      className="subagent-badge"
                      data-tooltip-id="tooltip"
                      data-tooltip-content={`Spawns ${subagentCount} subagent${subagentCount > 1 ? 's' : ''}`}
                    >
                      {subagentCount} subagent{subagentCount > 1 ? 's' : ''}
                    </span>
                  )}

                  {hasErrors && (
                    <span
                      className="error-badge"
                      data-tooltip-id="tooltip"
                      data-tooltip-content={`${toolErrors} tool error${toolErrors > 1 ? 's' : ''}`}
                    >
                      {toolErrors} error{toolErrors > 1 ? 's' : ''}
                    </span>
                  )}

                  {tokens && (
                    <span className="tokens">
                      {tokens.input_tokens || tokens} tokens
                    </span>
                  )}
                  {duration !== undefined && (
                    <span className="duration">
                      {duration.toFixed(0)}ms
                    </span>
                  )}
                </div>

                {!isCollapsed && (
                  <>
                    {hasSubagents && subagentSpawns.length > 0 && (
                      <div className="message subagent-info">
                        <div
                          className="label clickable"
                          onClick={() => togglePanel(idx, 'subagents')}
                        >
                          <span className="panel-icon">{collapsedPanels.has(`${idx}-subagents`) ? '▶' : '▼'}</span>
                          Subagent Spawns ({subagentCount})
                        </div>
                        {!collapsedPanels.has(`${idx}-subagents`) && (
                          <div className="content">
                            {subagentSpawns.map((spawn, spawnIdx) => (
                              <div key={spawnIdx} className="subagent-spawn-item">
                                <div className="subagent-type">{spawn.subagent_type || 'Unknown'}</div>
                                {spawn.description && (
                                  <div className="subagent-desc">{spawn.description}</div>
                                )}
                                {spawn.model && (
                                  <div className="subagent-model">Model: {spawn.model}</div>
                                )}
                                {spawn.has_resume && (
                                  <span className="resume-badge">RESUME</span>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    {toolInfo.count > 0 && (
                      <div className="message tool-info">
                        <div
                          className="label clickable"
                          onClick={() => togglePanel(idx, 'tools')}
                        >
                          <span className="panel-icon">{collapsedPanels.has(`${idx}-tools`) ? '▶' : '▼'}</span>
                          Tools Used ({toolInfo.count})
                        </div>
                        {!collapsedPanels.has(`${idx}-tools`) && (
                          <div className="content">
                            <div className="tool-list">
                              {toolInfo.tool_names.map((toolName, toolIdx) => {
                                const category = toolName ?
                                  (toolName.match(/Read|Glob|Grep|LSP/) ? 'read' :
                                   toolName.match(/Edit|Write/) ? 'write' :
                                   toolName.match(/Bash|KillShell/) ? 'execute' :
                                   toolName.match(/Task|TodoWrite|EnterPlanMode|ExitPlanMode/) ? 'orchestration' :
                                   toolName.match(/AskUserQuestion/) ? 'interaction' : 'other') : 'other'
                                return (
                                  <span key={toolIdx} className={`tool-badge tool-${category}`}>
                                    {toolName}
                                  </span>
                                )
                              })}
                            </div>
                            {Object.keys(toolInfo.categories).length > 0 && (
                              <div className="tool-categories">
                                {Object.entries(toolInfo.categories).map(([cat, count]) => (
                                  <span key={cat} className="category-badge">
                                    {cat}: {count}
                                  </span>
                                ))}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}

                    {userMsg && (
                      <div className="message user">
                        <div
                          className="label clickable"
                          onClick={() => togglePanel(idx, 'user')}
                        >
                          <span className="panel-icon">{collapsedPanels.has(`${idx}-user`) ? '▶' : '▼'}</span>
                          User → CC
                        </div>
                        {!collapsedPanels.has(`${idx}-user`) && (
                          <div className="content">{userMsg}</div>
                        )}
                      </div>
                    )}

                    <div className="message request">
                      <div
                        className="label clickable"
                        onClick={() => togglePanel(idx, 'request')}
                      >
                        <span className="panel-icon">{collapsedPanels.has(`${idx}-request`) ? '▶' : '▼'}</span>
                        CC → Inference
                      </div>
                      {!collapsedPanels.has(`${idx}-request`) && (
                        <div className="content">
                          <div className="endpoint">{log.method} {log.path}</div>
                          <div className="json-viewer-container">
                            <JsonView data={log.body} style={darkStyles} />
                          </div>
                        </div>
                      )}
                    </div>

                    <div className={`message response ${isError ? 'error' : ''}`}>
                      <div
                        className="label clickable"
                        onClick={() => togglePanel(idx, 'response')}
                      >
                        <span className="panel-icon">{collapsedPanels.has(`${idx}-response`) ? '▶' : '▼'}</span>
                        Inference → CC
                      </div>
                      {!collapsedPanels.has(`${idx}-response`) && (
                        <div className="content">
                          {isError ? (
                            <div className="error-msg">{assistantResp?.error || 'Error occurred'}</div>
                          ) : (
                            <pre>{typeof assistantResp === 'string' ? assistantResp : JSON.stringify(assistantResp, null, 2)}</pre>
                          )}
                        </div>
                      )}
                    </div>
                  </>
                )}
              </div>
            )
          })
        )}
        </main>

        {activeBottomPanel && (
          <>
            <div
              className="bottom-panel-resizer"
              ref={resizerRef}
              onMouseDown={handleResizerMouseDown}
            />
            <div className="bottom-panel-container" style={{ height: `${bottomPanelHeight}px` }}>
              {activeBottomPanel === 'timeline' && (
                <TimelinePanel
                  logs={filteredLogs}
                  onSelectLog={scrollToLog}
                  selectedLogIndex={selectedLogIndex}
                  colorByAgent={timelineColorByAgent}
                />
              )}
              {activeBottomPanel === 'stats' && (
                <StatsPanel logs={filteredLogs} />
              )}
              {activeBottomPanel === 'workflow' && (
                <WorkflowPanel
                  logs={logs}
                  workflowGraph={workflowGraph}
                  onRefresh={fetchWorkflow}
                  onSelectLog={scrollToLog}
                  selectedLogIndex={selectedLogIndex}
                />
              )}
              {activeBottomPanel === 'gantt' && (
                <AgentGanttPanel
                  entitiesData={entitiesData}
                  logs={logs}
                />
              )}
            </div>
          </>
        )}
      </div>
    </div>
  )
}

export default App

