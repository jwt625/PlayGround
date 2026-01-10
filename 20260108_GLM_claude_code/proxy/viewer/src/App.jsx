import { useState, useEffect, useRef, useMemo } from 'react'
import { JsonView, darkStyles } from 'react-json-view-lite'
import { Tooltip } from 'react-tooltip'
import TimelinePanel from './TimelinePanel'
import SearchBar from './SearchBar'
import 'react-json-view-lite/dist/index.css'
import 'react-tooltip/dist/react-tooltip.css'
import './App.css'

function App() {
  const [logs, setLogs] = useState([])
  const [filter, setFilter] = useState('all')
  const [loading, setLoading] = useState(true)
  const [collapsedItems, setCollapsedItems] = useState(new Set())
  const [collapsedPanels, setCollapsedPanels] = useState(new Set())
  const [selectedLogIndex, setSelectedLogIndex] = useState(null)
  const [windowSize, setWindowSize] = useState(10)
  const [headerCollapsed, setHeaderCollapsed] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [showTimeline, setShowTimeline] = useState(false)
  const [timelineHeight, setTimelineHeight] = useState(250)
  const [minDuration, setMinDuration] = useState(0)
  const [maxDuration, setMaxDuration] = useState(Infinity)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchType, setSearchType] = useState('all')
  const [searchFields, setSearchFields] = useState(['user_message', 'assistant_response'])
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
      setLogs(data)
      setLoading(false)
    } catch (error) {
      console.error('Failed to fetch logs:', error)
      setLoading(false)
    }
  }

  const filteredLogs = useMemo(() => {
    return logs.filter(log => {
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
  }, [logs, filter, minDuration, maxDuration, searchQuery, searchType, searchFields])

  useEffect(() => {
    fetchLogs()
    if (autoRefresh) {
      const interval = setInterval(fetchLogs, 2000)
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
      const clampedHeight = Math.max(150, Math.min(600, newHeight))
      setTimelineHeight(clampedHeight)
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
            let totalTokens = 'no tokens'
            if (usage) {
              if (usage.input_tokens !== undefined && usage.output_tokens !== undefined) {
                totalTokens = `${usage.input_tokens + usage.output_tokens} tokens (${usage.input_tokens} in + ${usage.output_tokens} out)`
              } else if (usage.total_tokens !== undefined) {
                totalTokens = `${usage.total_tokens} tokens`
              }
            }
            const durationText = duration !== undefined ? `${duration.toFixed(0)}ms` : 'no duration'

            return (
              <div
                key={idx}
                className={`minimap-item ${isError ? 'error' : 'success'} ${isSelected ? 'selected' : ''}`}
                onClick={() => scrollToLog(idx)}
                data-tooltip-id="minimap-tooltip"
                data-tooltip-content={`${timestamp}\n${model}\nStatus: ${status}\n${totalTokens}\nDuration: ${durationText}`}
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
                    className={showTimeline ? 'active' : ''}
                    onClick={() => setShowTimeline(!showTimeline)}
                    title={showTimeline ? 'Hide timeline' : 'Show timeline'}
                  >
                    Timeline
                  </button>
                  <button
                    onClick={() => setCollapsedItems(new Set(windowedLogs.map((_, idx) => idx)))}
                    title="Collapse all log entries"
                  >
                    Fold All
                  </button>
                  <button
                    onClick={() => setCollapsedItems(new Set())}
                    title="Expand all log entries"
                  >
                    Unfold All
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

        <main className="main" ref={mainRef} style={showTimeline ? { flex: `1 1 calc(100% - ${timelineHeight}px)` } : {}}>
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

            return (
              <div
                key={idx}
                className="log-entry"
                ref={el => logRefs.current[idx] = el}
              >
                <div className="metadata" onClick={() => toggleCollapse(idx)}>
                  <span className="collapse-icon">{isCollapsed ? '▶' : '▼'}</span>
                  <span className="time">{timestamp}</span>
                  <span className="model">{model}</span>
                  <span className={`status ${isError ? 'error' : 'success'}`}>
                    {log.response?.status || 'pending'}
                  </span>
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

        {showTimeline && (
          <>
            <div
              className="timeline-resizer"
              ref={resizerRef}
              onMouseDown={handleResizerMouseDown}
            />
            <div className="timeline-container" style={{ height: `${timelineHeight}px` }}>
              <TimelinePanel
                logs={filteredLogs}
                onSelectLog={scrollToLog}
                selectedLogIndex={selectedLogIndex}
              />
            </div>
          </>
        )}
      </div>
    </div>
  )
}

export default App

