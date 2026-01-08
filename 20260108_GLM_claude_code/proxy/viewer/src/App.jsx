import { useState, useEffect, useRef } from 'react'
import './App.css'

function App() {
  const [logs, setLogs] = useState([])
  const [filter, setFilter] = useState('all')
  const [loading, setLoading] = useState(true)
  const [collapsedItems, setCollapsedItems] = useState(new Set())
  const [collapsedPanels, setCollapsedPanels] = useState(new Set())
  const [visibleLogIndex, setVisibleLogIndex] = useState(null)
  const logRefs = useRef([])
  const mainRef = useRef(null)

  useEffect(() => {
    fetchLogs()
    const interval = setInterval(fetchLogs, 2000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    const handleScroll = () => {
      if (!mainRef.current) return

      const mainRect = mainRef.current.getBoundingClientRect()
      const mainCenter = mainRect.top + mainRect.height / 2

      let closestIndex = null
      let closestDistance = Infinity

      logRefs.current.forEach((ref, idx) => {
        if (!ref) return
        const rect = ref.getBoundingClientRect()
        const logCenter = rect.top + rect.height / 2
        const distance = Math.abs(logCenter - mainCenter)

        if (distance < closestDistance) {
          closestDistance = distance
          closestIndex = idx
        }
      })

      setVisibleLogIndex(closestIndex)
    }

    const main = mainRef.current
    if (main) {
      main.addEventListener('scroll', handleScroll)
      handleScroll()
    }

    return () => {
      if (main) {
        main.removeEventListener('scroll', handleScroll)
      }
    }
  }, [logs, filter])

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

  const filteredLogs = logs.filter(log => {
    if (filter === 'all') return true
    if (filter === 'errors') return log.response?.status >= 400
    if (filter === 'success') return log.response?.status < 400
    return true
  })

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

  const scrollToLog = (idx) => {
    logRefs.current[idx]?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }

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
            const isVisible = visibleLogIndex === idx
            return (
              <div
                key={idx}
                className={`minimap-item ${isError ? 'error' : 'success'} ${isVisible ? 'visible' : ''}`}
                onClick={() => scrollToLog(idx)}
                title={new Date(log.timestamp).toLocaleTimeString()}
              />
            )
          })}
        </div>
      </aside>

      <div className="content">
        <header className="header">
          <h1>Claude Log Viewer</h1>
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
          </div>
        </header>

        <main className="main" ref={mainRef}>
        {filteredLogs.length === 0 ? (
          <div className="empty">No logs found</div>
        ) : (
          filteredLogs.map((log, idx) => {
            const userMsg = extractUserMessage(log.body)
            const assistantResp = extractAssistantResponse(log.response?.body)
            const isError = log.response?.status >= 400
            const timestamp = new Date(log.timestamp).toLocaleString()
            const model = log.body?.model || 'unknown'
            const tokens = log.response?.body?.usage || log.response?.body?.input_tokens
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
                          <details>
                            <summary>Request body</summary>
                            <pre>{JSON.stringify(log.body, null, 2)}</pre>
                          </details>
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
      </div>
    </div>
  )
}

export default App

