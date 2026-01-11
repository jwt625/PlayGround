import { useEffect } from 'react'
import './MessagesModal.css'

function MessagesModal({ isOpen, onClose, request, agent, entitiesData }) {
  // Close on Escape key
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape') onClose()
    }
    if (isOpen) {
      window.addEventListener('keydown', handleEscape)
      return () => window.removeEventListener('keydown', handleEscape)
    }
  }, [isOpen, onClose])

  if (!isOpen || !request || !agent || !entitiesData) return null

  const messages = entitiesData.entities?.messages || []
  const contentBlocks = entitiesData.entities?.content_blocks || []
  const contentBlockMap = new Map(contentBlocks.map(cb => [cb.id, cb]))

  // Get the log entry for this request
  const logEntry = request.log || {}
  const requestBody = logEntry.body || {}
  const responseBody = logEntry.response?.body || {}

  // Extract system prompts
  const systemPrompts = requestBody.system || []

  // Get messages for this specific request and sort by position
  const requestMessages = messages.filter(m => {
    const reqId = m.request_id
    const reqIdNum = typeof reqId === 'number' ? reqId : parseInt(reqId.replace('req_', ''))
    return reqIdNum === request.reqId
  }).sort((a, b) => {
    return (a.position_in_conversation || 0) - (b.position_in_conversation || 0)
  })

  const escapeHtml = (text) => {
    const div = document.createElement('div')
    div.textContent = text
    return div.innerHTML
  }

  const renderContentBlock = (block) => {
    if (!block) return null

    if (block.type === 'text') {
      return <div dangerouslySetInnerHTML={{ __html: escapeHtml(block.text || '') }} />
    } else if (block.type === 'tool_use') {
      return (
        <div className="tool-use">
          <div style={{ fontWeight: 600, marginBottom: '4px' }}>Tool: {block.tool_name}</div>
          <div style={{ fontSize: '10px', color: '#858585' }}>
            {JSON.stringify(block.tool_input, null, 2)}
          </div>
        </div>
      )
    } else if (block.type === 'tool_result') {
      const resultText = typeof block.result_content === 'string'
        ? block.result_content
        : JSON.stringify(block.result_content, null, 2)
      return (
        <div className="tool-result">
          <div style={{ fontWeight: 600, marginBottom: '4px' }}>Tool Result</div>
          <div style={{ fontSize: '10px', color: '#858585' }}>
            {resultText}
          </div>
        </div>
      )
    }
    return null
  }

  const renderMessageContent = (msg) => {
    if (msg.content_blocks && msg.content_blocks.length > 0) {
      return msg.content_blocks.map((blockId, idx) => {
        const block = contentBlockMap.get(blockId)
        return <div key={idx}>{renderContentBlock(block)}</div>
      })
    } else if (typeof msg.content === 'string') {
      return <div dangerouslySetInnerHTML={{ __html: escapeHtml(msg.content) }} />
    }
    return null
  }

  return (
    <div className="messages-modal" onClick={onClose}>
      <div className="messages-modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="messages-modal-header">
          <div className="messages-modal-title">
            {agent.agent_id} - Request {request.idx + 1}/{agent.requests.length}
          </div>
          <button className="messages-modal-close" onClick={onClose}>&times;</button>
        </div>

        <div className="messages-modal-body">
          {/* Request metadata */}
          <div className="request-metadata">
            <div className="metadata-grid">
              <div><span className="metadata-label">Agent Type:</span> {request.agentType.label}</div>
              <div><span className="metadata-label">Request ID:</span> {request.reqId}</div>
              <div><span className="metadata-label">Turn:</span> {request.idx + 1}/{agent.requests.length}</div>
              <div><span className="metadata-label">Status:</span> {request.status}</div>
              <div><span className="metadata-label">Time:</span> {new Date(request.timestamp).toLocaleString()}</div>
              <div><span className="metadata-label">Duration:</span> {request.duration.toFixed(0)}ms</div>
            </div>
          </div>

          {/* Two-column layout */}
          <div className="messages-columns">
            {/* Left column: Messages */}
            <div className="messages-column">
              <div className="column-header">Messages</div>
              {requestMessages.map((msg, idx) => {
                const roleClass = msg.role === 'user' ? 'user' : 'assistant'
                const roleColor = msg.role === 'user' ? '#569cd6' : '#4ec9b0'

                return (
                  <div key={idx} className={`message-item ${roleClass}`}>
                    <div className="message-header">
                      <span className="message-role" style={{ color: roleColor }}>
                        {msg.role}
                      </span>
                      <span style={{ color: '#858585' }}>
                        Pos: {msg.position_in_conversation || 0}
                      </span>
                    </div>
                    <div className="message-content">
                      {renderMessageContent(msg)}
                    </div>
                  </div>
                )
              })}

              {requestMessages.length === 0 && (
                <div style={{ textAlign: 'center', color: '#858585', padding: '20px' }}>
                  No messages found for this request
                </div>
              )}
            </div>

            {/* Right column: System Prompt & Response */}
            <div className="info-column">
              {/* System Prompts */}
              <div className="info-section">
                <div className="column-header">System Prompts</div>
                {systemPrompts.length > 0 ? (
                  systemPrompts.map((prompt, idx) => (
                    <div key={idx} className="system-prompt-item">
                      <div className="system-prompt-header">
                        Prompt {idx + 1}/{systemPrompts.length}
                        {prompt.cache_control && (
                          <span className="cache-badge">cached</span>
                        )}
                      </div>
                      <div className="system-prompt-text">
                        {prompt.text || ''}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="empty-state">No system prompts</div>
                )}
              </div>

              {/* Inference Response */}
              <div className="info-section">
                <div className="column-header">Inference Response</div>
                <div className="response-details">
                  <div className="response-field">
                    <span className="field-label">Response ID:</span>
                    <span className="field-value">{responseBody.id || 'N/A'}</span>
                  </div>
                  <div className="response-field">
                    <span className="field-label">Model:</span>
                    <span className="field-value">{responseBody.model || requestBody.model || 'N/A'}</span>
                  </div>
                  <div className="response-field">
                    <span className="field-label">Stop Reason:</span>
                    <span className="field-value">{responseBody.stop_reason || 'N/A'}</span>
                  </div>
                  {responseBody.stop_sequence && (
                    <div className="response-field">
                      <span className="field-label">Stop Sequence:</span>
                      <span className="field-value">{responseBody.stop_sequence}</span>
                    </div>
                  )}
                  {responseBody.usage && (
                    <>
                      <div className="response-field">
                        <span className="field-label">Input Tokens:</span>
                        <span className="field-value">{responseBody.usage.input_tokens || 0}</span>
                      </div>
                      <div className="response-field">
                        <span className="field-label">Output Tokens:</span>
                        <span className="field-value">{responseBody.usage.output_tokens || 0}</span>
                      </div>
                      <div className="response-field">
                        <span className="field-label">Total Tokens:</span>
                        <span className="field-value">
                          {(responseBody.usage.input_tokens || 0) + (responseBody.usage.output_tokens || 0)}
                        </span>
                      </div>
                    </>
                  )}
                  {responseBody.content && (
                    <div className="response-field full-width">
                      <span className="field-label">Response Content:</span>
                      <div className="response-content">
                        {Array.isArray(responseBody.content) ? (
                          responseBody.content.map((block, idx) => (
                            <div key={idx} className="content-block-preview">
                              <strong>{block.type}:</strong>{' '}
                              {block.text ? block.text :
                               block.name ? `Tool: ${block.name}` :
                               JSON.stringify(block, null, 2)}
                            </div>
                          ))
                        ) : (
                          <div>{JSON.stringify(responseBody.content, null, 2)}</div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default MessagesModal

