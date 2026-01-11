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

  // Get messages for this specific request
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
            {resultText.substring(0, 500)}{resultText.length > 500 ? '...' : ''}
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

          {/* Messages */}
          {requestMessages.map((msg, idx) => {
            const roleClass = msg.role === 'user' ? 'user' : 'assistant'
            const roleColor = msg.role === 'user' ? '#569cd6' : '#4ec9b0'

            return (
              <div key={idx} className={`message-item ${roleClass}`}>
                <div className="message-header">
                  <span className="message-role" style={{ color: roleColor }}>
                    {msg.role}
                  </span>
                  <span style={{ color: '#858585' }}>{msg.timestamp || ''}</span>
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
      </div>
    </div>
  )
}

export default MessagesModal

