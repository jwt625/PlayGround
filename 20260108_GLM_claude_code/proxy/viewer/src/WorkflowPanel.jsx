import { useState, useEffect, useRef } from 'react'
import * as d3 from 'd3'
import { Tooltip } from 'react-tooltip'
import './WorkflowPanel.css'

function WorkflowPanel({ logs, workflowGraph, onRefresh, onSelectLog, selectedLogIndex }) {
  const [showToolEdges, setShowToolEdges] = useState(true)
  const [showSpawnEdges, setShowSpawnEdges] = useState(true)
  const [showContentEdges, setShowContentEdges] = useState(true)
  const svgRef = useRef(null)
  const simulationRef = useRef(null)
  const nodeCirclesRef = useRef(null) // Store reference to node circles for selection updates

  // D3 force simulation - only rebuild when graph data changes
  useEffect(() => {
    if (!workflowGraph?.nodes || workflowGraph.nodes.length === 0 || !svgRef.current) {
      return
    }

    const width = 1200
    const height = 800

    // Prepare data with tooltip content
    const nodes = workflowGraph.nodes.map(node => {
      // Build tooltip content similar to TimelinePanel
      const timestamp = new Date(node.timestamp).toLocaleString()
      const tooltipLines = [
        `Node #${node.id}`,
        timestamp,
        `Agent: ${node.agent_label || node.agent_type}`,
        `Model: ${node.model}`,
        `Session: ${node.session_id}`,
        `Stop: ${node.stop_reason || 'unknown'}`,
        `Tokens: ${node.tokens.total} (${node.tokens.input} in + ${node.tokens.output} out)`,
        `Duration: ${node.duration_ms ? node.duration_ms.toFixed(0) + 'ms' : 'unknown'}`,
      ]

      if (node.tool_count > 0) {
        tooltipLines.push(`Tools: ${node.tool_count}`)
      }
      if (node.subagent_count > 0) {
        tooltipLines.push(`Subagents: ${node.subagent_count}`)
      }
      if (node.has_errors) {
        tooltipLines.push('⚠️ Has errors')
      }

      return {
        ...node,
        id: node.log_index !== undefined ? node.log_index : node.id,
        tooltipContent: tooltipLines.join('\n')
      }
    })

    const edges = (workflowGraph.edges || []).filter(edge => {
      if (edge.type === 'tool_result' && !showToolEdges) return false
      if (edge.type === 'subagent_spawn' && !showSpawnEdges) return false
      if (edge.type === 'content_reuse' && !showContentEdges) return false
      return true
    }).map(edge => {
      // Build tooltip content for edges
      const tooltipLines = []

      if (edge.type === 'tool_result') {
        tooltipLines.push('Tool Dependency')
        tooltipLines.push(`Tool: ${edge.metadata.tool_name || 'unknown'}`)
        tooltipLines.push(`From: Node #${edge.source}`)
        tooltipLines.push(`To: Node #${edge.target}`)
        if (edge.metadata.is_error) {
          tooltipLines.push('⚠️ Tool returned error')
        }
        tooltipLines.push(`Confidence: ${(edge.confidence * 100).toFixed(0)}%`)
      } else if (edge.type === 'subagent_spawn') {
        tooltipLines.push('Subagent Spawn')
        tooltipLines.push(`Type: ${edge.metadata.subagent_type}`)
        tooltipLines.push(`Parent: Node #${edge.source}`)
        tooltipLines.push(`Child: Node #${edge.target}`)
        if (edge.metadata.time_diff_seconds !== undefined) {
          tooltipLines.push(`Time gap: ${edge.metadata.time_diff_seconds.toFixed(1)}s`)
        }
        if (edge.metadata.match_method) {
          tooltipLines.push(`Match: ${edge.metadata.match_method}`)
        }
        tooltipLines.push(`Confidence: ${(edge.confidence * 100).toFixed(0)}%`)
      } else if (edge.type === 'content_reuse') {
        tooltipLines.push('Content Reuse')
        tooltipLines.push(`From: Node #${edge.source}`)
        tooltipLines.push(`To: Node #${edge.target}`)
        if (edge.metadata.content_preview) {
          tooltipLines.push(`Preview: ${edge.metadata.content_preview}`)
        }
        tooltipLines.push(`Match: ${edge.metadata.match_method || 'hash'}`)
        tooltipLines.push(`Confidence: ${(edge.confidence * 100).toFixed(0)}%`)
      }

      return {
        ...edge,
        tooltipContent: tooltipLines.join('\n')
      }
    })

    // Clear previous content
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    // Create container group for zoom/pan
    const g = svg.append('g')

    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform)
      })

    svg.call(zoom)

    // Create arrow markers for edges
    svg.append('defs').selectAll('marker')
      .data(['tool_result', 'subagent_spawn', 'content_reuse'])
      .join('marker')
      .attr('id', d => `arrow-${d}`)
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 25)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', d => {
        if (d === 'tool_result') return '#94a3b8'
        if (d === 'subagent_spawn') return '#f59e0b'
        if (d === 'content_reuse') return '#8b5cf6'
        return '#666'
      })

    // Create force simulation
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(edges)
        .id(d => d.id)
        .distance(150)
        .strength(0.5))
      .force('charge', d3.forceManyBody()
        .strength(-500))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(60))
      .force('x', d3.forceX(width / 2).strength(0.05))
      .force('y', d3.forceY(height / 2).strength(0.05))

    simulationRef.current = simulation

    // Draw edges with tooltips
    const link = g.append('g')
      .selectAll('line')
      .data(edges)
      .join('line')
      .attr('stroke', d => {
        if (d.type === 'tool_result') return '#94a3b8'
        if (d.type === 'subagent_spawn') return '#f59e0b'
        if (d.type === 'content_reuse') return '#8b5cf6'
        return '#666'
      })
      .attr('stroke-width', 2)
      .attr('stroke-opacity', d => d.type === 'content_reuse' ? 0.4 : 0.6)
      .attr('stroke-dasharray', d => d.type === 'content_reuse' ? '5,5' : 'none')
      .attr('marker-end', d => `url(#arrow-${d.type})`)
      .attr('data-tooltip-id', 'workflow-tooltip')
      .attr('data-tooltip-content', d => d.tooltipContent)
      .style('cursor', 'pointer')
      .on('mouseenter', function() {
        d3.select(this).attr('stroke-width', 4).attr('stroke-opacity', 1.0)
      })
      .on('mouseleave', function(_event, d) {
        d3.select(this).attr('stroke-width', 2).attr('stroke-opacity', d.type === 'content_reuse' ? 0.4 : 0.6)
      })

    // Draw nodes with tooltips
    const node = g.append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended))
      .on('click', (_event, d) => {
        if (onSelectLog && logs && logs.length > 0) {
          // Node indices are chronological (0=oldest), but filteredLogs is reversed (0=newest)
          // Must invert the index to select the correct log
          const invertedIndex = logs.length - 1 - d.id
          onSelectLog(invertedIndex)
        }
      })
      .attr('data-tooltip-id', 'workflow-tooltip')
      .attr('data-tooltip-content', d => d.tooltipContent)

    // Node circles
    const nodeCircles = node.append('circle')
      .attr('r', 20)
      .attr('fill', d => d.agent_color || '#6b7280')
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .on('mouseenter', function() {
        d3.select(this).attr('r', 24)
      })
      .on('mouseleave', function() {
        d3.select(this).attr('r', 20)
      })

    // Store reference for selection updates
    nodeCirclesRef.current = nodeCircles

    // Node labels
    node.append('text')
      .text(d => `#${d.id}`)
      .attr('text-anchor', 'middle')
      .attr('dy', 4)
      .attr('fill', '#fff')
      .attr('font-size', '10px')
      .attr('pointer-events', 'none')

    // Node type labels
    node.append('text')
      .text(d => d.agent_label || d.agent_type || '')
      .attr('text-anchor', 'middle')
      .attr('dy', 35)
      .attr('fill', '#374151')
      .attr('font-size', '11px')
      .attr('pointer-events', 'none')

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => {
          const source = nodes.find(n => n.id === d.source.id || n.id === d.source)
          return source?.x || 0
        })
        .attr('y1', d => {
          const source = nodes.find(n => n.id === d.source.id || n.id === d.source)
          return source?.y || 0
        })
        .attr('x2', d => {
          const target = nodes.find(n => n.id === d.target.id || n.id === d.target)
          return target?.x || 0
        })
        .attr('y2', d => {
          const target = nodes.find(n => n.id === d.target.id || n.id === d.target)
          return target?.y || 0
        })

      node.attr('transform', d => `translate(${d.x},${d.y})`)
    })

    // Drag functions
    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart()
      d.fx = d.x
      d.fy = d.y
    }

    function dragged(event, d) {
      d.fx = event.x
      d.fy = event.y
    }

    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0)
      d.fx = null
      d.fy = null
    }

    // Cleanup
    return () => {
      simulation.stop()
    }
  }, [workflowGraph, showToolEdges, showSpawnEdges])

  // Separate effect to update selection styling without rebuilding the graph
  useEffect(() => {
    if (!nodeCirclesRef.current) return

    // Update all node circles to reflect current selection
    nodeCirclesRef.current
      .attr('stroke', d => (selectedLogIndex !== null && d.id === selectedLogIndex) ? '#fbbf24' : '#fff')
      .attr('stroke-width', d => (selectedLogIndex !== null && d.id === selectedLogIndex) ? 4 : 2)
  }, [selectedLogIndex])

  if (!workflowGraph || !workflowGraph.nodes || workflowGraph.nodes.length === 0) {
    return (
      <div className="workflow-panel">
        <div className="workflow-empty">
          No workflow data available
          {onRefresh && (
            <button onClick={onRefresh} style={{ marginLeft: '10px' }}>
              Refresh Workflow
            </button>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="workflow-panel">
      <div className="workflow-controls">
        <label>
          <input
            type="checkbox"
            checked={showToolEdges}
            onChange={(e) => setShowToolEdges(e.target.checked)}
          />
          <span style={{ color: '#94a3b8' }}>●</span> Tool Dependencies
        </label>
        <label>
          <input
            type="checkbox"
            checked={showSpawnEdges}
            onChange={(e) => setShowSpawnEdges(e.target.checked)}
          />
          <span style={{ color: '#f59e0b' }}>●</span> Subagent Spawns
        </label>
        <label>
          <input
            type="checkbox"
            checked={showContentEdges}
            onChange={(e) => setShowContentEdges(e.target.checked)}
          />
          <span style={{ color: '#8b5cf6' }}>●</span> Content Reuse
        </label>
        <span className="workflow-stats">
          {workflowGraph.nodes.length} nodes, {workflowGraph.edges.length} edges
        </span>
        {onRefresh && (
          <button onClick={onRefresh} title="Rebuild workflow graph from latest logs">
            Refresh
          </button>
        )}
      </div>

      <div className="workflow-canvas">
        <svg ref={svgRef} width="100%" height="800" style={{ border: '1px solid #e5e7eb' }}>
          {/* D3 will render the graph here */}
        </svg>
      </div>

      <Tooltip
        id="workflow-tooltip"
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

export default WorkflowPanel

