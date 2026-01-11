import { useState, useEffect, useRef } from 'react'
import * as d3 from 'd3'
import './WorkflowPanel.css'

function WorkflowPanel({ logs, workflowGraph, onRefresh }) {
  const [showToolEdges, setShowToolEdges] = useState(true)
  const [showSpawnEdges, setShowSpawnEdges] = useState(true)
  const [selectedNode, setSelectedNode] = useState(null)
  const svgRef = useRef(null)
  const simulationRef = useRef(null)

  // D3 force simulation
  useEffect(() => {
    if (!workflowGraph?.nodes || workflowGraph.nodes.length === 0 || !svgRef.current) {
      return
    }

    const width = 1200
    const height = 800

    // Prepare data
    const nodes = workflowGraph.nodes.map(node => ({
      ...node,
      id: node.log_index !== undefined ? node.log_index : node.id,
    }))

    const edges = (workflowGraph.edges || []).filter(edge => {
      if (edge.type === 'tool_result' && !showToolEdges) return false
      if (edge.type === 'subagent_spawn' && !showSpawnEdges) return false
      return true
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
      .data(['tool_result', 'subagent_spawn'])
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
      .attr('fill', d => d === 'tool_result' ? '#94a3b8' : '#f59e0b')

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

    // Draw edges
    const link = g.append('g')
      .selectAll('line')
      .data(edges)
      .join('line')
      .attr('stroke', d => d.type === 'tool_result' ? '#94a3b8' : '#f59e0b')
      .attr('stroke-width', 2)
      .attr('stroke-opacity', 0.6)
      .attr('marker-end', d => `url(#arrow-${d.type})`)

    // Draw nodes
    const node = g.append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended))
      .on('click', (event, d) => {
        setSelectedNode(d.id)
      })

    // Node circles
    node.append('circle')
      .attr('r', 20)
      .attr('fill', d => d.agent_color || '#6b7280')
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')

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
          Tool Dependencies
        </label>
        <label>
          <input
            type="checkbox"
            checked={showSpawnEdges}
            onChange={(e) => setShowSpawnEdges(e.target.checked)}
          />
          Subagent Spawns
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
    </div>
  )
}

export default WorkflowPanel

