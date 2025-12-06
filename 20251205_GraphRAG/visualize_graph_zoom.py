import pandas as pd
import networkx as nx
import os
import argparse
import json
from pathlib import Path

# Parse command line arguments
parser = argparse.ArgumentParser(description='Fast graph visualization with zoom')
parser.add_argument('--input-dir', type=str, required=True,
                    help='Directory containing entities.parquet and relationships.parquet')
parser.add_argument('--output-dir', type=str, required=True,
                    help='Directory to save visualization outputs')
parser.add_argument('--title', type=str, default='Knowledge Graph',
                    help='Title for the visualization')
parser.add_argument('--initial-nodes', type=int, default=100,
                    help='Number of top nodes to show initially (default: 100)')
args = parser.parse_args()

# Convert to Path objects
input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)
os.makedirs(output_dir, exist_ok=True)

# Load data
print("Loading data...")
entities_df = pd.read_parquet(input_dir / 'entities.parquet')
relationships_df = pd.read_parquet(input_dir / 'relationships.parquet')
print(f"Loaded {len(entities_df)} entities and {len(relationships_df)} relationships")

# Create graph
G = nx.Graph()
for _, entity in entities_df.iterrows():
    G.add_node(entity['title'], **entity.to_dict())

for _, rel in relationships_df.iterrows():
    if pd.notna(rel['source']) and pd.notna(rel['target']):
        G.add_edge(rel['source'], rel['target'], **rel.to_dict())

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Prepare data for visualization
type_colors = {
    'PERSON': '#FF6B6B',
    'ORGANIZATION': '#4ECDC4',
    'EVENT': '#FFE66D',
    'GEO': '#95E1D3',
}

# Build node and edge lists
nodes_data = []
for node in G.nodes():
    data = G.nodes[node]
    nodes_data.append({
        'id': node,
        'label': node,
        'type': data.get('type', ''),
        'degree': data.get('degree', 0),
        'description': str(data.get('description', ''))[:200],
        'color': type_colors.get(data.get('type', ''), '#CCCCCC'),
        'size': min(data.get('degree', 0) * 2 + 10, 50)
    })

edges_data = []
for source, target in G.edges():
    edge_data = G.edges[source, target]
    edges_data.append({
        'source': source,
        'target': target,
        'weight': edge_data.get('weight', 1.0),
        'description': str(edge_data.get('description', ''))[:200]
    })

# Sort nodes by degree
nodes_data.sort(key=lambda x: x['degree'], reverse=True)
max_degree = nodes_data[0]['degree'] if nodes_data else 0

# Create HTML
html_template = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>TITLE_PLACEHOLDER</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body { margin: 0; font-family: Arial, sans-serif; overflow: hidden; }
        #graph { width: 100vw; height: 100vh; }
        .node { cursor: pointer; }
        .link { stroke: #999; stroke-opacity: 0.6; }
        .node-label { font-size: 10px; pointer-events: none; }
        
        #controls {
            position: fixed; top: 10px; right: 10px;
            background: white; padding: 20px; border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2); z-index: 1000;
            max-width: 300px; max-height: 90vh; overflow-y: auto;
        }
        #controls h3 { margin-top: 0; font-size: 16px; }
        .control-group { margin: 15px 0; }
        .control-group label { display: block; margin-bottom: 5px; font-weight: bold; font-size: 14px; }
        input[type="range"] { width: 100%; }
        .value { color: #4ECDC4; font-weight: bold; }
        button {
            background: #4ECDC4; color: white; border: none;
            padding: 8px 16px; border-radius: 4px; cursor: pointer;
            font-size: 14px; margin: 5px 5px 0 0;
        }
        button:hover { background: #3db8af; }
        .stats { font-size: 12px; color: #666; margin-top: 10px; padding-top: 10px; border-top: 1px solid #ddd; }
        .legend { margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd; }
        .legend-item { display: flex; align-items: center; margin: 5px 0; font-size: 12px; }
        .legend-color { width: 16px; height: 16px; border-radius: 50%; margin-right: 8px; }
        
        #tooltip {
            position: fixed; background: rgba(0,0,0,0.85); color: white;
            padding: 10px; border-radius: 4px; font-size: 12px;
            pointer-events: none; display: none; z-index: 2000;
            max-width: 300px; word-wrap: break-word;
        }
        
        .zoom-hint {
            position: fixed; bottom: 10px; left: 10px;
            background: rgba(0,0,0,0.7); color: white;
            padding: 10px; border-radius: 4px; font-size: 12px;
            z-index: 1000;
        }
    </style>
</head>
<body>
    <svg id="graph"></svg>
    <div id="tooltip"></div>
    <div class="zoom-hint">Use mouse wheel to zoom, drag to pan</div>
    <div id="controls">
        <h3>Graph Controls</h3>
        <div class="control-group">
            <label>Top N Nodes: <span id="top-n-value" class="value">INITIAL_NODES_PLACEHOLDER</span></label>
            <input type="range" id="top-n-slider" min="10" max="MAX_NODES_PLACEHOLDER" value="INITIAL_NODES_PLACEHOLDER" step="10">
        </div>
        <div class="control-group">
            <label>Min Degree: <span id="min-degree-value" class="value">0</span></label>
            <input type="range" id="min-degree-slider" min="0" max="MAX_DEGREE_PLACEHOLDER" value="0" step="1">
        </div>
        <div class="control-group">
            <button onclick="showTop(50)">Top 50</button>
            <button onclick="showTop(100)">Top 100</button>
            <button onclick="showTop(200)">Top 200</button>
            <button onclick="showAll()">Show All</button>
        </div>
        <div class="control-group">
            <button onclick="resetZoom()">Reset Zoom</button>
        </div>
        <div class="stats">
            <div><strong>Visible:</strong> <span id="visible-count">0</span> nodes</div>
            <div><strong>Total:</strong> <span id="total-count">TOTAL_NODES_PLACEHOLDER</span> nodes</div>
        </div>

        <div class="legend">
            <div class="legend-item"><div class="legend-color" style="background: #FF6B6B;"></div><span>Person</span></div>
            <div class="legend-item"><div class="legend-color" style="background: #4ECDC4;"></div><span>Organization</span></div>
            <div class="legend-item"><div class="legend-color" style="background: #FFE66D;"></div><span>Event</span></div>
            <div class="legend-item"><div class="legend-color" style="background: #95E1D3;"></div><span>Geo</span></div>
        </div>
    </div>
    <script>
        const allNodes = NODES_DATA_PLACEHOLDER;
        const allEdges = EDGES_DATA_PLACEHOLDER;
        const initialNodes = INITIAL_NODES_PLACEHOLDER;
        
        let currentNodes = [];
        let currentEdges = [];
        let simulation, svg, g, link, node, label, zoom;
        
        function init() {
            const width = window.innerWidth;
            const height = window.innerHeight;
            
            svg = d3.select("#graph")
                .attr("width", width)
                .attr("height", height);
            
            // Create container group for zoom/pan
            g = svg.append("g");
            
            // Setup zoom behavior
            zoom = d3.zoom()
                .scaleExtent([0.1, 10])
                .on("zoom", (event) => {
                    g.attr("transform", event.transform);
                });
            
            svg.call(zoom);
            
            // Create groups for links, nodes, and labels
            link = g.append("g").attr("class", "links").selectAll("line");
            node = g.append("g").attr("class", "nodes").selectAll("circle");
            label = g.append("g").attr("class", "labels").selectAll("text");
            
            simulation = d3.forceSimulation()
                .force("link", d3.forceLink().id(d => d.id).distance(100))
                .force("charge", d3.forceManyBody().strength(-300))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collision", d3.forceCollide().radius(d => d.size + 5));
            
            filterGraph();
        }
        
        function resetZoom() {
            svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
        }
        
        function filterGraph() {
            const topN = parseInt(document.getElementById('top-n-slider').value);
            const minDegree = parseInt(document.getElementById('min-degree-slider').value);
            
            currentNodes = allNodes
                .filter(n => n.degree >= minDegree)
                .slice(0, topN);
            
            const nodeIds = new Set(currentNodes.map(n => n.id));
            currentEdges = allEdges.filter(e => nodeIds.has(e.source) && nodeIds.has(e.target));
            
            document.getElementById('visible-count').textContent = currentNodes.length;
            updateGraph();
        }
        
        function updateGraph() {
            // Update links
            link = link.data(currentEdges, d => `${d.source}-${d.target}`);
            link.exit().remove();
            link = link.enter().append("line")
                .attr("class", "link")
                .attr("stroke-width", d => Math.min(d.weight * 0.5, 5))
                .merge(link);
            
            // Update nodes
            node = node.data(currentNodes, d => d.id);
            node.exit().remove();
            node = node.enter().append("circle")
                .attr("class", "node")
                .attr("r", d => d.size)
                .attr("fill", d => d.color)
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended))
                .on("mouseover", showTooltip)
                .on("mouseout", hideTooltip)
                .merge(node);
            
            // Update labels (only for top 50 nodes)
            label = label.data(currentNodes.slice(0, 50), d => d.id);
            label.exit().remove();
            label = label.enter().append("text")
                .attr("class", "node-label")
                .text(d => d.label)
                .attr("font-size", "10px")
                .attr("dx", 12)
                .attr("dy", 4)
                .merge(label);
            
            // Update simulation
            simulation.nodes(currentNodes);
            simulation.force("link").links(currentEdges);
            simulation.alpha(1).restart();
            
            simulation.on("tick", () => {
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);
                
                node
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y);
                
                label
                    .attr("x", d => d.x)
                    .attr("y", d => d.y);
            });
        }
        
        function showTooltip(event, d) {
            const tooltip = document.getElementById('tooltip');
            tooltip.innerHTML = `<strong>${d.label}</strong><br>Type: ${d.type}<br>Degree: ${d.degree}<br>${d.description}`;
            tooltip.style.display = 'block';
            tooltip.style.left = (event.pageX + 10) + 'px';
            tooltip.style.top = (event.pageY + 10) + 'px';
        }
        
        function hideTooltip() {
            document.getElementById('tooltip').style.display = 'none';
        }
        
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
        
        function showTop(n) {
            document.getElementById('top-n-slider').value = n;
            document.getElementById('top-n-value').textContent = n;
            document.getElementById('min-degree-slider').value = 0;
            document.getElementById('min-degree-value').textContent = 0;
            filterGraph();
        }
        
        function showAll() {
            document.getElementById('top-n-slider').value = allNodes.length;
            document.getElementById('top-n-value').textContent = allNodes.length;
            document.getElementById('min-degree-slider').value = 0;
            document.getElementById('min-degree-value').textContent = 0;
            filterGraph();
        }
        
        document.getElementById('top-n-slider').addEventListener('input', function(e) {
            document.getElementById('top-n-value').textContent = e.target.value;
            filterGraph();
        });
        
        document.getElementById('min-degree-slider').addEventListener('input', function(e) {
            document.getElementById('min-degree-value').textContent = e.target.value;
            filterGraph();
        });
        
        init();
    </script>
</body>
</html>
'''

# Replace placeholders
html_content = html_template.replace('TITLE_PLACEHOLDER', args.title)
html_content = html_content.replace('NODES_DATA_PLACEHOLDER', json.dumps(nodes_data))
html_content = html_content.replace('EDGES_DATA_PLACEHOLDER', json.dumps(edges_data))
html_content = html_content.replace('INITIAL_NODES_PLACEHOLDER', str(args.initial_nodes))
html_content = html_content.replace('MAX_NODES_PLACEHOLDER', str(len(nodes_data)))
html_content = html_content.replace('MAX_DEGREE_PLACEHOLDER', str(max_degree))
html_content = html_content.replace('TOTAL_NODES_PLACEHOLDER', str(len(nodes_data)))

# Save HTML
html_file = output_dir / 'knowledge_graph.html'
with open(html_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\nVisualization saved to: {html_file}")
print(f"Opens with top {args.initial_nodes} nodes (use --initial-nodes to change)")
print(f"Features: Mouse wheel zoom, drag to pan, drag nodes, hover tooltips")
