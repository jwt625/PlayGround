import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import Counter
import os
import argparse
from pathlib import Path
from pyvis.network import Network

# Parse command line arguments
parser = argparse.ArgumentParser(description='Visualize GraphRAG knowledge graph')
parser.add_argument('--input-dir', type=str, default='./christmas/output',
                    help='Directory containing entities.parquet and relationships.parquet (default: ./christmas/output)')
parser.add_argument('--output-dir', type=str, default='./christmas/plot',
                    help='Directory to save visualization outputs (default: ./christmas/plot)')
parser.add_argument('--title', type=str, default='Knowledge Graph',
                    help='Title for the visualization (default: Knowledge Graph)')
args = parser.parse_args()

# Convert to Path objects
input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load entities and relationships
print("Loading data...")
entities_file = input_dir / 'entities.parquet'
relationships_file = input_dir / 'relationships.parquet'

if not entities_file.exists():
    raise FileNotFoundError(f"Entities file not found: {entities_file}")
if not relationships_file.exists():
    raise FileNotFoundError(f"Relationships file not found: {relationships_file}")

entities_df = pd.read_parquet(entities_file)
relationships_df = pd.read_parquet(relationships_file)

print(f"Loaded {len(entities_df)} entities and {len(relationships_df)} relationships")

# Create a directed graph
G = nx.Graph()

# Add nodes with attributes
for _, entity in entities_df.iterrows():
    G.add_node(
        entity['title'],
        type=entity['type'],
        degree=entity['degree'],
        description=entity['description'][:100] if pd.notna(entity['description']) else ""
    )

# Add edges with attributes
for _, rel in relationships_df.iterrows():
    if pd.notna(rel['source']) and pd.notna(rel['target']):
        G.add_edge(
            rel['source'],
            rel['target'],
            weight=rel.get('weight', 1.0) if 'weight' in rel else 1.0,
            description=rel.get('description', '') if pd.notna(rel.get('description')) else ''
        )

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Get top nodes by degree
top_nodes = entities_df.nlargest(20, 'degree')['title'].tolist()

# Create subgraph with top nodes and their neighbors
subgraph_nodes = set(top_nodes)
for node in top_nodes:
    if node in G:
        subgraph_nodes.update(G.neighbors(node))

subgraph = G.subgraph(subgraph_nodes)
print(f"Subgraph has {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges")

# Set up the plot
plt.figure(figsize=(20, 16))

# Use spring layout for positioning
pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)

# Get node colors based on type
entity_types = nx.get_node_attributes(subgraph, 'type')
type_colors = {
    'PERSON': '#FF6B6B',
    'ORGANIZATION': '#4ECDC4',
    'EVENT': '#FFE66D',
    'GEO': '#95E1D3',
}
node_colors = [type_colors.get(entity_types.get(node, 'OTHER'), '#CCCCCC') for node in subgraph.nodes()]

# Get node sizes based on degree
degrees = dict(subgraph.degree())
node_sizes = [degrees[node] * 100 + 300 for node in subgraph.nodes()]

# Draw the graph
nx.draw_networkx_edges(subgraph, pos, alpha=0.2, width=1, edge_color='gray')
nx.draw_networkx_nodes(
    subgraph, pos,
    node_color=node_colors,
    node_size=node_sizes,
    alpha=0.8,
    linewidths=2,
    edgecolors='black'
)

# Draw labels for top nodes only
labels = {node: node for node in top_nodes if node in subgraph}
nx.draw_networkx_labels(
    subgraph, pos,
    labels,
    font_size=10,
    font_weight='bold',
    font_color='black'
)

plt.title(f"{args.title}\n(Top 20 entities and their connections)",
          fontsize=16, fontweight='bold', pad=20)

# Add legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
               markersize=10, label=entity_type)
    for entity_type, color in type_colors.items()
]
plt.legend(handles=legend_elements, loc='upper left', fontsize=12)

plt.axis('off')
plt.tight_layout()

# Save the static PNG figure
png_file = output_dir / 'knowledge_graph.png'
plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nStatic graph visualization saved to: {png_file}")
plt.close()

# Print statistics
print("\n" + "="*80)
print("GRAPH STATISTICS")
print("="*80)
print(f"Total nodes: {G.number_of_nodes()}")
print(f"Total edges: {G.number_of_edges()}")
print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
print(f"\nEntity type distribution:")
type_counts = Counter(entity_types.values())
for entity_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {entity_type}: {count}")

print(f"\nTop 10 entities by degree:")
for row in top_nodes[:10]:
    entity = entities_df[entities_df['title'] == row].iloc[0]
    print(f"  {entity['title']} ({entity['type']}): degree={entity['degree']}")

# Create interactive HTML visualization using pyvis
print("\n" + "="*80)
print("Creating interactive HTML visualization...")
print("="*80)

net = Network(height='800px', width='100%', bgcolor='#ffffff', font_color='black')
net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=200, spring_strength=0.001)

# Add nodes to pyvis network
type_color_map = {
    'PERSON': '#FF6B6B',
    'ORGANIZATION': '#4ECDC4',
    'EVENT': '#FFE66D',
    'GEO': '#95E1D3',
}

for node in subgraph.nodes():
    node_type = entity_types.get(node, 'OTHER')
    color = type_color_map.get(node_type, '#CCCCCC')
    degree = degrees[node]
    size = min(degree * 3 + 10, 50)  # Cap size at 50

    # Get description
    entity_row = entities_df[entities_df['title'] == node]
    description = ""
    if len(entity_row) > 0:
        desc = entity_row.iloc[0]['description']
        description = desc[:200] if pd.notna(desc) else ""

    title = f"{node}\nType: {node_type}\nDegree: {degree}\n{description}"

    net.add_node(
        node,
        label=node,
        title=title,
        color=color,
        size=size,
        font={'size': 12 if node in top_nodes else 8}
    )

# Add edges to pyvis network with hover information
for edge in subgraph.edges(data=True):
    source, target, edge_data = edge[0], edge[1], edge[2]

    # Get edge attributes
    weight = edge_data.get('weight', 1.0)
    description = edge_data.get('description', '')

    # Create hover title for edge
    edge_title = f"{source} → {target}\nWeight: {weight:.1f}"
    if description:
        edge_title += f"\nRelationship: {description[:200]}"

    net.add_edge(
        source,
        target,
        color='#CCCCCC',
        title=edge_title,
        width=min(weight * 0.5, 5)  # Scale edge width by weight, cap at 5
    )

# Save interactive HTML
html_file = output_dir / 'knowledge_graph.html'
net.save_graph(str(html_file))
print(f"Interactive graph visualization saved to: {html_file}")
print("\nVisualization complete! Open the HTML file in a browser to explore the graph interactively.")

