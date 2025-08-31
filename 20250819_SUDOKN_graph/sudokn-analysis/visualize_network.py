#!/usr/bin/env python3
"""
SUDOKN Knowledge Graph Visualization
Creates maps and network graphs of the manufacturing data
"""

from rdflib import Graph
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter, defaultdict
import numpy as np

def load_graph():
    """Load the SUDOKN graph"""
    g = Graph()
    g.parse("../graph/sudokn-triples-NC-7-21-2024.ttl", format="turtle")
    return g

def get_manufacturer_coordinates(g):
    """Get all manufacturers with their coordinates and basic info"""
    query = """
    PREFIX sdk: <http://asu.edu/semantics/SUDOKN/>
    PREFIX iof-core: <https://spec.industrialontologies.org/ontology/core/Core/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?manufacturer ?name ?lat ?lon ?city WHERE {
        ?manufacturer a iof-core:Manufacturer .
        ?manufacturer sdk:organizationLocatedIn ?site .
        ?site sdk:hasSpatialCoordinates ?coordinates .
        ?coordinates sdk:hasLatitudeValue ?lat .
        ?coordinates sdk:hasLongitudeValue ?lon .
        ?site sdk:locatedInCity ?city_uri .
        ?city_uri rdfs:label ?city .
        OPTIONAL { ?manufacturer rdfs:label ?name }
    }
    """
    
    results = g.query(query)
    data = []
    
    for row in results:
        try:
            lat = float(row.lat)
            lon = float(row.lon)
            data.append({
                'manufacturer_uri': str(row.manufacturer),
                'name': str(row.name) if row.name else 'Unknown',
                'lat': lat,
                'lon': lon,
                'city': str(row.city)
            })
        except (ValueError, TypeError):
            # Skip entries with invalid coordinates
            print(f"Skipping {row.name} due to invalid coordinates: {row.lat}, {row.lon}")
            continue
    
    return pd.DataFrame(data)

def get_manufacturer_capabilities(g):
    """Get manufacturers with their capabilities"""
    query = """
    PREFIX sdk: <http://asu.edu/semantics/SUDOKN/>
    PREFIX iof-core: <https://spec.industrialontologies.org/ontology/core/Core/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?manufacturer ?name ?capability WHERE {
        ?manufacturer a iof-core:Manufacturer .
        ?manufacturer sdk:hasProcessCapability ?cap_uri .
        ?cap_uri rdfs:label ?capability .
        OPTIONAL { ?manufacturer rdfs:label ?name }
    }
    """
    
    results = g.query(query)
    data = []
    
    for row in results:
        data.append({
            'manufacturer_uri': str(row.manufacturer),
            'name': str(row.name) if row.name else 'Unknown',
            'capability': str(row.capability)
        })
    
    return pd.DataFrame(data)

def create_geographic_map(df_coords, df_caps=None):
    """Create a geographic scatter plot of manufacturers"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Basic geographic plot
    ax1.scatter(df_coords['lon'], df_coords['lat'], alpha=0.6, s=50)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('SUDOKN Manufacturers in North Carolina')
    ax1.grid(True, alpha=0.3)
    
    # Add city labels for major clusters
    city_centers = df_coords.groupby('city')[['lat', 'lon']].mean()
    city_counts = df_coords['city'].value_counts()
    
    for city, count in city_counts.head(10).items():
        if city in city_centers.index:
            center = city_centers.loc[city]
            ax1.annotate(f'{city} ({count})', 
                        (center['lon'], center['lat']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
    
    # Capability-colored map
    if df_caps is not None:
        # Get most common capability for each manufacturer
        mfg_main_cap = df_caps.groupby('manufacturer_uri')['capability'].first().to_dict()
        df_coords['main_capability'] = df_coords['manufacturer_uri'].map(mfg_main_cap)
        
        # Color by top capabilities
        top_capabilities = df_caps['capability'].value_counts().head(8).index
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_capabilities)))
        color_map = dict(zip(top_capabilities, colors))
        
        for i, cap in enumerate(top_capabilities):
            mask = df_coords['main_capability'] == cap
            if mask.any():
                ax2.scatter(df_coords.loc[mask, 'lon'], 
                           df_coords.loc[mask, 'lat'],
                           c=[color_map[cap]], label=cap, alpha=0.7, s=50)
        
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title('Manufacturers by Primary Capability')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sudokn_geographic_map.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_capability_network(g):
    """Create a network graph of manufacturers and capabilities"""
    query = """
    PREFIX sdk: <http://asu.edu/semantics/SUDOKN/>
    PREFIX iof-core: <https://spec.industrialontologies.org/ontology/core/Core/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?manufacturer ?name ?capability WHERE {
        ?manufacturer a iof-core:Manufacturer .
        ?manufacturer sdk:hasProcessCapability ?cap_uri .
        ?cap_uri rdfs:label ?capability .
        OPTIONAL { ?manufacturer rdfs:label ?name }
    }
    """
    
    results = g.query(query)
    
    # Create bipartite network
    G = nx.Graph()
    
    for row in results:
        mfg_name = str(row.name) if row.name else 'Unknown'
        capability = str(row.capability)
        
        # Add nodes with types
        G.add_node(mfg_name, node_type='manufacturer', size=1)
        G.add_node(capability, node_type='capability', size=1)
        
        # Add edge
        G.add_edge(mfg_name, capability)
    
    # Update capability node sizes based on connections
    for node in G.nodes():
        if G.nodes[node]['node_type'] == 'capability':
            G.nodes[node]['size'] = G.degree(node)
    
    return G

def visualize_capability_network(G, layout_type='spring'):
    """Visualize the capability network"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Separate nodes by type
    manufacturers = [n for n, d in G.nodes(data=True) if d['node_type'] == 'manufacturer']
    capabilities = [n for n, d in G.nodes(data=True) if d['node_type'] == 'capability']
    
    # Create layout
    if layout_type == 'spring':
        pos = nx.spring_layout(G, k=1, iterations=50)
    elif layout_type == 'bipartite':
        pos = nx.bipartite_layout(G, manufacturers)
    else:
        pos = nx.circular_layout(G)
    
    # Plot 1: Full network (sample for readability)
    # Sample nodes to avoid overcrowding
    sample_caps = sorted(capabilities, key=lambda x: G.degree(x), reverse=True)[:15]
    sample_mfgs = []
    for cap in sample_caps:
        sample_mfgs.extend(list(G.neighbors(cap))[:3])  # Top 3 manufacturers per capability
    
    sample_nodes = set(sample_caps + sample_mfgs)
    G_sample = G.subgraph(sample_nodes)
    pos_sample = {n: pos[n] for n in sample_nodes if n in pos}
    
    # Draw sample network
    mfg_nodes = [n for n in G_sample.nodes() if n in manufacturers]
    cap_nodes = [n for n in G_sample.nodes() if n in capabilities]
    
    nx.draw_networkx_nodes(G_sample, pos_sample, nodelist=mfg_nodes, 
                          node_color='lightblue', node_size=100, ax=ax1)
    nx.draw_networkx_nodes(G_sample, pos_sample, nodelist=cap_nodes,
                          node_color='lightcoral', 
                          node_size=[G.degree(n)*20 for n in cap_nodes], ax=ax1)
    nx.draw_networkx_edges(G_sample, pos_sample, alpha=0.3, ax=ax1)
    nx.draw_networkx_labels(G_sample, pos_sample, 
                           {n: n for n in cap_nodes}, font_size=8, ax=ax1)
    
    ax1.set_title('Sample Manufacturer-Capability Network')
    ax1.axis('off')
    
    # Plot 2: Capability degree distribution
    cap_degrees = [G.degree(cap) for cap in capabilities]
    ax2.hist(cap_degrees, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Manufacturers')
    ax2.set_ylabel('Number of Capabilities')
    ax2.set_title('Distribution of Capability Popularity')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sudokn_capability_network.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_city_capability_heatmap(df_coords, df_caps):
    """Create a heatmap of capabilities by city"""
    # Merge dataframes
    df_merged = df_coords.merge(df_caps, on='manufacturer_uri', suffixes=('', '_cap'))
    
    # Create city-capability matrix
    city_cap_counts = df_merged.groupby(['city', 'capability']).size().unstack(fill_value=0)
    
    # Focus on top cities and capabilities
    top_cities = df_coords['city'].value_counts().head(10).index
    top_capabilities = df_caps['capability'].value_counts().head(10).index
    
    heatmap_data = city_cap_counts.loc[top_cities, top_capabilities]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(top_capabilities)))
    ax.set_yticks(range(len(top_cities)))
    ax.set_xticklabels(top_capabilities, rotation=45, ha='right')
    ax.set_yticklabels(top_cities)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Number of Manufacturers')
    
    # Add text annotations
    for i in range(len(top_cities)):
        for j in range(len(top_capabilities)):
            text = ax.text(j, i, int(heatmap_data.iloc[i, j]),
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Manufacturing Capabilities by City')
    plt.tight_layout()
    plt.savefig('sudokn_city_capability_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("Loading SUDOKN knowledge graph...")
    g = load_graph()
    
    print("Extracting coordinate data...")
    df_coords = get_manufacturer_coordinates(g)
    print(f"Found {len(df_coords)} manufacturers with coordinates")
    
    print("Extracting capability data...")
    df_caps = get_manufacturer_capabilities(g)
    print(f"Found {len(df_caps)} manufacturer-capability relationships")
    
    print("\n1. Creating geographic maps...")
    create_geographic_map(df_coords, df_caps)
    
    print("2. Creating capability network...")
    network = create_capability_network(g)
    print(f"Network has {network.number_of_nodes()} nodes and {network.number_of_edges()} edges")
    visualize_capability_network(network)
    
    print("3. Creating city-capability heatmap...")
    create_city_capability_heatmap(df_coords, df_caps)
    
    print("\nVisualization complete! Check the generated PNG files.")

if __name__ == "__main__":
    main()
