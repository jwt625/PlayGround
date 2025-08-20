#!/usr/bin/env python3
"""
Advanced SUDOKN Knowledge Graph Analysis with more sophisticated queries
"""

from rdflib import Graph, Namespace, RDF, RDFS
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def load_sudokn_graph(ttl_file):
    """Load the SUDOKN TTL file into an RDF graph"""
    g = Graph()
    g.parse(ttl_file, format="turtle")
    print(f"Loaded {len(g)} triples from {ttl_file}")
    return g

def find_manufacturers_by_capability(g, capability_name):
    """Find manufacturers with a specific capability"""
    query = f"""
    PREFIX sdk: <http://asu.edu/semantics/SUDOKN/>
    PREFIX iof-core: <https://spec.industrialontologies.org/ontology/core/Core/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?manufacturer ?manufacturer_name ?city_name WHERE {{
        ?manufacturer a iof-core:Manufacturer .
        ?manufacturer sdk:hasProcessCapability ?capability .
        ?capability rdfs:label ?cap_label .
        FILTER(CONTAINS(LCASE(?cap_label), LCASE("{capability_name}")))
        OPTIONAL {{ ?manufacturer rdfs:label ?manufacturer_name }}
        OPTIONAL {{ 
            ?manufacturer sdk:organizationLocatedIn ?site .
            ?site sdk:locatedInCity ?city_uri .
            ?city_uri rdfs:label ?city_name 
        }}
    }}
    """
    
    results = g.query(query)
    manufacturers = []
    
    for row in results:
        manufacturers.append({
            'name': str(row.manufacturer_name) if row.manufacturer_name else 'Unknown',
            'city': str(row.city_name) if row.city_name else 'Unknown'
        })
    
    return manufacturers

def find_manufacturers_by_material(g, material_name):
    """Find manufacturers that can process a specific material"""
    query = f"""
    PREFIX sdk: <http://asu.edu/semantics/SUDOKN/>
    PREFIX iof-core: <https://spec.industrialontologies.org/ontology/core/Core/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?manufacturer ?manufacturer_name ?material_cap ?city_name WHERE {{
        ?manufacturer a iof-core:Manufacturer .
        ?manufacturer sdk:hasMaterialCapability ?material_cap .
        ?material_cap rdfs:label ?mat_label .
        FILTER(CONTAINS(LCASE(?mat_label), LCASE("{material_name}")))
        OPTIONAL {{ ?manufacturer rdfs:label ?manufacturer_name }}
        OPTIONAL {{ 
            ?manufacturer sdk:organizationLocatedIn ?site .
            ?site sdk:locatedInCity ?city_uri .
            ?city_uri rdfs:label ?city_name 
        }}
    }}
    """
    
    results = g.query(query)
    manufacturers = []
    
    for row in results:
        manufacturers.append({
            'name': str(row.manufacturer_name) if row.manufacturer_name else 'Unknown',
            'city': str(row.city_name) if row.city_name else 'Unknown',
            'material_capability': str(row.material_cap)
        })
    
    return manufacturers

def analyze_certifications(g):
    """Analyze ISO certifications across manufacturers"""
    query = """
    PREFIX sdk: <http://asu.edu/semantics/SUDOKN/>
    PREFIX iof-core: <https://spec.industrialontologies.org/ontology/core/Core/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?manufacturer ?manufacturer_name ?cert ?cert_label WHERE {
        ?manufacturer a iof-core:Manufacturer .
        ?manufacturer sdk:hasCertificate ?cert .
        OPTIONAL { ?manufacturer rdfs:label ?manufacturer_name }
        OPTIONAL { ?cert rdfs:label ?cert_label }
    }
    """
    
    results = g.query(query)
    certifications = []
    
    for row in results:
        certifications.append({
            'manufacturer': str(row.manufacturer_name) if row.manufacturer_name else 'Unknown',
            'certification': str(row.cert_label) if row.cert_label else str(row.cert)
        })
    
    return certifications

def find_supply_chain_connections(g):
    """Find which industries manufacturers supply to"""
    query = """
    PREFIX sdk: <http://asu.edu/semantics/SUDOKN/>
    PREFIX iof-core: <https://spec.industrialontologies.org/ontology/core/Core/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?manufacturer ?manufacturer_name ?industry ?industry_label WHERE {
        ?manufacturer a iof-core:Manufacturer .
        ?manufacturer sdk:suppliesToIndustry ?industry .
        OPTIONAL { ?manufacturer rdfs:label ?manufacturer_name }
        OPTIONAL { ?industry rdfs:label ?industry_label }
    }
    """
    
    results = g.query(query)
    connections = []
    
    for row in results:
        connections.append({
            'manufacturer': str(row.manufacturer_name) if row.manufacturer_name else 'Unknown',
            'industry': str(row.industry_label) if row.industry_label else str(row.industry)
        })
    
    return connections

def create_capability_network(g):
    """Create a network graph of manufacturers and their capabilities"""
    query = """
    PREFIX sdk: <http://asu.edu/semantics/SUDOKN/>
    PREFIX iof-core: <https://spec.industrialontologies.org/ontology/core/Core/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?manufacturer ?manufacturer_name ?capability ?capability_name WHERE {
        ?manufacturer a iof-core:Manufacturer .
        ?manufacturer sdk:hasProcessCapability ?capability .
        OPTIONAL { ?manufacturer rdfs:label ?manufacturer_name }
        OPTIONAL { ?capability rdfs:label ?capability_name }
    }
    """
    
    results = g.query(query)
    
    # Create NetworkX graph
    G = nx.Graph()
    
    for row in results:
        mfg_name = str(row.manufacturer_name) if row.manufacturer_name else 'Unknown'
        cap_name = str(row.capability_name) if row.capability_name else str(row.capability)
        
        # Add nodes
        G.add_node(mfg_name, node_type='manufacturer')
        G.add_node(cap_name, node_type='capability')
        
        # Add edge
        G.add_edge(mfg_name, cap_name)
    
    return G

def main():
    # Load the graph
    ttl_file = "../graph/sudokn-triples-NC-7-21-2024.ttl"
    g = load_sudokn_graph(ttl_file)
    
    print("\n=== Advanced SUDOKN Knowledge Graph Analysis ===\n")
    
    # 1. Find manufacturers by specific capability
    print("1. Manufacturers with CNC Machining capability:")
    cnc_manufacturers = find_manufacturers_by_capability(g, "CNC")
    for mfg in cnc_manufacturers[:10]:  # Show first 10
        print(f"  - {mfg['name']} ({mfg['city']})")
    
    # 2. Find manufacturers by material capability
    print(f"\n2. Manufacturers that can process Aluminum:")
    aluminum_manufacturers = find_manufacturers_by_material(g, "Aluminum")
    print(f"Found {len(aluminum_manufacturers)} manufacturers")
    for mfg in aluminum_manufacturers[:10]:  # Show first 10
        print(f"  - {mfg['name']} ({mfg['city']})")
    
    # 3. Certification analysis
    print(f"\n3. Certification Analysis:")
    certifications = analyze_certifications(g)
    cert_counts = Counter([cert['certification'] for cert in certifications])
    print("Most common certifications:")
    for cert, count in cert_counts.most_common(10):
        print(f"  {cert}: {count} manufacturers")
    
    # 4. Supply chain analysis
    print(f"\n4. Supply Chain Analysis:")
    connections = find_supply_chain_connections(g)
    industry_counts = Counter([conn['industry'] for conn in connections])
    print("Industries served:")
    for industry, count in industry_counts.most_common():
        print(f"  {industry}: {count} manufacturers")
    
    # 5. Create and analyze capability network
    print(f"\n5. Network Analysis:")
    network = create_capability_network(g)
    print(f"Network has {network.number_of_nodes()} nodes and {network.number_of_edges()} edges")
    
    # Find most connected capabilities
    capability_nodes = [n for n, d in network.nodes(data=True) if d.get('node_type') == 'capability']
    capability_degrees = [(node, network.degree(node)) for node in capability_nodes]
    capability_degrees.sort(key=lambda x: x[1], reverse=True)
    
    print("Most connected capabilities:")
    for cap, degree in capability_degrees[:10]:
        print(f"  {cap}: connected to {degree} manufacturers")

if __name__ == "__main__":
    main()
