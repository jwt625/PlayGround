#!/usr/bin/env python3
"""
Example script showing how to work with the SUDOKN TTL file using RDFLib
"""

from rdflib import Graph, Namespace, RDF, RDFS
from collections import Counter
import pandas as pd

def load_sudokn_graph(ttl_file):
    """Load the SUDOKN TTL file into an RDF graph"""
    g = Graph()
    g.parse(ttl_file, format="turtle")
    print(f"Loaded {len(g)} triples from {ttl_file}")
    return g

def explore_manufacturers(g):
    """Find all manufacturers and their basic info"""
    SDK = Namespace("http://asu.edu/semantics/SUDOKN/")
    IOF_CORE = Namespace("https://spec.industrialontologies.org/ontology/core/Core/")
    
    # SPARQL query to find manufacturers
    query = """
    PREFIX sdk: <http://asu.edu/semantics/SUDOKN/>
    PREFIX iof-core: <https://spec.industrialontologies.org/ontology/core/Core/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?manufacturer ?label ?year ?naics WHERE {
        ?manufacturer a iof-core:Manufacturer .
        OPTIONAL { ?manufacturer rdfs:label ?label }
        OPTIONAL { ?manufacturer sdk:hasOrganizationYearOfEstablishment ?year }
        OPTIONAL { ?manufacturer sdk:hasPrimaryNAICSClassifier ?naics_obj .
                   ?naics_obj sdk:hasNAICSTextValue ?naics }
    }
    """
    
    results = g.query(query)
    manufacturers = []
    
    for row in results:
        manufacturers.append({
            'uri': str(row.manufacturer),
            'name': str(row.label) if row.label else 'Unknown',
            'year_established': str(row.year) if row.year else 'Unknown',
            'naics_description': str(row.naics) if row.naics else 'Unknown'
        })
    
    return manufacturers

def find_capabilities(g):
    """Find all manufacturing capabilities"""
    query = """
    PREFIX sdk: <http://asu.edu/semantics/SUDOKN/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?manufacturer ?manufacturer_name ?capability ?capability_name WHERE {
        ?manufacturer sdk:hasProcessCapability ?capability .
        OPTIONAL { ?manufacturer rdfs:label ?manufacturer_name }
        OPTIONAL { ?capability rdfs:label ?capability_name }
    }
    """
    
    results = g.query(query)
    capabilities = []
    
    for row in results:
        capabilities.append({
            'manufacturer': str(row.manufacturer_name) if row.manufacturer_name else 'Unknown',
            'capability': str(row.capability_name) if row.capability_name else str(row.capability)
        })
    
    return capabilities

def find_by_location(g, city_name=None):
    """Find manufacturers by location"""
    if city_name:
        query = f"""
        PREFIX sdk: <http://asu.edu/semantics/SUDOKN/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?manufacturer ?manufacturer_name ?city WHERE {{
            ?manufacturer sdk:organizationLocatedIn ?site .
            ?site sdk:siteLocatedIn ?location .
            ?location sdk:locatedIn ?city_obj .
            ?city_obj rdfs:label ?city .
            FILTER(CONTAINS(LCASE(?city), LCASE("{city_name}")))
            OPTIONAL {{ ?manufacturer rdfs:label ?manufacturer_name }}
        }}
        """
    else:
        query = """
        PREFIX sdk: <http://asu.edu/semantics/SUDOKN/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?manufacturer ?manufacturer_name ?city WHERE {
            ?manufacturer sdk:organizationLocatedIn ?site .
            ?site sdk:siteLocatedIn ?location .
            ?location sdk:locatedIn ?city_obj .
            ?city_obj rdfs:label ?city .
            OPTIONAL { ?manufacturer rdfs:label ?manufacturer_name }
        }
        """
    
    results = g.query(query)
    locations = []
    
    for row in results:
        locations.append({
            'manufacturer': str(row.manufacturer_name) if row.manufacturer_name else 'Unknown',
            'city': str(row.city) if row.city else 'Unknown'
        })
    
    return locations

def main():
    # Load the graph
    ttl_file = "graph/sudokn-triples-NC-7-21-2024.ttl"
    g = load_sudokn_graph(ttl_file)
    
    print("\n=== SUDOKN Knowledge Graph Analysis ===\n")
    
    # 1. Get all manufacturers
    print("1. Loading manufacturers...")
    manufacturers = explore_manufacturers(g)
    print(f"Found {len(manufacturers)} manufacturers")
    
    # Show first 5 manufacturers
    print("\nFirst 5 manufacturers:")
    for i, mfg in enumerate(manufacturers[:5]):
        print(f"  {i+1}. {mfg['name']} (Est. {mfg['year_established']})")
        print(f"     Industry: {mfg['naics_description']}")
    
    # 2. Analyze capabilities
    print("\n2. Analyzing capabilities...")
    capabilities = find_capabilities(g)
    capability_counts = Counter([cap['capability'] for cap in capabilities])
    
    print("Top 10 most common capabilities:")
    for capability, count in capability_counts.most_common(10):
        print(f"  {capability}: {count} manufacturers")
    
    # 3. Geographic analysis
    print("\n3. Geographic distribution...")
    locations = find_by_location(g)
    city_counts = Counter([loc['city'] for loc in locations])
    
    print("Top 10 cities by number of manufacturers:")
    for city, count in city_counts.most_common(10):
        print(f"  {city}: {count} manufacturers")
    
    # 4. Example: Find manufacturers in a specific city
    print("\n4. Example: Manufacturers in Charlotte...")
    charlotte_mfgs = find_by_location(g, "Charlotte")
    for mfg in charlotte_mfgs[:5]:  # Show first 5
        print(f"  - {mfg['manufacturer']} in {mfg['city']}")

if __name__ == "__main__":
    main()
