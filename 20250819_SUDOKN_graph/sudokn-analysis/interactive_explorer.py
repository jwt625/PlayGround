#!/usr/bin/env python3
"""
Interactive SUDOKN Knowledge Graph Explorer
Provides a command-line interface for exploring the data
"""

from rdflib import Graph
import sys

def load_graph():
    """Load the SUDOKN graph"""
    print("Loading SUDOKN knowledge graph...")
    g = Graph()
    g.parse("../graph/sudokn-triples-NC-7-21-2024.ttl", format="turtle")
    print(f"Loaded {len(g)} triples")
    return g

def search_manufacturers(g, search_term):
    """Search manufacturers by name"""
    query = f"""
    PREFIX sdk: <http://asu.edu/semantics/SUDOKN/>
    PREFIX iof-core: <https://spec.industrialontologies.org/ontology/core/Core/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?manufacturer ?name ?city ?year WHERE {{
        ?manufacturer a iof-core:Manufacturer .
        ?manufacturer rdfs:label ?name .
        FILTER(CONTAINS(LCASE(?name), LCASE("{search_term}")))
        OPTIONAL {{ ?manufacturer sdk:hasOrganizationYearOfEstablishment ?year }}
        OPTIONAL {{ 
            ?manufacturer sdk:organizationLocatedIn ?site .
            ?site sdk:locatedInCity ?city_uri .
            ?city_uri rdfs:label ?city 
        }}
    }}
    """
    
    results = list(g.query(query))
    if results:
        print(f"\nFound {len(results)} manufacturers matching '{search_term}':")
        for i, row in enumerate(results, 1):
            city = row.city if row.city else "Unknown"
            year = row.year if row.year else "Unknown"
            print(f"  {i}. {row.name} - {city} (Est. {year})")
    else:
        print(f"No manufacturers found matching '{search_term}'")
    
    return results

def get_manufacturer_details(g, manufacturer_name):
    """Get detailed information about a specific manufacturer"""
    query = f"""
    PREFIX sdk: <http://asu.edu/semantics/SUDOKN/>
    PREFIX iof-core: <https://spec.industrialontologies.org/ontology/core/Core/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?manufacturer ?name ?city ?year ?capability ?material ?cert ?industry WHERE {{
        ?manufacturer a iof-core:Manufacturer .
        ?manufacturer rdfs:label ?name .
        FILTER(LCASE(?name) = LCASE("{manufacturer_name}"))
        
        OPTIONAL {{ ?manufacturer sdk:hasOrganizationYearOfEstablishment ?year }}
        OPTIONAL {{ 
            ?manufacturer sdk:organizationLocatedIn ?site .
            ?site sdk:locatedInCity ?city_uri .
            ?city_uri rdfs:label ?city 
        }}
        OPTIONAL {{ 
            ?manufacturer sdk:hasProcessCapability ?cap_uri .
            ?cap_uri rdfs:label ?capability 
        }}
        OPTIONAL {{ 
            ?manufacturer sdk:hasMaterialCapability ?mat_uri .
            ?mat_uri rdfs:label ?material 
        }}
        OPTIONAL {{ 
            ?manufacturer sdk:hasCertificate ?cert_uri .
            ?cert_uri rdfs:label ?cert 
        }}
        OPTIONAL {{ 
            ?manufacturer sdk:suppliesToIndustry ?ind_uri .
            ?ind_uri rdfs:label ?industry 
        }}
    }}
    """
    
    results = list(g.query(query))
    if results:
        # Group results by manufacturer
        mfg_data = {
            'name': results[0].name,
            'city': results[0].city if results[0].city else "Unknown",
            'year': results[0].year if results[0].year else "Unknown",
            'capabilities': set(),
            'materials': set(),
            'certifications': set(),
            'industries': set()
        }
        
        for row in results:
            if row.capability:
                mfg_data['capabilities'].add(str(row.capability))
            if row.material:
                mfg_data['materials'].add(str(row.material))
            if row.cert:
                mfg_data['certifications'].add(str(row.cert))
            if row.industry:
                mfg_data['industries'].add(str(row.industry))
        
        print(f"\n=== {mfg_data['name']} ===")
        print(f"Location: {mfg_data['city']}")
        print(f"Established: {mfg_data['year']}")
        
        if mfg_data['capabilities']:
            print(f"\nCapabilities ({len(mfg_data['capabilities'])}):")
            for cap in sorted(mfg_data['capabilities']):
                print(f"  • {cap}")
        
        if mfg_data['materials']:
            print(f"\nMaterial Capabilities ({len(mfg_data['materials'])}):")
            for mat in sorted(mfg_data['materials']):
                print(f"  • {mat}")
        
        if mfg_data['certifications']:
            print(f"\nCertifications ({len(mfg_data['certifications'])}):")
            for cert in sorted(mfg_data['certifications']):
                print(f"  • {cert}")
        
        if mfg_data['industries']:
            print(f"\nIndustries Served ({len(mfg_data['industries'])}):")
            for ind in sorted(mfg_data['industries']):
                print(f"  • {ind}")
    else:
        print(f"No manufacturer found with name '{manufacturer_name}'")

def search_by_capability(g, capability):
    """Find manufacturers with a specific capability"""
    query = f"""
    PREFIX sdk: <http://asu.edu/semantics/SUDOKN/>
    PREFIX iof-core: <https://spec.industrialontologies.org/ontology/core/Core/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?manufacturer ?name ?city WHERE {{
        ?manufacturer a iof-core:Manufacturer .
        ?manufacturer sdk:hasProcessCapability ?cap_uri .
        ?cap_uri rdfs:label ?cap_label .
        FILTER(CONTAINS(LCASE(?cap_label), LCASE("{capability}")))
        OPTIONAL {{ ?manufacturer rdfs:label ?name }}
        OPTIONAL {{ 
            ?manufacturer sdk:organizationLocatedIn ?site .
            ?site sdk:locatedInCity ?city_uri .
            ?city_uri rdfs:label ?city 
        }}
    }}
    """
    
    results = list(g.query(query))
    if results:
        print(f"\nFound {len(results)} manufacturers with '{capability}' capability:")
        for i, row in enumerate(results, 1):
            name = row.name if row.name else "Unknown"
            city = row.city if row.city else "Unknown"
            print(f"  {i}. {name} - {city}")
    else:
        print(f"No manufacturers found with '{capability}' capability")

def search_by_city(g, city):
    """Find manufacturers in a specific city"""
    query = f"""
    PREFIX sdk: <http://asu.edu/semantics/SUDOKN/>
    PREFIX iof-core: <https://spec.industrialontologies.org/ontology/core/Core/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?manufacturer ?name WHERE {{
        ?manufacturer a iof-core:Manufacturer .
        ?manufacturer sdk:organizationLocatedIn ?site .
        ?site sdk:locatedInCity ?city_uri .
        ?city_uri rdfs:label ?city_name .
        FILTER(CONTAINS(LCASE(?city_name), LCASE("{city}")))
        OPTIONAL {{ ?manufacturer rdfs:label ?name }}
    }}
    """
    
    results = list(g.query(query))
    if results:
        print(f"\nFound {len(results)} manufacturers in cities matching '{city}':")
        for i, row in enumerate(results, 1):
            name = row.name if row.name else "Unknown"
            print(f"  {i}. {name}")
    else:
        print(f"No manufacturers found in cities matching '{city}'")

def main():
    g = load_graph()
    
    print("\n=== SUDOKN Interactive Explorer ===")
    print("Commands:")
    print("  search <name>     - Search manufacturers by name")
    print("  details <name>    - Get detailed info about a manufacturer")
    print("  capability <cap>  - Find manufacturers with specific capability")
    print("  city <city>       - Find manufacturers in a city")
    print("  help              - Show this help")
    print("  quit              - Exit")
    
    while True:
        try:
            command = input("\n> ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif command.lower() in ['help', 'h']:
                print("Commands:")
                print("  search <name>     - Search manufacturers by name")
                print("  details <name>    - Get detailed info about a manufacturer")
                print("  capability <cap>  - Find manufacturers with specific capability")
                print("  city <city>       - Find manufacturers in a city")
                print("  help              - Show this help")
                print("  quit              - Exit")
            elif command.startswith('search '):
                search_term = command[7:].strip()
                search_manufacturers(g, search_term)
            elif command.startswith('details '):
                manufacturer_name = command[8:].strip()
                get_manufacturer_details(g, manufacturer_name)
            elif command.startswith('capability '):
                capability = command[11:].strip()
                search_by_capability(g, capability)
            elif command.startswith('city '):
                city = command[5:].strip()
                search_by_city(g, city)
            elif command.strip() == '':
                continue
            else:
                print("Unknown command. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
