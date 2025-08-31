# SUDOKN Knowledge Graph Analysis

This project provides tools and examples for analyzing the SUDOKN (Supply Chain Data and Ontology Knowledge Network) knowledge graph data in TTL (Turtle) format.

## What is the SUDOKN TTL File?

The `sudokn-triples-NC-7-21-2024.ttl` file contains a knowledge graph with:
- **304 Manufacturers** in North Carolina with detailed information
- **Manufacturing capabilities** (machining, fabrication, welding, etc.)
- **Material processing capabilities** (aluminum, steel, stainless steel, etc.)
- **Geographic locations** and postal addresses
- **Industry certifications** (ISO9001, ISO9000, AS9100, etc.)
- **NAICS industry classifications**
- **Supply chain relationships** (which industries they serve)

## Quick Start

1. **Setup the environment:**
   ```bash
   # The project uses uv for package management
   cd sudokn-analysis
   uv run python main.py
   ```

2. **Run basic analysis:**
   ```bash
   uv run python main.py
   ```

3. **Run advanced analysis:**
   ```bash
   uv run python advanced_analysis.py
   ```

## Key Analysis Capabilities

### 1. **Basic Manufacturer Analysis** (`main.py`)
- List all manufacturers with establishment years and NAICS codes
- Analyze manufacturing capabilities distribution
- Geographic distribution by city
- Search manufacturers by location

### 2. **Advanced Analysis** (`advanced_analysis.py`)
- Find manufacturers by specific capabilities (e.g., "CNC Machining")
- Find manufacturers by material processing (e.g., "Aluminum")
- Certification analysis across the supply chain
- Industry supply relationships
- Network analysis of capability connections

## Example Queries and Use Cases

### Find CNC Machining Capabilities
```python
from rdflib import Graph
g = Graph()
g.parse("../graph/sudokn-triples-NC-7-21-2024.ttl", format="turtle")

query = """
PREFIX sdk: <http://asu.edu/semantics/SUDOKN/>
PREFIX iof-core: <https://spec.industrialontologies.org/ontology/core/Core/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?manufacturer ?name ?city WHERE {
    ?manufacturer a iof-core:Manufacturer .
    ?manufacturer sdk:hasProcessCapability ?capability .
    ?capability rdfs:label ?cap_label .
    FILTER(CONTAINS(LCASE(?cap_label), "cnc"))
    OPTIONAL { ?manufacturer rdfs:label ?name }
    OPTIONAL {
        ?manufacturer sdk:organizationLocatedIn ?site .
        ?site sdk:locatedInCity ?city_uri .
        ?city_uri rdfs:label ?city
    }
}
"""

for row in g.query(query):
    print(f"{row.name} in {row.city}")
```

### Find Aerospace Suppliers
```python
query = """
PREFIX sdk: <http://asu.edu/semantics/SUDOKN/>
PREFIX iof-core: <https://spec.industrialontologies.org/ontology/core/Core/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?manufacturer ?name ?cert WHERE {
    ?manufacturer a iof-core:Manufacturer .
    ?manufacturer sdk:suppliesToIndustry ?industry .
    ?industry rdfs:label "Aerospace" .
    OPTIONAL { ?manufacturer rdfs:label ?name }
    OPTIONAL { ?manufacturer sdk:hasCertificate ?cert }
}
"""
```

## Data Structure Overview

The knowledge graph uses these main predicates:

**Manufacturer Properties:**
- `sdk:hasProcessCapability` - Manufacturing processes (machining, welding, etc.)
- `sdk:hasMaterialCapability` - Materials they can process
- `sdk:hasCertificate` - Quality certifications
- `sdk:suppliesToIndustry` - Industries they serve
- `sdk:organizationLocatedIn` - Geographic location
- `sdk:hasOrganizationYearOfEstablishment` - When established
- `sdk:hasPrimaryNAICSClassifier` - Industry classification

**Location Structure:**
- `sdk:organizationLocatedIn` → Site
- `sdk:locatedInCity` → City
- `sdk:locatedInState` → State

## Key Insights from the Data

- **Most Common Capabilities:** Fabrication (121), Machining (103), Assembly (60)
- **Top Cities:** Charlotte (15), Greensboro (11), Raleigh (10)
- **Common Certifications:** ISO9001 (82), ISO9000 (31), AS9100 (20)
- **Top Industries Served:** Industrial Machinery (90), Metals Products (60), Construction (46)

## Dependencies

The project uses these main libraries:
- `rdflib` - RDF graph processing and SPARQL queries
- `pandas` - Data analysis and manipulation
- `networkx` - Network graph analysis
- `matplotlib` - Data visualization
- `sparqlwrapper` - Advanced SPARQL endpoint support

## Other Ways to Use TTL Files

### 1. **Apache Jena (Java)**
```bash
# Load into Jena TDB database
tdbloader --loc=./tdb sudokn-triples-NC-7-21-2024.ttl
```

### 2. **SPARQL Endpoints**
- Load into Blazegraph, Virtuoso, or GraphDB
- Query via web interface or API

### 3. **Protégé Ontology Editor**
- Open TTL file directly in Protégé
- Visual exploration and editing

### 4. **Command Line Tools**
```bash
# Convert to other formats
rapper -i turtle -o rdfxml sudokn-triples-NC-7-21-2024.ttl > output.rdf
```

### 5. **Web-based Tools**
- RDF Translator: https://rdf-translator.appspot.com/
- EasyRDF Converter: http://www.easyrdf.org/converter