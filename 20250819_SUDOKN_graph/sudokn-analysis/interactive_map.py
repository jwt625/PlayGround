#!/usr/bin/env python3
"""
Interactive SUDOKN Map Visualization
Creates interactive maps with real geographic data using Folium
"""

from rdflib import Graph
import pandas as pd
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import webbrowser
import os

def load_graph():
    """Load the SUDOKN graph"""
    g = Graph()
    g.parse("../graph/sudokn-triples-NC-7-21-2024.ttl", format="turtle")
    return g

def get_manufacturer_data(g):
    """Get comprehensive manufacturer data with coordinates"""
    query = """
    PREFIX sdk: <http://asu.edu/semantics/SUDOKN/>
    PREFIX iof-core: <https://spec.industrialontologies.org/ontology/core/Core/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?manufacturer ?name ?lat ?lon ?city ?year ?capability ?material ?cert ?industry WHERE {
        ?manufacturer a iof-core:Manufacturer .
        ?manufacturer sdk:organizationLocatedIn ?site .
        ?site sdk:hasSpatialCoordinates ?coordinates .
        ?coordinates sdk:hasLatitudeValue ?lat .
        ?coordinates sdk:hasLongitudeValue ?lon .
        ?site sdk:locatedInCity ?city_uri .
        ?city_uri rdfs:label ?city .
        
        OPTIONAL { ?manufacturer rdfs:label ?name }
        OPTIONAL { ?manufacturer sdk:hasOrganizationYearOfEstablishment ?year }
        OPTIONAL { 
            ?manufacturer sdk:hasProcessCapability ?cap_uri .
            ?cap_uri rdfs:label ?capability 
        }
        OPTIONAL { 
            ?manufacturer sdk:hasMaterialCapability ?mat_uri .
            ?mat_uri rdfs:label ?material 
        }
        OPTIONAL { 
            ?manufacturer sdk:hasCertificate ?cert_uri .
            ?cert_uri rdfs:label ?cert 
        }
        OPTIONAL { 
            ?manufacturer sdk:suppliesToIndustry ?ind_uri .
            ?ind_uri rdfs:label ?industry 
        }
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
                'city': str(row.city),
                'year': str(row.year) if row.year else 'Unknown',
                'capability': str(row.capability) if row.capability else None,
                'material': str(row.material) if row.material else None,
                'certification': str(row.cert) if row.cert else None,
                'industry': str(row.industry) if row.industry else None
            })
        except (ValueError, TypeError):
            continue
    
    return pd.DataFrame(data)

def create_folium_map(df):
    """Create an interactive Folium map"""
    # Center on North Carolina
    nc_center = [35.5, -79.0]
    
    # Create base map
    m = folium.Map(
        location=nc_center,
        zoom_start=7,
        tiles='OpenStreetMap'
    )
    
    # Add different tile layers
    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png',
        attr='Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors',
        name='Stamen Terrain'
    ).add_to(m)
    folium.TileLayer(
        tiles='CartoDB positron',
        name='CartoDB Positron'
    ).add_to(m)
    
    # Group manufacturers by capabilities for different colors
    capabilities = df['capability'].dropna().value_counts().head(10).index
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
              'lightred', 'beige', 'darkblue', 'darkgreen']
    color_map = dict(zip(capabilities, colors))
    
    # Create feature groups for each capability
    capability_groups = {}
    for cap in capabilities:
        capability_groups[cap] = folium.FeatureGroup(name=f"{cap} ({df[df['capability']==cap].shape[0]})")
    
    # Add manufacturers to map
    for idx, row in df.iterrows():
        # Aggregate data for this manufacturer
        mfg_data = df[df['manufacturer_uri'] == row['manufacturer_uri']]
        
        # Get all capabilities, materials, etc. for this manufacturer
        caps = mfg_data['capability'].dropna().unique()
        materials = mfg_data['material'].dropna().unique()
        certs = mfg_data['certification'].dropna().unique()
        industries = mfg_data['industry'].dropna().unique()
        
        # Create popup content
        popup_html = f"""
        <div style="width: 300px;">
            <h4>{row['name']}</h4>
            <p><strong>Location:</strong> {row['city']}</p>
            <p><strong>Established:</strong> {row['year']}</p>
            
            <p><strong>Capabilities ({len(caps)}):</strong><br>
            {'<br>'.join(['• ' + cap for cap in caps[:5]])}
            {f'<br>... and {len(caps)-5} more' if len(caps) > 5 else ''}</p>
            
            <p><strong>Materials ({len(materials)}):</strong><br>
            {'<br>'.join(['• ' + mat for mat in materials[:3]])}
            {f'<br>... and {len(materials)-3} more' if len(materials) > 3 else ''}</p>
            
            <p><strong>Certifications ({len(certs)}):</strong><br>
            {'<br>'.join(['• ' + cert for cert in certs[:3]])}
            {f'<br>... and {len(certs)-3} more' if len(certs) > 3 else ''}</p>
            
            <p><strong>Industries Served ({len(industries)}):</strong><br>
            {'<br>'.join(['• ' + ind for ind in industries[:3]])}
            {f'<br>... and {len(industries)-3} more' if len(industries) > 3 else ''}</p>
        </div>
        """
        
        # Determine marker color based on primary capability
        primary_cap = caps[0] if len(caps) > 0 else 'Other'
        color = color_map.get(primary_cap, 'gray')
        
        # Create marker
        marker = folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=8,
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=f"{row['name']} - {row['city']}",
            color='black',
            weight=1,
            fillColor=color,
            fillOpacity=0.7
        )
        
        # Add to appropriate capability group
        if primary_cap in capability_groups:
            marker.add_to(capability_groups[primary_cap])
        else:
            marker.add_to(m)
    
    # Add capability groups to map
    for group in capability_groups.values():
        group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add marker cluster plugin
    marker_cluster = plugins.MarkerCluster().add_to(m)
    
    # Add all manufacturers to cluster as well
    for idx, row in df.drop_duplicates('manufacturer_uri').iterrows():
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=f"{row['name']} - {row['city']}",
            tooltip=row['name']
        ).add_to(marker_cluster)
    
    return m

def create_plotly_map(df):
    """Create an interactive Plotly map"""
    # Aggregate data by manufacturer
    mfg_summary = df.groupby(['manufacturer_uri', 'name', 'lat', 'lon', 'city', 'year']).agg({
        'capability': lambda x: ', '.join(x.dropna().unique()[:3]),
        'material': lambda x: ', '.join(x.dropna().unique()[:3]),
        'certification': lambda x: ', '.join(x.dropna().unique()[:3]),
        'industry': lambda x: ', '.join(x.dropna().unique()[:3])
    }).reset_index()
    
    # Get primary capability for coloring
    primary_caps = df.groupby('manufacturer_uri')['capability'].first().to_dict()
    mfg_summary['primary_capability'] = mfg_summary['manufacturer_uri'].map(primary_caps)
    
    # Create hover text
    mfg_summary['hover_text'] = (
        '<b>' + mfg_summary['name'] + '</b><br>' +
        'City: ' + mfg_summary['city'] + '<br>' +
        'Established: ' + mfg_summary['year'] + '<br>' +
        'Capabilities: ' + mfg_summary['capability'] + '<br>' +
        'Materials: ' + mfg_summary['material'] + '<br>' +
        'Certifications: ' + mfg_summary['certification']
    )
    
    # Create map
    fig = px.scatter_mapbox(
        mfg_summary,
        lat='lat',
        lon='lon',
        color='primary_capability',
        size_max=15,
        zoom=6,
        hover_name='name',
        hover_data={
            'city': True,
            'year': True,
            'lat': False,
            'lon': False,
            'primary_capability': False
        },
        title='SUDOKN Manufacturing Network - North Carolina',
        mapbox_style='open-street-map',
        height=700
    )
    
    # Update layout
    fig.update_layout(
        mapbox=dict(
            center=dict(lat=35.5, lon=-79.0),
            zoom=6
        ),
        margin=dict(r=0, t=50, l=0, b=0)
    )
    
    return fig

def create_city_analysis(df):
    """Create city-level analysis"""
    city_stats = df.groupby('city').agg({
        'manufacturer_uri': 'nunique',
        'capability': lambda x: len(x.dropna().unique()),
        'lat': 'first',
        'lon': 'first'
    }).reset_index()
    
    city_stats.columns = ['city', 'num_manufacturers', 'num_capabilities', 'lat', 'lon']
    city_stats = city_stats.sort_values('num_manufacturers', ascending=False)
    
    # Create bubble map
    fig = px.scatter_mapbox(
        city_stats.head(20),
        lat='lat',
        lon='lon',
        size='num_manufacturers',
        color='num_capabilities',
        hover_name='city',
        hover_data={
            'num_manufacturers': True,
            'num_capabilities': True,
            'lat': False,
            'lon': False
        },
        title='Manufacturing Hubs - Top 20 Cities by Number of Manufacturers',
        mapbox_style='open-street-map',
        size_max=30,
        height=600
    )
    
    fig.update_layout(
        mapbox=dict(
            center=dict(lat=35.5, lon=-79.0),
            zoom=6
        ),
        margin=dict(r=0, t=50, l=0, b=0)
    )
    
    return fig, city_stats

def main():
    print("Loading SUDOKN knowledge graph...")
    g = load_graph()
    
    print("Extracting manufacturer data with coordinates...")
    df = get_manufacturer_data(g)
    print(f"Found {df['manufacturer_uri'].nunique()} unique manufacturers with coordinates")
    print(f"Total data points: {len(df)}")
    
    print("\n1. Creating Folium interactive map...")
    folium_map = create_folium_map(df)
    folium_map.save('sudokn_interactive_map.html')
    print("Saved: sudokn_interactive_map.html")
    
    print("2. Creating Plotly manufacturer map...")
    plotly_fig = create_plotly_map(df)
    plotly_fig.write_html('sudokn_plotly_map.html')
    print("Saved: sudokn_plotly_map.html")
    
    print("3. Creating city analysis...")
    city_fig, city_stats = create_city_analysis(df)
    city_fig.write_html('sudokn_city_analysis.html')
    print("Saved: sudokn_city_analysis.html")
    
    print("\nTop 10 manufacturing cities:")
    print(city_stats.head(10)[['city', 'num_manufacturers', 'num_capabilities']].to_string(index=False))
    
    print(f"\nInteractive maps created! Open the HTML files in your browser:")
    print(f"- sudokn_interactive_map.html (Folium - most detailed)")
    print(f"- sudokn_plotly_map.html (Plotly - manufacturer view)")
    print(f"- sudokn_city_analysis.html (Plotly - city analysis)")
    
    # Automatically open the main map
    map_path = os.path.abspath('sudokn_interactive_map.html')
    print(f"\nOpening main map: {map_path}")
    webbrowser.open(f'file://{map_path}')

if __name__ == "__main__":
    main()
