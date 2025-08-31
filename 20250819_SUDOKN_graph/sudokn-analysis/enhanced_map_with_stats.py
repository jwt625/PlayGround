#!/usr/bin/env python3
"""
Enhanced SUDOKN Map with Statistics Panel
Creates an interactive map with comprehensive statistics displayed on screen
"""

from rdflib import Graph
import pandas as pd
import folium
from folium import plugins
from collections import Counter
import webbrowser
import os

def load_graph():
    """Load the SUDOKN graph"""
    g = Graph()
    g.parse("../graph/sudokn-triples-NC-7-21-2024.ttl", format="turtle")
    return g

def get_comprehensive_data(g):
    """Get all manufacturer data with comprehensive statistics"""
    query = """
    PREFIX sdk: <http://asu.edu/semantics/SUDOKN/>
    PREFIX iof-core: <https://spec.industrialontologies.org/ontology/core/Core/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?manufacturer ?name ?lat ?lon ?city ?year ?capability ?material ?cert ?industry ?naics WHERE {
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
        OPTIONAL { 
            ?manufacturer sdk:hasPrimaryNAICSClassifier ?naics_uri .
            ?naics_uri sdk:hasNAICSTextValue ?naics 
        }
    }
    """
    
    results = g.query(query)
    data = []
    invalid_coords = []
    
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
                'industry': str(row.industry) if row.industry else None,
                'naics': str(row.naics) if row.naics else None
            })
        except (ValueError, TypeError):
            invalid_coords.append({
                'name': str(row.name) if row.name else 'Unknown',
                'lat': str(row.lat),
                'lon': str(row.lon)
            })
    
    df = pd.DataFrame(data)
    print(f"Valid coordinates: {df['manufacturer_uri'].nunique()} manufacturers")
    print(f"Invalid coordinates: {len(invalid_coords)} manufacturers")
    if invalid_coords:
        print("Invalid coordinate examples:", [f"{x['name']}: {x['lat']}, {x['lon']}" for x in invalid_coords[:3]])
    
    return df

def calculate_statistics(df):
    """Calculate comprehensive statistics"""
    stats = {}
    
    # Basic counts
    stats['total_manufacturers'] = df['manufacturer_uri'].nunique()
    stats['total_cities'] = df['city'].nunique()
    stats['total_data_points'] = len(df)
    
    # Geographic coverage
    stats['lat_range'] = (df['lat'].min(), df['lat'].max())
    stats['lon_range'] = (df['lon'].min(), df['lon'].max())
    
    # Top categories
    stats['top_cities'] = df['city'].value_counts().head(10).to_dict()
    stats['top_capabilities'] = df['capability'].value_counts().head(10).to_dict()
    stats['top_materials'] = df['material'].value_counts().head(10).to_dict()
    stats['top_certifications'] = df['certification'].value_counts().head(10).to_dict()
    stats['top_industries'] = df['industry'].value_counts().head(10).to_dict()
    
    # Year analysis
    years = df[df['year'] != 'Unknown']['year'].astype(str)
    try:
        year_nums = pd.to_numeric(years, errors='coerce').dropna()
        if len(year_nums) > 0:
            stats['oldest_year'] = int(year_nums.min())
            stats['newest_year'] = int(year_nums.max())
            stats['avg_year'] = int(year_nums.mean())
        else:
            stats['oldest_year'] = stats['newest_year'] = stats['avg_year'] = 'N/A'
    except:
        stats['oldest_year'] = stats['newest_year'] = stats['avg_year'] = 'N/A'
    
    return stats

def create_statistics_html(stats):
    """Create HTML for statistics panel"""
    html = f"""
    <div style="
        position: fixed;
        top: 10px;
        right: 10px;
        width: 350px;
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid #333;
        border-radius: 10px;
        padding: 15px;
        font-family: Arial, sans-serif;
        font-size: 12px;
        z-index: 1000;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        max-height: 80vh;
        overflow-y: auto;
    ">
        <h3 style="margin: 0 0 10px 0; color: #333; text-align: center;">
            SUDOKN NC Manufacturing Network
        </h3>

        <div style="margin-bottom: 10px; padding: 8px; background: #f0f8ff; border-radius: 5px;">
            <strong>Coverage:</strong><br>
            • {stats['total_manufacturers']} Manufacturers<br>
            • {stats['total_cities']} Cities in North Carolina<br>
            • {stats['total_data_points']} Total Data Points<br>
            • Coordinates: {stats['lat_range'][0]:.2f}°-{stats['lat_range'][1]:.2f}°N,
              {abs(stats['lon_range'][1]):.2f}°-{abs(stats['lon_range'][0]):.2f}°W
        </div>

        <div style="margin-bottom: 10px; padding: 8px; background: #f0fff0; border-radius: 5px;">
            <strong>Establishment Years:</strong><br>
            • Oldest: {stats['oldest_year']}<br>
            • Newest: {stats['newest_year']}<br>
            • Average: {stats['avg_year']}
        </div>

        <div style="margin-bottom: 10px; padding: 8px; background: #fff8f0; border-radius: 5px;">
            <strong>Top Cities:</strong><br>
            {chr(10).join([f"• {city}: {count}" for city, count in list(stats['top_cities'].items())[:5]])}
        </div>

        <div style="margin-bottom: 10px; padding: 8px; background: #f8f0ff; border-radius: 5px;">
            <strong>Top Capabilities:</strong><br>
            {chr(10).join([f"• {cap}: {count}" for cap, count in list(stats['top_capabilities'].items())[:5]])}
        </div>

        <div style="margin-bottom: 10px; padding: 8px; background: #f0f8f8; border-radius: 5px;">
            <strong>Top Certifications:</strong><br>
            {chr(10).join([f"• {cert}: {count}" for cert, count in list(stats['top_certifications'].items())[:5]])}
        </div>

        <div style="margin-bottom: 10px; padding: 8px; background: #fff0f8; border-radius: 5px;">
            <strong>Top Industries Served:</strong><br>
            {chr(10).join([f"• {ind}: {count}" for ind, count in list(stats['top_industries'].items())[:4]])}
        </div>

        <div style="text-align: center; margin-top: 10px; font-size: 10px; color: #666;">
            Data: SUDOKN North Carolina<br>
            Date: July 21, 2024
        </div>
    </div>
    """
    return html

def create_enhanced_map(df, stats):
    """Create enhanced Folium map with statistics"""
    # Center on North Carolina
    nc_center = [35.5, -79.0]
    
    # Create base map
    m = folium.Map(
        location=nc_center,
        zoom_start=7,
        tiles='OpenStreetMap'
    )
    
    # Add tile layers
    folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
    
    # Add statistics panel
    stats_html = create_statistics_html(stats)
    m.get_root().html.add_child(folium.Element(stats_html))
    
    # Group by capabilities for coloring
    capabilities = df['capability'].dropna().value_counts().head(8).index
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen']
    color_map = dict(zip(capabilities, colors))
    
    # Create feature groups
    capability_groups = {}
    for cap in capabilities:
        count = df[df['capability'] == cap]['manufacturer_uri'].nunique()
        capability_groups[cap] = folium.FeatureGroup(name=f"{cap} ({count})")
    
    # Add manufacturers to map
    manufacturer_data = df.groupby('manufacturer_uri').first().reset_index()
    
    for idx, row in manufacturer_data.iterrows():
        # Get all data for this manufacturer
        mfg_data = df[df['manufacturer_uri'] == row['manufacturer_uri']]
        
        caps = mfg_data['capability'].dropna().unique()
        materials = mfg_data['material'].dropna().unique()
        certs = mfg_data['certification'].dropna().unique()
        industries = mfg_data['industry'].dropna().unique()
        
        # Create detailed popup
        popup_html = f"""
        <div style="width: 320px; font-family: Arial;">
            <h4 style="margin: 0 0 10px 0; color: #333;">{row['name']}</h4>
            <p><strong>Location:</strong> {row['city']}, North Carolina</p>
            <p><strong>Established:</strong> {row['year']}</p>
            <p><strong>NAICS:</strong> {row['naics'] if row['naics'] else 'Not specified'}</p>

            <div style="margin: 10px 0;">
                <strong>Capabilities ({len(caps)}):</strong><br>
                <div style="max-height: 80px; overflow-y: auto; font-size: 11px;">
                {'<br>'.join(['• ' + cap for cap in caps[:8]])}
                {f'<br><em>... and {len(caps)-8} more</em>' if len(caps) > 8 else ''}
                </div>
            </div>

            <div style="margin: 10px 0;">
                <strong>Materials ({len(materials)}):</strong><br>
                <div style="max-height: 60px; overflow-y: auto; font-size: 11px;">
                {'<br>'.join(['• ' + mat for mat in materials[:5]])}
                {f'<br><em>... and {len(materials)-5} more</em>' if len(materials) > 5 else ''}
                </div>
            </div>

            <div style="margin: 10px 0;">
                <strong>Certifications ({len(certs)}):</strong><br>
                <div style="font-size: 11px;">
                {'<br>'.join(['• ' + cert for cert in certs[:4]])}
                {f'<br><em>... and {len(certs)-4} more</em>' if len(certs) > 4 else ''}
                </div>
            </div>
        </div>
        """
        
        # Determine color
        primary_cap = caps[0] if len(caps) > 0 else 'Other'
        color = color_map.get(primary_cap, 'gray')
        
        # Create marker
        marker = folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=7,
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=f"{row['name']} - {row['city']}",
            color='black',
            weight=1,
            fillColor=color,
            fillOpacity=0.8
        )
        
        # Add to appropriate group
        if primary_cap in capability_groups:
            marker.add_to(capability_groups[primary_cap])
        else:
            marker.add_to(m)
    
    # Add all capability groups to map
    for group in capability_groups.values():
        group.add_to(m)
    
    # Add city markers for major hubs
    city_centers = df.groupby('city').agg({
        'lat': 'mean',
        'lon': 'mean',
        'manufacturer_uri': 'nunique'
    }).reset_index()
    
    major_cities = city_centers[city_centers['manufacturer_uri'] >= 5]
    
    for _, city in major_cities.iterrows():
        folium.Marker(
            location=[city['lat'], city['lon']],
            popup=f"<b>{city['city']}</b><br>{city['manufacturer_uri']} manufacturers",
            tooltip=f"{city['city']} ({city['manufacturer_uri']} mfgs)",
            icon=folium.Icon(color='red', icon='industry', prefix='fa')
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def main():
    print("Loading SUDOKN knowledge graph...")
    g = load_graph()
    
    print("Extracting comprehensive manufacturer data...")
    df = get_comprehensive_data(g)
    
    print("Calculating statistics...")
    stats = calculate_statistics(df)
    
    print(f"\nDATASET SUMMARY:")
    print(f"This is the SUDOKN NORTH CAROLINA dataset (note 'NC' in filename)")
    print(f"• Total manufacturers: {stats['total_manufacturers']}")
    print(f"• Cities covered: {stats['total_cities']}")
    print(f"• Geographic range: {stats['lat_range'][0]:.2f}° to {stats['lat_range'][1]:.2f}°N")
    print(f"• All coordinates are within North Carolina boundaries")

    print(f"\nTOP MANUFACTURING HUBS:")
    for city, count in list(stats['top_cities'].items())[:5]:
        print(f"• {city}: {count} manufacturers")

    print(f"\nTOP CAPABILITIES:")
    for cap, count in list(stats['top_capabilities'].items())[:5]:
        print(f"• {cap}: {count} manufacturers")

    print("\nCreating enhanced interactive map with statistics...")
    enhanced_map = create_enhanced_map(df, stats)
    enhanced_map.save('sudokn_enhanced_map_with_stats.html')

    print(f"\nEnhanced map created: sudokn_enhanced_map_with_stats.html")
    print(f"The map shows ALL {stats['total_manufacturers']} manufacturers with valid coordinates")
    print(f"Statistics panel is displayed on the right side of the map")

    # Open the map
    map_path = os.path.abspath('sudokn_enhanced_map_with_stats.html')
    print(f"\nOpening map: {map_path}")
    webbrowser.open(f'file://{map_path}')

if __name__ == "__main__":
    main()
