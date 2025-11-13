#!/usr/bin/env python3
"""
Tab Tree History Analysis Script

Analyzes tab-tree JSON files to extract insights about browsing patterns,
including domain distribution, temporal patterns, and tab behavior.
"""

import json
import os
from pathlib import Path
from urllib.parse import urlparse
from collections import Counter, defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Configuration
DATA_DIR = Path("./data")
PLOT_DIR = Path("./plots")
PLOT_DIR.mkdir(exist_ok=True)

# Plot Configuration - Global settings for easy adjustment
PLOT_CONFIG = {
    # Figure sizes (width, height) - closer to 1:1 aspect ratio for mobile
    'fig_size_square': (10, 10),      # For pie charts, heatmaps
    'fig_size_standard': (12, 10),    # For most plots
    'fig_size_wide': (14, 12),        # For time series with many points

    # Font sizes
    'title_size': 20,
    'label_size': 20,
    'tick_size': 18,
    'legend_size': 18,
    'annotation_size': 18,

    # Plotly settings
    'plotly_height': 800,             # Height for interactive plots
    'plotly_font_size': 24,
    'plotly_title_size': 22,

    # Other settings
    'dpi': 300,
    'line_width': 3,
    'marker_size': 10,
}

# Color scheme
COLORS = px.colors.qualitative.Set3

# Set matplotlib default font sizes
plt.rcParams.update({
    'font.size': PLOT_CONFIG['tick_size'],
    'axes.titlesize': PLOT_CONFIG['title_size'],
    'axes.labelsize': PLOT_CONFIG['label_size'],
    'xtick.labelsize': PLOT_CONFIG['tick_size'],
    'ytick.labelsize': PLOT_CONFIG['tick_size'],
    'legend.fontsize': PLOT_CONFIG['legend_size'],
})


def extract_date_from_filename(filename):
    """Extract date from filename like 'tab-tree-2024-11-15T06-57-33-715Z.json'."""
    import re
    match = re.search(r'tab-tree-(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3}Z)', filename)
    if match:
        date_str = match.group(1)
        # Convert format: 2024-11-15T06-57-33-715Z -> 2024-11-15T06:57:33.715Z
        date_str = date_str.replace('-', ':', 2)  # Replace first two hyphens after T
        date_str = date_str.rsplit('-', 1)[0] + '.' + date_str.rsplit('-', 1)[1]
        return date_str
    return None


def load_all_tab_trees():
    """Load all tab-tree JSON files from the data directory."""
    all_data = []
    json_files = sorted(DATA_DIR.glob("tab-tree-*.json"))

    print(f"Found {len(json_files)} tab-tree files")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # Check if this is the new format (with metadata) or old format (without)
                if 'metadata' in data and 'tabTree' in data:
                    # New format
                    metadata = data['metadata']
                    tab_tree = data['tabTree']
                else:
                    # Old format - the entire JSON is the tabTree
                    tab_tree = data
                    metadata = {}
                    # Extract date from filename
                    date_from_filename = extract_date_from_filename(json_file.name)
                    if date_from_filename:
                        metadata['exportDate'] = date_from_filename
                    # Count nodes
                    node_count = 0
                    for root_node in tab_tree.values():
                        nodes = []
                        flatten_tree(root_node, nodes)
                        node_count += len(nodes)
                    metadata['nodeCount'] = node_count

                all_data.append({
                    'filename': json_file.name,
                    'metadata': metadata,
                    'tabTree': tab_tree
                })
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return all_data


def extract_domain(url):
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        # Remove www. prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain if domain else 'unknown'
    except:
        return 'unknown'


def flatten_tree(node, all_nodes):
    """Recursively flatten the tree structure."""
    all_nodes.append(node)
    for child in node.get('children', []):
        flatten_tree(child, all_nodes)


def collect_all_tabs(all_data):
    """Collect all tabs from all files."""
    all_tabs = []

    for file_data in all_data:
        export_date = file_data['metadata'].get('exportDate')
        tab_tree = file_data['tabTree']

        for root_id, root_node in tab_tree.items():
            nodes = []
            flatten_tree(root_node, nodes)

            for node in nodes:
                tab_info = {
                    'url': node.get('url', ''),
                    'title': node.get('title', ''),
                    'domain': extract_domain(node.get('url', '')),
                    'createdAt': node.get('createdAt'),
                    'closedAt': node.get('closedAt'),
                    'export_date': export_date,
                    'filename': file_data['filename']
                }
                all_tabs.append(tab_info)

    return all_tabs


def analyze_domain_distribution(all_tabs, top_n=15):
    """Analyze and visualize domain distribution."""
    print(f"\n{'='*60}")
    print("DOMAIN DISTRIBUTION ANALYSIS")
    print(f"{'='*60}")

    domains = [tab['domain'] for tab in all_tabs if tab['domain'] != 'unknown']
    domain_counts = Counter(domains)

    print(f"Total unique domains: {len(domain_counts)}")
    print(f"Total tabs analyzed: {len(domains)}")
    print(f"\nTop {top_n} domains:")
    for domain, count in domain_counts.most_common(top_n):
        percentage = (count / len(domains)) * 100
        print(f"  {domain:40s}: {count:6d} ({percentage:5.2f}%)")

    # Prepare data for plotting
    top_domains = domain_counts.most_common(top_n)
    other_count = sum(count for domain, count in domain_counts.items()
                     if domain not in dict(top_domains))

    labels = [domain for domain, _ in top_domains]
    values = [count for _, count in top_domains]

    if other_count > 0:
        labels.append('Others')
        values.append(other_count)

    # Matplotlib pie chart
    plt.figure(figsize=PLOT_CONFIG['fig_size_square'])
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': PLOT_CONFIG['annotation_size']})
    plt.title(f'Top {top_n} Domain Distribution', fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'domain_distribution_pie.png', dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {PLOT_DIR / 'domain_distribution_pie.png'}")

    # Plotly interactive pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3,
                                 textfont=dict(size=PLOT_CONFIG['plotly_font_size']))])
    fig.update_layout(
        title=dict(text=f'Top {top_n} Domain Distribution (Interactive)',
                  font=dict(size=PLOT_CONFIG['plotly_title_size'])),
        font=dict(size=PLOT_CONFIG['plotly_font_size']),
        height=PLOT_CONFIG['plotly_height']
    )
    fig.write_html(PLOT_DIR / 'domain_distribution_pie.html')
    print(f"✓ Saved: {PLOT_DIR / 'domain_distribution_pie.html'}")

    # Bar chart
    plt.figure(figsize=PLOT_CONFIG['fig_size_standard'])
    plt.barh(range(len(labels)), values, color=plt.cm.Set3(range(len(labels))))
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('Number of Tabs')
    plt.title(f'Top {top_n} Domains by Tab Count', fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'domain_distribution_bar.png', dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {PLOT_DIR / 'domain_distribution_bar.png'}")

    # Plotly interactive bar chart
    fig = go.Figure(data=[go.Bar(x=values, y=labels, orientation='h',
                                  marker=dict(color=values, colorscale='Viridis'))])
    fig.update_layout(
        title=dict(text=f'Top {top_n} Domains by Tab Count (Interactive)',
                  font=dict(size=PLOT_CONFIG['plotly_title_size'])),
        xaxis_title='Number of Tabs',
        yaxis_title='Domain',
        font=dict(size=PLOT_CONFIG['plotly_font_size']),
        height=PLOT_CONFIG['plotly_height'],
        yaxis={'categoryorder': 'total ascending'}
    )
    fig.write_html(PLOT_DIR / 'domain_distribution_bar.html')
    print(f"✓ Saved: {PLOT_DIR / 'domain_distribution_bar.html'}")

    return domain_counts


def analyze_temporal_patterns(all_data, all_tabs):
    """Analyze temporal patterns based on individual tab timestamps."""
    print(f"\n{'='*60}")
    print("TEMPORAL PATTERNS ANALYSIS")
    print(f"{'='*60}")

    # Analyze export dates
    export_dates = []
    node_counts = []

    for file_data in all_data:
        export_date = file_data['metadata'].get('exportDate')
        node_count = file_data['metadata'].get('nodeCount', 0)

        if export_date:
            try:
                dt = datetime.fromisoformat(export_date.replace('Z', '+00:00'))
                export_dates.append(dt)
                node_counts.append(node_count)
            except:
                pass

    if export_dates:
        df_exports = pd.DataFrame({
            'date': export_dates,
            'node_count': node_counts
        }).sort_values('date')

        print(f"Export date range: {df_exports['date'].min()} to {df_exports['date'].max()}")
        print(f"Average tabs per export: {df_exports['node_count'].mean():.1f}")

    # Analyze individual tab creation times
    tab_created_dates = []
    tab_closed_dates = []

    for tab in all_tabs:
        if tab['createdAt']:
            try:
                dt = datetime.fromtimestamp(tab['createdAt'] / 1000)  # Convert ms to seconds
                tab_created_dates.append(dt)
            except:
                pass

        if tab['closedAt']:
            try:
                dt = datetime.fromtimestamp(tab['closedAt'] / 1000)
                tab_closed_dates.append(dt)
            except:
                pass

    print(f"\nTab creation date range: {min(tab_created_dates)} to {max(tab_created_dates)}")
    print(f"Total tabs with creation timestamps: {len(tab_created_dates):,}")
    print(f"Total tabs with closure timestamps: {len(tab_closed_dates):,}")

    # Daily tab creation counts
    df_created = pd.DataFrame({'date': tab_created_dates})
    df_created['date'] = pd.to_datetime(df_created['date']).dt.date
    daily_created = df_created.groupby('date').size().reset_index(name='count')
    daily_created['date'] = pd.to_datetime(daily_created['date'])
    daily_created = daily_created.sort_values('date')

    # Matplotlib time series - Tab creation over time
    plt.figure(figsize=PLOT_CONFIG['fig_size_wide'])
    plt.plot(daily_created['date'], daily_created['count'],
            linewidth=PLOT_CONFIG['line_width'], alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('Tabs Created')
    plt.title('Daily Tab Creation Over Time', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'tab_creation_timeline.png', dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {PLOT_DIR / 'tab_creation_timeline.png'}")

    # Plotly interactive time series
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_created['date'], y=daily_created['count'],
                            mode='lines',
                            name='Tabs Created',
                            line=dict(width=PLOT_CONFIG['line_width']),
                            fill='tozeroy'))
    fig.update_layout(
        title=dict(text='Daily Tab Creation Over Time (Interactive)',
                  font=dict(size=PLOT_CONFIG['plotly_title_size'])),
        xaxis_title='Date',
        yaxis_title='Number of Tabs Created',
        font=dict(size=PLOT_CONFIG['plotly_font_size']),
        hovermode='x unified',
        height=PLOT_CONFIG['plotly_height']
    )
    fig.write_html(PLOT_DIR / 'tab_creation_timeline.html')
    print(f"✓ Saved: {PLOT_DIR / 'tab_creation_timeline.html'}")

    # Monthly aggregation of tab creation
    df_created_monthly = df_created.copy()
    df_created_monthly['month'] = pd.to_datetime(df_created_monthly['date']).dt.to_period('M')
    monthly_created = df_created_monthly.groupby('month').size().reset_index(name='count')
    monthly_created['month'] = monthly_created['month'].dt.to_timestamp()

    plt.figure(figsize=PLOT_CONFIG['fig_size_standard'])
    plt.bar(monthly_created['month'], monthly_created['count'], width=20, color='steelblue', alpha=0.7)
    plt.xlabel('Month')
    plt.ylabel('Tabs Created')
    plt.title('Monthly Tab Creation', fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'monthly_tab_creation.png', dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {PLOT_DIR / 'monthly_tab_creation.png'}")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly_created['month'], y=monthly_created['count'],
                        name='Tabs Created',
                        marker_color='steelblue'))
    fig.update_layout(
        title=dict(text='Monthly Tab Creation (Interactive)',
                  font=dict(size=PLOT_CONFIG['plotly_title_size'])),
        xaxis_title='Month',
        yaxis_title='Number of Tabs Created',
        font=dict(size=PLOT_CONFIG['plotly_font_size']),
        height=PLOT_CONFIG['plotly_height']
    )
    fig.write_html(PLOT_DIR / 'monthly_tab_creation.html')
    print(f"✓ Saved: {PLOT_DIR / 'monthly_tab_creation.html'}")

    # Hour of day analysis
    df_hourly = pd.DataFrame({'datetime': tab_created_dates})
    df_hourly['hour'] = df_hourly['datetime'].dt.hour
    hourly_counts = df_hourly.groupby('hour').size().reset_index(name='count')

    plt.figure(figsize=PLOT_CONFIG['fig_size_standard'])
    plt.bar(hourly_counts['hour'], hourly_counts['count'], color='coral', alpha=0.7)
    plt.xlabel('Hour of Day')
    plt.ylabel('Tabs Created')
    plt.title('Tab Creation by Hour of Day', fontweight='bold')
    plt.xticks(range(24))
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'hourly_tab_creation.png', dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {PLOT_DIR / 'hourly_tab_creation.png'}")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=hourly_counts['hour'], y=hourly_counts['count'],
                        marker_color='coral'))
    fig.update_layout(
        title=dict(text='Tab Creation by Hour of Day (Interactive)',
                  font=dict(size=PLOT_CONFIG['plotly_title_size'])),
        xaxis_title='Hour of Day (0-23)',
        yaxis_title='Number of Tabs Created',
        font=dict(size=PLOT_CONFIG['plotly_font_size']),
        height=PLOT_CONFIG['plotly_height'],
        xaxis=dict(tickmode='linear', tick0=0, dtick=1)
    )
    fig.write_html(PLOT_DIR / 'hourly_tab_creation.html')
    print(f"✓ Saved: {PLOT_DIR / 'hourly_tab_creation.html'}")

    # Day of week analysis
    df_dow = pd.DataFrame({'datetime': tab_created_dates})
    df_dow['day_of_week'] = df_dow['datetime'].dt.day_name()
    df_dow['day_num'] = df_dow['datetime'].dt.dayofweek
    dow_counts = df_dow.groupby(['day_num', 'day_of_week']).size().reset_index(name='count')
    dow_counts = dow_counts.sort_values('day_num')

    plt.figure(figsize=PLOT_CONFIG['fig_size_standard'])
    plt.bar(dow_counts['day_of_week'], dow_counts['count'], color='mediumseagreen', alpha=0.7)
    plt.xlabel('Day of Week')
    plt.ylabel('Tabs Created')
    plt.title('Tab Creation by Day of Week', fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'daily_tab_creation.png', dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {PLOT_DIR / 'daily_tab_creation.png'}")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=dow_counts['day_of_week'], y=dow_counts['count'],
                        marker_color='mediumseagreen'))
    fig.update_layout(
        title=dict(text='Tab Creation by Day of Week (Interactive)',
                  font=dict(size=PLOT_CONFIG['plotly_title_size'])),
        xaxis_title='Day of Week',
        yaxis_title='Number of Tabs Created',
        font=dict(size=PLOT_CONFIG['plotly_font_size']),
        height=PLOT_CONFIG['plotly_height']
    )
    fig.write_html(PLOT_DIR / 'daily_tab_creation.html')
    print(f"✓ Saved: {PLOT_DIR / 'daily_tab_creation.html'}")

    # Heatmap: Day of week vs Hour of day
    df_heatmap = pd.DataFrame({'datetime': tab_created_dates})
    df_heatmap['hour'] = df_heatmap['datetime'].dt.hour
    df_heatmap['day_of_week'] = df_heatmap['datetime'].dt.day_name()
    df_heatmap['day_num'] = df_heatmap['datetime'].dt.dayofweek

    # Create pivot table
    heatmap_data = df_heatmap.groupby(['day_num', 'day_of_week', 'hour']).size().reset_index(name='count')
    pivot = heatmap_data.pivot_table(values='count', index='day_of_week', columns='hour', fill_value=0)

    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])

    # Matplotlib heatmap
    plt.figure(figsize=PLOT_CONFIG['fig_size_wide'])
    plt.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
    cbar = plt.colorbar(label='Number of Tabs Created')
    cbar.ax.tick_params(labelsize=PLOT_CONFIG['tick_size'])
    plt.xticks(range(24), range(24))
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.title('Tab Creation Heatmap: Day of Week vs Hour of Day', fontweight='bold')

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(24):
            plt.text(j, i, int(pivot.values[i, j]),
                    ha="center", va="center", color="black",
                    fontsize=PLOT_CONFIG['annotation_size'])

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'tab_creation_heatmap.png', dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {PLOT_DIR / 'tab_creation_heatmap.png'}")

    # Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=list(range(24)),
        y=pivot.index,
        colorscale='YlOrRd',
        text=pivot.values,
        texttemplate='%{text}',
        textfont={"size": PLOT_CONFIG['annotation_size']},
        colorbar=dict(
            title=dict(text="Tabs Created", font=dict(size=PLOT_CONFIG['plotly_font_size'])),
            tickfont=dict(size=PLOT_CONFIG['tick_size'])
        )
    ))
    fig.update_layout(
        title=dict(text='Tab Creation Heatmap: Day of Week vs Hour of Day (Interactive)',
                  font=dict(size=PLOT_CONFIG['plotly_title_size'])),
        xaxis_title='Hour of Day (0-23)',
        yaxis_title='Day of Week',
        font=dict(size=PLOT_CONFIG['plotly_font_size']),
        height=PLOT_CONFIG['plotly_height'],
        xaxis=dict(tickmode='linear', tick0=0, dtick=1)
    )
    fig.write_html(PLOT_DIR / 'tab_creation_heatmap.html')
    print(f"✓ Saved: {PLOT_DIR / 'tab_creation_heatmap.html'}")


def analyze_tab_lifecycle(all_tabs):
    """Analyze tab lifecycle (open vs closed tabs) and lifespan."""
    print(f"\n{'='*60}")
    print("TAB LIFECYCLE ANALYSIS")
    print(f"{'='*60}")

    open_tabs = sum(1 for tab in all_tabs if tab['closedAt'] is None)
    closed_tabs = sum(1 for tab in all_tabs if tab['closedAt'] is not None)

    print(f"Open tabs: {open_tabs:,} ({open_tabs/(open_tabs+closed_tabs)*100:.1f}%)")
    print(f"Closed tabs: {closed_tabs:,} ({closed_tabs/(open_tabs+closed_tabs)*100:.1f}%)")

    # Calculate tab lifespan for closed tabs
    lifespans = []
    for tab in all_tabs:
        if tab['createdAt'] and tab['closedAt']:
            try:
                lifespan_ms = tab['closedAt'] - tab['createdAt']
                lifespan_hours = lifespan_ms / (1000 * 60 * 60)
                if lifespan_hours > 0:  # Filter out negative or zero lifespans
                    lifespans.append(lifespan_hours)
            except:
                pass

    if lifespans:
        print(f"\nTab Lifespan Statistics (for {len(lifespans):,} closed tabs):")
        print(f"  Average lifespan: {np.mean(lifespans):.1f} hours ({np.mean(lifespans)/24:.1f} days)")
        print(f"  Median lifespan: {np.median(lifespans):.1f} hours ({np.median(lifespans)/24:.1f} days)")
        print(f"  Max lifespan: {np.max(lifespans):.1f} hours ({np.max(lifespans)/24:.1f} days)")
        print(f"  Min lifespan: {np.min(lifespans):.1f} hours")

    # Pie chart - Open vs Closed
    labels = ['Open Tabs', 'Closed Tabs']
    values = [open_tabs, closed_tabs]
    colors_custom = ['#66c2a5', '#fc8d62']

    plt.figure(figsize=PLOT_CONFIG['fig_size_square'])
    plt.pie(values, labels=labels, autopct='%1.1f%%', colors=colors_custom, startangle=90,
            textprops={'fontsize': PLOT_CONFIG['annotation_size']})
    plt.title('Tab Lifecycle: Open vs Closed', fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'tab_lifecycle.png', dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {PLOT_DIR / 'tab_lifecycle.png'}")

    # Plotly
    fig = go.Figure(data=[go.Pie(labels=labels, values=values,
                                 marker=dict(colors=colors_custom), hole=0.3,
                                 textfont=dict(size=PLOT_CONFIG['plotly_font_size']))])
    fig.update_layout(
        title=dict(text='Tab Lifecycle: Open vs Closed (Interactive)',
                  font=dict(size=PLOT_CONFIG['plotly_title_size'])),
        font=dict(size=PLOT_CONFIG['plotly_font_size']),
        height=PLOT_CONFIG['plotly_height']
    )
    fig.write_html(PLOT_DIR / 'tab_lifecycle.html')
    print(f"✓ Saved: {PLOT_DIR / 'tab_lifecycle.html'}")

    # Lifespan distribution
    if lifespans:
        # Convert to days for better readability
        lifespans_days = [h / 24 for h in lifespans]

        # Histogram
        plt.figure(figsize=PLOT_CONFIG['fig_size_standard'])
        plt.hist(lifespans_days, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Tab Lifespan (days)')
        plt.ylabel('Number of Tabs')
        plt.title('Distribution of Tab Lifespan', fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(PLOT_DIR / 'tab_lifespan_distribution.png', dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {PLOT_DIR / 'tab_lifespan_distribution.png'}")

        # Plotly histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=lifespans_days, nbinsx=50,
                                   marker_color='skyblue',
                                   name='Tab Lifespan'))
        fig.update_layout(
            title=dict(text='Distribution of Tab Lifespan (Interactive)',
                      font=dict(size=PLOT_CONFIG['plotly_title_size'])),
            xaxis_title='Tab Lifespan (days)',
            yaxis_title='Number of Tabs',
            font=dict(size=PLOT_CONFIG['plotly_font_size']),
            height=PLOT_CONFIG['plotly_height']
        )
        fig.write_html(PLOT_DIR / 'tab_lifespan_distribution.html')
        print(f"✓ Saved: {PLOT_DIR / 'tab_lifespan_distribution.html'}")

        # Box plot for lifespan by log scale
        fig = go.Figure()
        fig.add_trace(go.Box(y=lifespans_days, name='Tab Lifespan',
                            marker_color='lightblue'))
        fig.update_layout(
            title=dict(text='Tab Lifespan Distribution (Box Plot)',
                      font=dict(size=PLOT_CONFIG['plotly_title_size'])),
            yaxis_title='Tab Lifespan (days)',
            yaxis_type='log',
            font=dict(size=PLOT_CONFIG['plotly_font_size']),
            height=PLOT_CONFIG['plotly_height']
        )
        fig.write_html(PLOT_DIR / 'tab_lifespan_boxplot.html')
        print(f"✓ Saved: {PLOT_DIR / 'tab_lifespan_boxplot.html'}")


def analyze_domain_trends_over_time(all_tabs, top_n=10):
    """Analyze how domain usage changes over time."""
    print(f"\n{'='*60}")
    print("DOMAIN TRENDS OVER TIME")
    print(f"{'='*60}")

    # Get top domains overall
    domains = [tab['domain'] for tab in all_tabs if tab['domain'] != 'unknown']
    top_domains = [domain for domain, _ in Counter(domains).most_common(top_n)]

    # Group by export date and domain
    domain_by_file = defaultdict(lambda: defaultdict(int))
    file_dates = {}

    for tab in all_tabs:
        if tab['domain'] in top_domains and tab['export_date']:
            try:
                dt = datetime.fromisoformat(tab['export_date'].replace('Z', '+00:00'))
                file_dates[tab['filename']] = dt
                domain_by_file[tab['filename']][tab['domain']] += 1
            except:
                pass

    # Plotly stacked area chart
    fig = go.Figure()

    for domain in top_domains:
        counts = []
        for filename in sorted(file_dates.keys(), key=lambda x: file_dates[x]):
            counts.append(domain_by_file[filename].get(domain, 0))

        fig.add_trace(go.Scatter(
            x=[file_dates[f] for f in sorted(file_dates.keys(), key=lambda x: file_dates[x])],
            y=counts,
            mode='lines',
            name=domain,
            stackgroup='one',
            hovertemplate='%{y} tabs<extra></extra>'
        ))

    fig.update_layout(
        title=dict(text=f'Top {top_n} Domain Usage Over Time (Stacked)',
                  font=dict(size=PLOT_CONFIG['plotly_title_size'])),
        xaxis_title='Date',
        yaxis_title='Number of Tabs',
        font=dict(size=PLOT_CONFIG['plotly_font_size']),
        hovermode='x unified',
        height=PLOT_CONFIG['plotly_height']
    )
    fig.write_html(PLOT_DIR / 'domain_trends_stacked.html')
    print(f"\n✓ Saved: {PLOT_DIR / 'domain_trends_stacked.html'}")

    # Line chart (non-stacked)
    fig2 = go.Figure()

    for domain in top_domains:
        counts = []
        for filename in sorted(file_dates.keys(), key=lambda x: file_dates[x]):
            counts.append(domain_by_file[filename].get(domain, 0))

        fig2.add_trace(go.Scatter(
            x=[file_dates[f] for f in sorted(file_dates.keys(), key=lambda x: file_dates[x])],
            y=counts,
            mode='lines+markers',
            name=domain,
            line=dict(width=PLOT_CONFIG['line_width']),
            marker=dict(size=PLOT_CONFIG['marker_size'])
        ))

    fig2.update_layout(
        title=dict(text=f'Top {top_n} Domain Usage Over Time',
                  font=dict(size=PLOT_CONFIG['plotly_title_size'])),
        xaxis_title='Date',
        yaxis_title='Number of Tabs',
        font=dict(size=PLOT_CONFIG['plotly_font_size']),
        hovermode='x unified',
        height=PLOT_CONFIG['plotly_height']
    )
    fig2.write_html(PLOT_DIR / 'domain_trends_lines.html')
    print(f"✓ Saved: {PLOT_DIR / 'domain_trends_lines.html'}")


def analyze_domain_categories(all_tabs):
    """Categorize domains and analyze distribution."""
    print(f"\n{'='*60}")
    print("DOMAIN CATEGORY ANALYSIS")
    print(f"{'='*60}")

    # Define categories based on common patterns
    categories = {
        'Email': ['gmail.com', 'outlook.com', 'mail.google.com', 'yahoo.com'],
        'Social Media': ['facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com', 'reddit.com'],
        'Development': ['github.com', 'stackoverflow.com', 'gitlab.com', 'bitbucket.org'],
        'Video': ['youtube.com', 'vimeo.com', 'twitch.tv', 'netflix.com'],
        'News': ['nytimes.com', 'bbc.com', 'cnn.com', 'reuters.com', 'theguardian.com'],
        'Shopping': ['amazon.com', 'ebay.com', 'etsy.com', 'walmart.com'],
        'Google Services': ['google.com', 'docs.google.com', 'drive.google.com', 'calendar.google.com'],
        'Cloud/Productivity': ['notion.so', 'slack.com', 'trello.com', 'asana.com', 'monday.com'],
    }

    category_counts = defaultdict(int)
    uncategorized = 0

    for tab in all_tabs:
        domain = tab['domain']
        categorized = False

        for category, domain_list in categories.items():
            if any(d in domain for d in domain_list):
                category_counts[category] += 1
                categorized = True
                break

        if not categorized and domain != 'unknown':
            uncategorized += 1

    if uncategorized > 0:
        category_counts['Other'] = uncategorized

    print(f"Category distribution:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category:20s}: {count:6d}")

    # Pie chart
    labels = list(category_counts.keys())
    values = list(category_counts.values())

    plt.figure(figsize=PLOT_CONFIG['fig_size_square'])
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': PLOT_CONFIG['annotation_size']})
    plt.title('Domain Categories Distribution', fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'domain_categories.png', dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {PLOT_DIR / 'domain_categories.png'}")

    # Plotly
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3,
                                 textfont=dict(size=PLOT_CONFIG['plotly_font_size']))])
    fig.update_layout(
        title=dict(text='Domain Categories Distribution (Interactive)',
                  font=dict(size=PLOT_CONFIG['plotly_title_size'])),
        font=dict(size=PLOT_CONFIG['plotly_font_size']),
        height=PLOT_CONFIG['plotly_height']
    )
    fig.write_html(PLOT_DIR / 'domain_categories.html')
    print(f"✓ Saved: {PLOT_DIR / 'domain_categories.html'}")


def generate_summary_dashboard(all_tabs, all_data, domain_counts):
    """Generate a comprehensive summary dashboard."""
    print(f"\n{'='*60}")
    print("GENERATING SUMMARY DASHBOARD")
    print(f"{'='*60}")

    # Calculate statistics
    total_tabs = len(all_tabs)
    unique_domains = len(set(tab['domain'] for tab in all_tabs if tab['domain'] != 'unknown'))
    open_tabs = sum(1 for tab in all_tabs if tab['closedAt'] is None)
    closed_tabs = sum(1 for tab in all_tabs if tab['closedAt'] is not None)

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Top 10 Domains', 'Tab Lifecycle',
                       'Tabs Over Time', 'Domain Diversity'),
        specs=[[{'type': 'bar'}, {'type': 'pie'}],
               [{'type': 'scatter'}, {'type': 'indicator'}]]
    )

    # Top domains bar chart
    top_10 = domain_counts.most_common(10)
    fig.add_trace(
        go.Bar(x=[d for d, _ in top_10], y=[c for _, c in top_10],
               marker_color='lightblue', name='Tabs'),
        row=1, col=1
    )

    # Lifecycle pie chart
    fig.add_trace(
        go.Pie(labels=['Open', 'Closed'], values=[open_tabs, closed_tabs],
               marker=dict(colors=['#66c2a5', '#fc8d62']), name='Lifecycle'),
        row=1, col=2
    )

    # Tabs over time
    export_dates = []
    node_counts = []
    for file_data in all_data:
        export_date = file_data['metadata'].get('exportDate')
        node_count = file_data['metadata'].get('nodeCount', 0)
        if export_date:
            try:
                dt = datetime.fromisoformat(export_date.replace('Z', '+00:00'))
                export_dates.append(dt)
                node_counts.append(node_count)
            except:
                pass

    df_temp = pd.DataFrame({'date': export_dates, 'count': node_counts}).sort_values('date')
    fig.add_trace(
        go.Scatter(x=df_temp['date'], y=df_temp['count'],
                  mode='lines+markers', name='Tab Count',
                  line=dict(color='purple', width=2)),
        row=2, col=1
    )

    # Domain diversity indicator
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=unique_domains,
            title={'text': "Unique Domains"},
            delta={'reference': 100},
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=1200,
        showlegend=False,
        title_text=f"Tab Tree Analysis Dashboard - {total_tabs:,} Total Tabs",
        title_font_size=PLOT_CONFIG['plotly_title_size'],
        font=dict(size=PLOT_CONFIG['plotly_font_size'])
    )

    fig.write_html(PLOT_DIR / 'summary_dashboard.html')
    print(f"\n✓ Saved: {PLOT_DIR / 'summary_dashboard.html'}")


def main():
    """Main analysis function."""
    print("="*60)
    print("TAB TREE HISTORY ANALYSIS")
    print("="*60)

    # Load data
    all_data = load_all_tab_trees()
    if not all_data:
        print("No data files found!")
        return

    # Collect all tabs
    all_tabs = collect_all_tabs(all_data)
    print(f"\nTotal tabs collected: {len(all_tabs):,}")

    # Run analyses
    domain_counts = analyze_domain_distribution(all_tabs, top_n=15)
    analyze_temporal_patterns(all_data, all_tabs)
    analyze_tab_lifecycle(all_tabs)
    analyze_domain_trends_over_time(all_tabs, top_n=10)
    analyze_domain_categories(all_tabs)
    generate_summary_dashboard(all_tabs, all_data, domain_counts)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print(f"All plots saved to: {PLOT_DIR.absolute()}")
    print(f"\nGenerated files:")
    for plot_file in sorted(PLOT_DIR.glob('*')):
        print(f"  - {plot_file.name}")


if __name__ == "__main__":
    main()

