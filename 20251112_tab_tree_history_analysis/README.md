# Tab Tree History Analysis

This project analyzes browser tab history exported from a tab tree extension to provide insights into browsing patterns and behavior.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your tab-tree JSON files are in the `./data/` directory

## Running the Analysis

```bash
python analyze_tab_tree.py
```

## Analyses Performed

### 1. **Domain Distribution Analysis**
- Identifies the most frequently visited domains
- Generates pie charts and bar charts showing top domains
- Outputs: `domain_distribution_pie.png/html`, `domain_distribution_bar.png/html`

### 2. **Temporal Patterns Analysis**
- **Daily Tab Creation Timeline**: Shows tab creation over the entire date range (Nov 2024 - Nov 2025)
- **Monthly Tab Creation**: Aggregates tabs created per month
- **Hourly Patterns**: Analyzes what time of day you create most tabs
- **Day of Week Patterns**: Shows which days you're most active
- Outputs: `tab_creation_timeline.png/html`, `monthly_tab_creation.png/html`, `hourly_tab_creation.png/html`, `daily_tab_creation.png/html`

### 3. **Tab Lifecycle Analysis**
- Compares open vs closed tabs
- **Tab Lifespan Analysis**: Shows how long tabs stay open before being closed
- Statistics: average, median, min, max lifespan
- Distribution and box plot visualizations
- Outputs: `tab_lifecycle.png/html`, `tab_lifespan_distribution.png/html`, `tab_lifespan_boxplot.html`

### 4. **Domain Trends Over Time**
- Tracks how usage of top domains evolves across exports
- Shows both stacked and line chart views
- Outputs: `domain_trends_stacked.html`, `domain_trends_lines.html`

### 5. **Domain Category Analysis**
- Categorizes domains (Email, Social Media, Development, Video, etc.)
- Shows distribution across categories
- Outputs: `domain_categories.png/html`

### 6. **Summary Dashboard**
- Comprehensive overview combining multiple metrics
- Interactive dashboard with key statistics
- Outputs: `summary_dashboard.html`

## Output Files

All visualizations are saved to the `./plots/` directory in both:
- **PNG format**: High-resolution static images (300 DPI, mobile-friendly aspect ratios)
- **HTML format**: Interactive Plotly visualizations (responsive design)

## Customizing Plot Appearance

All plot settings (font sizes, figure dimensions, colors) are centralized in the `PLOT_CONFIG` dictionary at the top of `analyze_tab_tree.py`.

See `PLOT_CONFIG_GUIDE.md` for detailed customization instructions.

**Quick adjustments:**
- **Larger fonts**: Increase `title_size`, `label_size`, `tick_size` values
- **Bigger plots**: Increase `fig_size_*` tuples
- **Higher resolution**: Increase `dpi` value
- **Mobile-friendly**: Current settings already optimized (aspect ratios ~1:1)

## Data Structure

The script expects JSON files with the following structure:
```json
{
  "metadata": {
    "exportDate": "ISO timestamp",
    "nodeCount": number
  },
  "tabTree": {
    "node-id": {
      "url": "string",
      "title": "string",
      "createdAt": timestamp,
      "closedAt": timestamp or null,
      "children": [...]
    }
  }
}
```

