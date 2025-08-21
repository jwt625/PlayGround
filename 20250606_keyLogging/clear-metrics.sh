#!/bin/bash

# Script to clear/reset metrics from Grafana Cloud and local Prometheus
# This helps when you've had metric explosion and want to start fresh

set -e

echo "🧹 Metrics Cleanup Script"
echo "========================="

# Load environment variables if .env exists
if [ -f .env ]; then
    echo "📂 Loading .env file..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Function to clear local Prometheus data
clear_local_prometheus() {
    echo ""
    echo "🗂️  Clearing Local Prometheus Data..."
    
    # Stop Prometheus container
    echo "⏹️  Stopping Prometheus container..."
    docker-compose stop prometheus || true
    
    # Remove Prometheus data volume
    echo "🗑️  Removing Prometheus data volume..."
    docker volume rm 20250606_keylogging_prometheus_data 2>/dev/null || true
    
    # Restart Prometheus
    echo "🔄 Restarting Prometheus..."
    docker-compose up -d prometheus
    
    echo "✅ Local Prometheus data cleared!"
}

# Function to clear application state files
clear_app_state() {
    echo ""
    echo "🗂️  Clearing Application State Files..."
    
    # Remove all temporary state files
    rm -f /tmp/keystroke_tracker_* 2>/dev/null || true
    
    echo "✅ Application state files cleared!"
}

# Function to restart the application
restart_app() {
    echo ""
    echo "🔄 Restarting Application..."
    
    # Kill any running instances
    pkill -f "keystroke-tracker" 2>/dev/null || true
    pkill -f "tracker.swift" 2>/dev/null || true
    
    echo "✅ Application processes stopped!"
    echo "💡 Run './start.sh' to restart with clean metrics"
}

# Function to show Grafana Cloud cleanup instructions
show_grafana_cloud_instructions() {
    echo ""
    echo "☁️  Grafana Cloud Cleanup Instructions"
    echo "======================================"
    echo ""
    echo "To clear metrics from Grafana Cloud, you have a few options:"
    echo ""
    echo "1. 📊 Dashboard Level (Recommended):"
    echo "   - Go to your Grafana dashboard"
    echo "   - Click the time range picker"
    echo "   - Select a future time range (e.g., 'Last 5 minutes' from now)"
    echo "   - This will show empty graphs until new data arrives"
    echo ""
    echo "2. 🔄 Metric Relabeling (Advanced):"
    echo "   - Add metric_relabel_configs to prometheus.yml to drop old metrics"
    echo "   - This prevents old high-cardinality metrics from being scraped"
    echo ""
    echo "3. ⏰ Wait for Retention:"
    echo "   - Grafana Cloud free tier typically retains data for 14 days"
    echo "   - Old metrics will automatically expire"
    echo ""
    echo "4. 🆕 Create New Workspace (Nuclear Option):"
    echo "   - Create a new Grafana Cloud workspace"
    echo "   - Update your .env file with new credentials"
    echo ""
    
    if [ ! -z "$GRAFANA_ENDPOINT" ]; then
        echo "Your current Grafana endpoint: $GRAFANA_ENDPOINT"
    fi
}

# Function to add metric relabeling to drop old high-cardinality metrics
add_metric_relabeling() {
    echo ""
    echo "🏷️  Adding Metric Relabeling to Drop High-Cardinality Metrics..."
    
    # Check if prometheus.yml.template exists
    if [ ! -f prometheus.yml.template ]; then
        echo "❌ prometheus.yml.template not found!"
        return 1
    fi
    
    # Create a backup
    cp prometheus.yml.template prometheus.yml.template.backup
    
    # Add metric relabeling configuration
    cat >> prometheus.yml.template << 'EOF'

  # Drop high-cardinality metrics to prevent explosion
  metric_relabel_configs:
    # Drop old domain-specific metrics (before we fixed cardinality)
    - source_labels: [__name__]
      regex: '(keystrokes_total|keystroke_interval_activity|mouse_clicks_total).*'
      target_label: __tmp_has_domain
      replacement: 'check'
    - source_labels: [__tmp_has_domain, domain]
      regex: 'check;.+'
      action: drop
    
    # Drop individual domain metrics for Chrome (keep only categories)
    - source_labels: [__name__, domain]
      regex: 'chrome_.*_total;(?!development|google_services|social_media|entertainment|communication|information|other).*'
      action: drop
EOF
    
    echo "✅ Metric relabeling added to prometheus.yml.template"
    echo "🔄 Regenerating prometheus.yml..."
    
    # Regenerate prometheus.yml
    ./generate-prometheus-config.sh
    
    # Restart Prometheus
    docker-compose restart prometheus
    
    echo "✅ Prometheus restarted with metric relabeling!"
}

# Main menu
echo ""
echo "What would you like to clear?"
echo ""
echo "1) 🗂️  Local Prometheus data (removes all stored metrics)"
echo "2) 📁 Application state files (clears temp files)"
echo "3) 🔄 Restart application processes"
echo "4) 🏷️  Add metric relabeling (prevents high-cardinality metrics)"
echo "5) ☁️  Show Grafana Cloud cleanup instructions"
echo "6) 🧹 Full cleanup (all of the above)"
echo "7) ❌ Cancel"
echo ""
read -p "Enter your choice (1-7): " choice

case $choice in
    1)
        clear_local_prometheus
        ;;
    2)
        clear_app_state
        ;;
    3)
        restart_app
        ;;
    4)
        add_metric_relabeling
        ;;
    5)
        show_grafana_cloud_instructions
        ;;
    6)
        echo "🧹 Performing full cleanup..."
        clear_app_state
        restart_app
        clear_local_prometheus
        add_metric_relabeling
        show_grafana_cloud_instructions
        echo ""
        echo "✅ Full cleanup complete!"
        ;;
    7)
        echo "❌ Cancelled"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "🎉 Cleanup operations completed!"
echo ""
echo "📊 Next steps:"
echo "1. Run './start.sh' to restart the application with fixed metrics"
echo "2. Check your Grafana dashboard - it should show much fewer metrics"
echo "3. Monitor the /metrics endpoint to verify low cardinality"
