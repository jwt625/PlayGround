#!/usr/bin/env python3
"""
Export all metrics used in the Grafana dashboard for the last 24 hours

This script exports all the metrics that are displayed in the local Grafana dashboard
from the cloud Prometheus instance for analysis and backup purposes.
"""

import os
import sys
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_prometheus_data import PrometheusDataExporter

# Load environment variables
load_dotenv()

# Metrics used in the Grafana dashboard
DASHBOARD_METRICS = [
    # Core traffic metrics
    'traffic_vehicles_total',
    'traffic_flow_rate_per_minute', 
    
    # System health metrics
    'system_status',
    
    # Performance metrics
    'tracked_objects_active',
    'motion_detector_fps',
    'frame_processing_time_seconds_bucket',
    
    # Additional metrics that might be available
    'traffic_vehicles_created',
    'frame_processing_time_seconds_count',
    'frame_processing_time_seconds_sum',
]

def calculate_time_range():
    """Calculate start and end times for the last 24 hours"""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=24)
    
    return start_time.isoformat() + 'Z', end_time.isoformat() + 'Z'

def export_all_dashboard_metrics():
    """Export all dashboard metrics for the last 24 hours"""
    
    # Get configuration from environment
    prometheus_url = os.getenv('PROMETHEUS_QUERY_URL')
    username = os.getenv('PROMETHEUS_USERNAME')
    password = os.getenv('GRAFANA_READ_TOKEN')
    
    if not all([prometheus_url, username, password]):
        print("Error: Missing required environment variables")
        print("Required: PROMETHEUS_QUERY_URL, PROMETHEUS_USERNAME, GRAFANA_READ_TOKEN")
        return False
    
    # Calculate time range
    start_time, end_time = calculate_time_range()
    print(f"Exporting dashboard metrics from {start_time} to {end_time}")
    print(f"Time range: Last 24 hours")
    print()
    
    # Create exporter
    exporter = PrometheusDataExporter(prometheus_url, username, password)
    
    # Create output directory
    output_dir = f"dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    successful_exports = 0
    failed_exports = 0
    
    # Export each metric
    for metric in DASHBOARD_METRICS:
        print(f"📊 Exporting {metric}...")
        
        try:
            # Export metric data
            data = exporter.export_metric_data(metric, start_time, end_time, step=30)
            
            if data and data.get('status') == 'success':
                result = data.get('data', {}).get('result', [])
                
                if result:
                    # Save raw JSON
                    json_file = os.path.join(output_dir, f"{metric}.json")
                    with open(json_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    # Save Prometheus format
                    txt_file = os.path.join(output_dir, f"{metric}.txt")
                    exporter.export_to_prometheus_format(data, txt_file)
                    
                    # Count data points
                    total_points = sum(len(series.get('values', [])) for series in result)
                    print(f"  ✅ Success: {len(result)} series, {total_points} data points")
                    successful_exports += 1
                else:
                    print(f"  ⚠️  No data available for {metric}")
                    failed_exports += 1
            else:
                print(f"  ❌ Failed to export {metric}")
                failed_exports += 1
                
        except Exception as e:
            print(f"  ❌ Error exporting {metric}: {e}")
            failed_exports += 1
    
    print()
    print("=" * 60)
    print(f"📈 Export Summary")
    print(f"✅ Successful exports: {successful_exports}")
    print(f"❌ Failed exports: {failed_exports}")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 60)
    
    # Create summary file
    summary = {
        "export_timestamp": datetime.now().isoformat(),
        "time_range": {
            "start": start_time,
            "end": end_time,
            "duration_hours": 24
        },
        "metrics_exported": DASHBOARD_METRICS,
        "results": {
            "successful": successful_exports,
            "failed": failed_exports,
            "total": len(DASHBOARD_METRICS)
        },
        "output_directory": output_dir
    }
    
    summary_file = os.path.join(output_dir, "export_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Export summary saved to: {summary_file}")
    
    return successful_exports > 0

def main():
    """Main entry point"""
    print("🚀 Bay Bridge Traffic Dashboard Metrics Export")
    print("=" * 60)
    print(f"📊 Metrics to export: {len(DASHBOARD_METRICS)}")
    print(f"⏰ Time range: Last 24 hours")
    print(f"🔗 Source: Cloud Prometheus")
    print("=" * 60)
    print()
    
    success = export_all_dashboard_metrics()
    
    if success:
        print("\n🎉 Export completed successfully!")
        print("💡 You can now analyze the exported data or import it into other systems.")
    else:
        print("\n❌ Export failed. Check your configuration and network connection.")
        sys.exit(1)

if __name__ == '__main__':
    main()
