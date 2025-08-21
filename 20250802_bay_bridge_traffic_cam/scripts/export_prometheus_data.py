#!/usr/bin/env python3
"""
Export data from Prometheus API directly

This script exports metric data from Prometheus API using basic authentication.
It queries the Prometheus endpoint directly instead of going through Grafana.

Usage:
    python export_prometheus_data.py --start "2025-08-03T10:00:00Z" --end "2025-08-03T16:00:00Z"

Example:
    python export_prometheus_data.py --metric "traffic_vehicles_total" --start "2025-08-03T10:00:00Z" --end "2025-08-03T16:00:00Z"

Environment Variables (set in .env file):
    PROMETHEUS_QUERY_URL - Prometheus API URL
    PROMETHEUS_USERNAME - Username for authentication
    GRAFANA_READ_TOKEN - API token for authentication
"""

import requests
import json
import argparse
import sys
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PrometheusDataExporter:
    def __init__(self, prometheus_url, username, password):
        self.prometheus_url = prometheus_url.rstrip('/')
        self.username = username
        self.password = password
        self.session = requests.Session()
        
        # Set up basic auth
        self.session.auth = (username, password)
        
        # Set up session headers
        self.session.headers.update({
            'User-Agent': 'PrometheusDataExporter/1.0'
        })
    
    def export_metric_data(self, metric_name, start_time, end_time, step=60):
        """
        Export data for a specific metric from Prometheus
        
        Args:
            metric_name: Prometheus metric name (e.g., 'traffic_vehicles_total')
            start_time: Start timestamp (Unix timestamp or ISO string)
            end_time: End timestamp (Unix timestamp or ISO string)
            step: Query step in seconds (default: 60)
        
        Returns:
            dict: Query response data
        """
        
        # Convert timestamps if needed
        if isinstance(start_time, str):
            start_time = int(datetime.fromisoformat(start_time.replace('Z', '+00:00')).timestamp())
        if isinstance(end_time, str):
            end_time = int(datetime.fromisoformat(end_time.replace('Z', '+00:00')).timestamp())
        
        # Build query parameters
        params = {
            'query': metric_name,
            'start': start_time,
            'end': end_time,
            'step': f'{step}s'
        }
        
        url = f"{self.prometheus_url}/api/v1/query_range"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error querying metric {metric_name}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return None
    
    def list_available_metrics(self):
        """
        Get list of available metrics from Prometheus
        """
        url = f"{self.prometheus_url}/api/v1/label/__name__/values"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error listing metrics: {e}")
            return None
    
    def export_to_prometheus_format(self, data, output_file):
        """
        Convert Prometheus response to Prometheus exposition format
        """
        if not data or data.get('status') != 'success':
            print("No data to export or query failed")
            return False
        
        try:
            with open(output_file, 'w') as f:
                result = data.get('data', {})
                result_type = result.get('resultType', '')
                
                if result_type == 'matrix':
                    for series in result.get('result', []):
                        metric = series.get('metric', {})
                        values = series.get('values', [])
                        
                        # Write metric name and labels
                        metric_name = metric.get('__name__', 'unknown_metric')
                        labels = []
                        for key, value in metric.items():
                            if key != '__name__':
                                labels.append(f'{key}="{value}"')
                        
                        label_str = '{' + ','.join(labels) + '}' if labels else ''
                        
                        # Write time series data
                        for timestamp, value in values:
                            f.write(f"{metric_name}{label_str} {value} {int(float(timestamp) * 1000)}\n")
                
                elif result_type == 'vector':
                    for series in result.get('result', []):
                        metric = series.get('metric', {})
                        value_data = series.get('value', [])
                        
                        if len(value_data) >= 2:
                            timestamp, value = value_data
                            
                            # Write metric name and labels
                            metric_name = metric.get('__name__', 'unknown_metric')
                            labels = []
                            for key, val in metric.items():
                                if key != '__name__':
                                    labels.append(f'{key}="{val}"')
                            
                            label_str = '{' + ','.join(labels) + '}' if labels else ''
                            f.write(f"{metric_name}{label_str} {value} {int(float(timestamp) * 1000)}\n")
                
                print(f"✓ Exported to Prometheus format: {output_file}")
                return True
                
        except Exception as e:
            print(f"Error writing Prometheus format: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Export data from Prometheus API')
    parser.add_argument('--prometheus-url',
                       default=os.getenv('PROMETHEUS_QUERY_URL'),
                       help='Prometheus API URL (default: from PROMETHEUS_QUERY_URL env var)')
    parser.add_argument('--username',
                       default=os.getenv('PROMETHEUS_USERNAME'),
                       help='Prometheus username (default: from PROMETHEUS_USERNAME env var)')
    parser.add_argument('--password',
                       default=os.getenv('GRAFANA_READ_TOKEN'),
                       help='Prometheus password/API key (default: from GRAFANA_READ_TOKEN env var)')
    parser.add_argument('--metric', default='traffic_vehicles_total',
                       help='Metric name to export (default: traffic_vehicles_total)')
    parser.add_argument('--start',
                       help='Start time (ISO format: 2025-08-03T10:00:00Z)')
    parser.add_argument('--end',
                       help='End time (ISO format: 2025-08-03T16:00:00Z)')
    parser.add_argument('--step', type=int, default=60,
                       help='Query step in seconds (default: 60)')
    parser.add_argument('--output', default='exported_data.txt',
                       help='Output file (default: exported_data.txt)')
    parser.add_argument('--list-metrics', action='store_true',
                       help='List available metrics and exit')
    
    args = parser.parse_args()

    # Validate required configuration
    if not args.prometheus_url:
        print("Error: Prometheus URL is required")
        print("Set PROMETHEUS_QUERY_URL environment variable or use --prometheus-url")
        sys.exit(1)

    if not args.username:
        print("Error: Username is required")
        print("Set PROMETHEUS_USERNAME environment variable or use --username")
        sys.exit(1)

    if not args.password:
        print("Error: Password/API key is required")
        print("Set GRAFANA_READ_TOKEN environment variable or use --password")
        sys.exit(1)

    # Validate required arguments for data export
    if not args.list_metrics and (not args.start or not args.end):
        print("Error: Start and end times are required for data export")
        print("Use --start and --end with ISO format: 2025-08-03T10:00:00Z")
        sys.exit(1)
    
    # Create exporter
    exporter = PrometheusDataExporter(args.prometheus_url, args.username, args.password)
    
    # List metrics if requested
    if args.list_metrics:
        print("Fetching available metrics...")
        metrics = exporter.list_available_metrics()
        if metrics:
            print(json.dumps(metrics, indent=2))
        sys.exit(0)
    
    # Export metric data
    print(f"Exporting {args.metric} from {args.start} to {args.end}")
    data = exporter.export_metric_data(args.metric, args.start, args.end, args.step)
    
    if data:
        # Save raw JSON response
        json_file = args.output.replace('.txt', '.json')
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Raw data saved to {json_file}")
        
        # Convert to Prometheus format
        exporter.export_to_prometheus_format(data, args.output)
    else:
        print("Failed to export data")
        sys.exit(1)

if __name__ == '__main__':
    main()
