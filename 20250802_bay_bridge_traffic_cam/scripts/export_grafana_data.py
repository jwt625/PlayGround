#!/usr/bin/env python3
"""
Grafana Cloud Data Export Tool

This script exports historical data from Grafana Cloud using session-based authentication.
Based on the investigation documented in RFD-006.

Usage:
1. Open your Grafana Cloud dashboard in browser
2. Open Developer Tools (F12) -> Network tab
3. Refresh a dashboard or run a query
4. Find a request to /api/ds/query
5. Copy the grafana_session cookie value
6. Run this script with the session cookie

Example:
    python export_grafana_data.py --session "your_session_cookie" --start "2025-08-03T10:00:00Z" --end "2025-08-03T16:00:00Z"
"""

import argparse
import json
import requests
import sys
from datetime import datetime, timezone
import time
import os

class GrafanaDataExporter:
    def __init__(self, grafana_url, session_cookie, org_id=1):
        self.grafana_url = grafana_url.rstrip('/')
        self.session_cookie = session_cookie
        self.org_id = org_id
        self.session = requests.Session()
        
        # Set up session headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'X-Grafana-Org-Id': str(org_id),
            'X-Datasource-Uid': 'grafanacloud-prom',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # Set session cookie
        self.session.cookies.set('grafana_session', session_cookie, domain=grafana_url.split('//')[1])
    
    def export_metric_data(self, metric_name, start_time, end_time, step=60):
        """
        Export data for a specific metric from Grafana Cloud
        
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
        
        query_data = {
            "queries": [
                {
                    "refId": "A",
                    "expr": metric_name,
                    "range": True,
                    "format": "time_series",
                    "start": start_time,
                    "end": end_time,
                    "step": step,
                    "maxDataPoints": 1000,
                    "datasource": {
                        "uid": "grafanacloud-prom",
                        "type": "prometheus"
                    }
                }
            ],
            "from": str(start_time * 1000),  # Grafana expects milliseconds
            "to": str(end_time * 1000)
        }
        
        url = f"{self.grafana_url}/api/ds/query?ds_type=prometheus"
        
        try:
            response = self.session.post(url, json=query_data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error querying metric {metric_name}: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            return None
    
    def list_available_metrics(self):
        """
        Get list of available metrics from Grafana Cloud
        """
        query_data = {
            "queries": [
                {
                    "refId": "A",
                    "expr": "{__name__=~\".+\"}",
                    "range": False,
                    "format": "table",
                    "datasource": {
                        "uid": "grafanacloud-prom",
                        "type": "prometheus"
                    }
                }
            ]
        }
        
        url = f"{self.grafana_url}/api/ds/query?ds_type=prometheus"
        
        try:
            response = self.session.post(url, json=query_data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error listing metrics: {e}")
            return None
    
    def export_to_prometheus_format(self, data, output_file):
        """
        Convert Grafana response to Prometheus exposition format
        """
        if not data or 'results' not in data:
            print("No data to export")
            return False
        
        with open(output_file, 'w') as f:
            for result in data['results'].values():
                if 'frames' not in result:
                    continue
                
                for frame in result['frames']:
                    if 'data' not in frame or 'values' not in frame['data']:
                        continue
                    
                    # Extract metric name and labels
                    metric_name = frame.get('name', 'unknown_metric')
                    
                    # Write data points
                    timestamps = frame['data']['values'][0] if len(frame['data']['values']) > 0 else []
                    values = frame['data']['values'][1] if len(frame['data']['values']) > 1 else []
                    
                    for timestamp, value in zip(timestamps, values):
                        if value is not None:
                            # Convert timestamp from milliseconds to seconds
                            ts_seconds = int(timestamp / 1000)
                            f.write(f"{metric_name} {value} {ts_seconds}\n")
        
        print(f"Data exported to {output_file}")
        return True

def main():
    parser = argparse.ArgumentParser(description='Export data from Grafana Cloud')
    parser.add_argument('--grafana-url', default='https://USERNAME.grafana.net',
                       help='Grafana Cloud URL (replace USERNAME with your instance)')
    parser.add_argument('--session', required=True,
                       help='Grafana session cookie value')
    parser.add_argument('--metric', default='traffic_vehicles_total',
                       help='Metric name to export (default: traffic_vehicles_total)')
    parser.add_argument('--start',
                       help='Start time (ISO format: 2025-08-03T10:00:00Z)')
    parser.add_argument('--end',
                       help='End time (ISO format: 2025-08-03T16:00:00Z)')
    parser.add_argument('--step', type=int, default=60,
                       help='Query step in seconds (default: 60)')
    parser.add_argument('--output', default='exported_metrics.txt',
                       help='Output file name (default: exported_metrics.txt)')
    parser.add_argument('--list-metrics', action='store_true',
                       help='List available metrics and exit')
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.session:
        print("Error: Session cookie is required. Extract it from browser Developer Tools.")
        sys.exit(1)

    # Check if we're just listing metrics
    if args.list_metrics:
        # Skip time validation for list-metrics
        pass
    elif not args.start or not args.end:
        print("Error: Start and end times are required for data export")
        print("Use --start and --end with ISO format: 2025-08-03T10:00:00Z")
        sys.exit(1)
    
    # Create exporter
    exporter = GrafanaDataExporter(args.grafana_url, args.session)
    
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
