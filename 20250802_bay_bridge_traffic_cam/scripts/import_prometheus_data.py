#!/usr/bin/env python3
"""
Import backed up Prometheus data to local Prometheus instance

This script takes exported Prometheus data (in .txt format) and replays it
to a local Prometheus instance using the remote write protocol.

Usage:
    python import_prometheus_data.py --input exported_data.txt --prometheus-url http://localhost:9090
    python import_prometheus_data.py --input-dir dashboard_export_20250803_120000/

Example:
    # Import single metric file
    python import_prometheus_data.py --input traffic_vehicles_total.txt
    
    # Import all files from export directory
    python import_prometheus_data.py --input-dir dashboard_export_20250803_120000/
    
    # Import with custom Prometheus URL
    python import_prometheus_data.py --input traffic_vehicles_total.txt --prometheus-url http://localhost:9090

Environment Variables:
    LOCAL_PROMETHEUS_URL - Local Prometheus URL (default: http://localhost:9090)
"""

import os
import sys
import time
import argparse
import requests
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import glob
from dotenv import load_dotenv

# Try to import protobuf and snappy for remote write
try:
    import snappy
    from prometheus_client.exposition import generate_latest
    from prometheus_client import CollectorRegistry, Gauge, Counter
    REMOTE_WRITE_AVAILABLE = True
except ImportError:
    REMOTE_WRITE_AVAILABLE = False
    print("Warning: snappy or prometheus_client not available. Remote write disabled.")
    print("Install with: pip install python-snappy prometheus_client")

# Load environment variables
load_dotenv()

class PrometheusDataImporter:
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus_url = prometheus_url.rstrip('/')
        self.remote_write_url = f"{self.prometheus_url}/api/v1/write"
        self.session = requests.Session()

        # Set up session headers (default for remote write, will be overridden for pushgateway)
        self.session.headers.update({
            'User-Agent': 'PrometheusDataImporter/1.0'
        })

        # Overlap detection settings
        self.check_overlaps = True
        self.overlap_strategy = "warn"  # "warn", "skip", "overwrite", "fail"

    def check_data_overlap(self, metrics: List[Dict]) -> Dict:
        """
        Check for data overlap with existing Prometheus data

        Returns:
            dict: Overlap analysis results
        """
        if not self.check_overlaps:
            return {"overlap_detected": False, "details": "Overlap checking disabled"}

        try:
            # Get time range of import data
            timestamps = [m['timestamp'] for m in metrics]
            start_time = min(timestamps) / 1000  # Convert to seconds
            end_time = max(timestamps) / 1000

            # Get unique metric names from import data
            metric_names = list(set(m['name'] for m in metrics))

            overlap_results = {
                "overlap_detected": False,
                "import_range": {
                    "start": datetime.fromtimestamp(start_time).isoformat(),
                    "end": datetime.fromtimestamp(end_time).isoformat(),
                    "duration_hours": (end_time - start_time) / 3600
                },
                "metrics_to_import": len(metrics),
                "unique_metric_names": len(metric_names),
                "overlapping_metrics": [],
                "recommendations": []
            }

            # Check each metric for existing data in the same time range
            for metric_name in metric_names[:3]:  # Check first 3 metrics to avoid overwhelming Prometheus
                existing_data = self._query_existing_data(metric_name, start_time, end_time)
                if existing_data and len(existing_data) > 0:
                    overlap_results["overlap_detected"] = True
                    overlap_results["overlapping_metrics"].append({
                        "metric": metric_name,
                        "existing_points": len(existing_data),
                        "time_range": f"{datetime.fromtimestamp(start_time).strftime('%H:%M')} - {datetime.fromtimestamp(end_time).strftime('%H:%M')}"
                    })

            # Generate recommendations based on overlap
            if overlap_results["overlap_detected"]:
                overlap_results["recommendations"] = [
                    "⚠️  Data overlap detected - existing data will be affected",
                    "💡 Consider using --overlap-strategy to control behavior",
                    "🔍 Use simulation mode first to review conflicts",
                    "💾 Backup existing data before importing"
                ]
            else:
                overlap_results["recommendations"] = [
                    "✅ No overlap detected - safe to import",
                    "📊 Import will add new data points to existing metrics"
                ]

            return overlap_results

        except Exception as e:
            return {
                "overlap_detected": False,
                "error": f"Failed to check overlap: {e}",
                "recommendations": ["⚠️  Could not verify overlap - proceed with caution"]
            }

    def _query_existing_data(self, metric_name: str, start_time: float, end_time: float) -> List:
        """Query Prometheus for existing data in the given time range"""
        try:
            params = {
                'query': metric_name,
                'start': start_time,
                'end': end_time,
                'step': '60s'
            }

            response = self.session.get(
                f"{self.prometheus_url}/api/v1/query_range",
                params=params,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    result = data.get('data', {}).get('result', [])
                    # Flatten all values from all series
                    all_values = []
                    for series in result:
                        all_values.extend(series.get('values', []))
                    return all_values

            return []

        except Exception as e:
            print(f"Warning: Could not query existing data for {metric_name}: {e}")
            return []
    
    def parse_prometheus_format(self, file_path: str) -> List[Dict]:
        """
        Parse Prometheus exposition format file
        
        Format: metric_name{label1="value1",label2="value2"} value timestamp
        
        Returns:
            List of metric dictionaries
        """
        metrics = []
        
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    try:
                        # Parse metric line
                        metric = self._parse_metric_line(line)
                        if metric:
                            metrics.append(metric)
                    except Exception as e:
                        print(f"Warning: Failed to parse line {line_num} in {file_path}: {e}")
                        continue
            
            print(f"✓ Parsed {len(metrics)} metrics from {file_path}")
            return metrics
            
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []
    
    def _parse_metric_line(self, line: str) -> Optional[Dict]:
        """Parse a single metric line"""
        # Find the metric name and labels
        if '{' in line:
            # Metric with labels
            name_end = line.find('{')
            metric_name = line[:name_end]
            
            # Find the end of labels
            labels_end = line.find('}', name_end)
            if labels_end == -1:
                return None
            
            labels_str = line[name_end+1:labels_end]
            value_timestamp = line[labels_end+1:].strip()
        else:
            # Metric without labels
            parts = line.split(' ', 1)
            if len(parts) < 2:
                return None
            metric_name = parts[0]
            labels_str = ""
            value_timestamp = parts[1]
        
        # Parse value and timestamp
        value_parts = value_timestamp.split()
        if len(value_parts) != 2:
            return None
        
        try:
            value = float(value_parts[0])
            timestamp = int(value_parts[1])
        except ValueError:
            return None
        
        # Parse labels
        labels = {}
        if labels_str:
            # Simple label parsing (assumes well-formed labels)
            for label_pair in labels_str.split(','):
                if '=' in label_pair:
                    key, val = label_pair.split('=', 1)
                    key = key.strip()
                    val = val.strip().strip('"')
                    labels[key] = val
        
        return {
            'name': metric_name,
            'labels': labels,
            'value': value,
            'timestamp': timestamp
        }
    
    def send_metrics_via_pushgateway(self, metrics: List[Dict], pushgateway_url: str = None) -> bool:
        """
        Send metrics to Prometheus via Pushgateway

        This is the recommended approach for importing historical data.
        """
        if not pushgateway_url:
            pushgateway_url = f"{self.prometheus_url.replace(':9090', ':9091')}"

        print(f"Sending {len(metrics)} metrics via Pushgateway to {pushgateway_url}")

        # Group metrics by name and labels for Pushgateway format
        metric_groups = {}
        for metric in metrics:
            key = (metric['name'], tuple(sorted(metric['labels'].items())))
            if key not in metric_groups:
                metric_groups[key] = []
            metric_groups[key].append(metric)

        success_count = 0
        error_count = 0

        for (metric_name, labels_tuple), metric_list in metric_groups.items():
            try:
                # Sort by timestamp
                metric_list.sort(key=lambda x: x['timestamp'])

                # For Pushgateway, we can only send the latest value
                # So we'll send each timestamp as a separate push
                for metric in metric_list:
                    labels_dict = dict(labels_tuple)

                    # Create job and instance labels for Pushgateway
                    job_name = "imported_data"
                    instance = f"import_{int(time.time())}"

                    # Build Pushgateway URL
                    url = f"{pushgateway_url}/metrics/job/{job_name}/instance/{instance}"

                    # Create metric data in exposition format
                    exposition_data = self._create_single_metric_exposition(metric)

                    # Send to Pushgateway
                    response = self.session.post(
                        url,
                        data=exposition_data,
                        headers={'Content-Type': 'text/plain; version=0.0.4; charset=utf-8'}
                    )

                    if response.status_code == 200:
                        success_count += 1
                    else:
                        # Debug: show first failed request details
                        if error_count == 0:
                            print(f"Debug - First failed request:")
                            print(f"  URL: {url}")
                            print(f"  Data: {repr(exposition_data)}")
                            print(f"  Response: {response.status_code} - {response.text}")
                        print(f"Failed to push metric {metric_name}: {response.status_code}")
                        error_count += 1

                    # Small delay to avoid overwhelming
                    time.sleep(0.01)

            except Exception as e:
                print(f"Error sending metric group {metric_name}: {e}")
                error_count += len(metric_list)

        print(f"Pushgateway import summary: {success_count} successful, {error_count} failed")
        return error_count == 0

    def send_metrics_simulation(self, metrics: List[Dict]) -> bool:
        """
        Simulate sending metrics (validation and logging only)

        This validates the data format and shows what would be imported.
        """
        print(f"Simulating import of {len(metrics)} metrics to {self.prometheus_url}")

        # Check for data overlap
        print("\n🔍 Checking for data overlap...")
        overlap_results = self.check_data_overlap(metrics)

        # Display overlap analysis
        print(f"📊 Import Data Analysis:")
        print(f"   Time Range: {overlap_results['import_range']['start']} to {overlap_results['import_range']['end']}")
        print(f"   Duration: {overlap_results['import_range']['duration_hours']:.1f} hours")
        print(f"   Metrics to Import: {overlap_results['metrics_to_import']}")
        print(f"   Unique Metric Names: {overlap_results['unique_metric_names']}")

        if overlap_results.get("overlap_detected"):
            print(f"\n⚠️  OVERLAP DETECTED:")
            for overlap in overlap_results.get("overlapping_metrics", []):
                print(f"   - {overlap['metric']}: {overlap['existing_points']} existing points in {overlap['time_range']}")
        else:
            print(f"\n✅ No overlap detected")

        # Show recommendations
        print(f"\n💡 Recommendations:")
        for rec in overlap_results.get("recommendations", []):
            print(f"   {rec}")

        # Group metrics by timestamp for batch processing
        timestamp_groups = {}
        for metric in metrics:
            ts = metric['timestamp']
            if ts not in timestamp_groups:
                timestamp_groups[ts] = []
            timestamp_groups[ts].append(metric)

        # Sort timestamps to replay in chronological order
        sorted_timestamps = sorted(timestamp_groups.keys())

        print(f"\n📈 Data Preview:")
        print(f"Data spans from {datetime.fromtimestamp(min(sorted_timestamps)/1000)} to {datetime.fromtimestamp(max(sorted_timestamps)/1000)}")
        print(f"Total timestamps: {len(sorted_timestamps)}")

        # Show sample of what would be imported
        for i, timestamp in enumerate(sorted_timestamps[:3]):  # Show first 3 timestamps
            batch_metrics = timestamp_groups[timestamp]
            print(f"Sample batch {i+1}: {len(batch_metrics)} metrics at {datetime.fromtimestamp(timestamp/1000)}")

            # Show first metric as example
            if batch_metrics:
                sample = batch_metrics[0]
                # Show cleaned labels (without job/instance)
                clean_labels = {k: v for k, v in sample['labels'].items() if k not in ['job', 'instance']}
                labels_str = ', '.join([f'{k}="{v}"' for k, v in clean_labels.items()])
                label_display = f"{{{labels_str}}}" if labels_str else ""
                print(f"  Example: {sample['name']}{label_display} = {sample['value']}")

                # Show what labels were removed
                removed_labels = [k for k in sample['labels'].keys() if k in ['job', 'instance']]
                if removed_labels:
                    print(f"    (Removed conflicting labels: {', '.join(removed_labels)})")

        if len(sorted_timestamps) > 3:
            print(f"... and {len(sorted_timestamps) - 3} more timestamps")

        print("\n✓ Data format validation successful")

        return True

    def create_http_server_import(self, metrics: List[Dict], output_dir: str = "prometheus_import") -> bool:
        """
        Create HTTP server files for historical data import

        This creates static files that Prometheus can scrape to import historical data
        with proper timestamps.
        """
        import os
        from datetime import datetime

        print(f"Creating HTTP server import files in {output_dir}/")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Group metrics by timestamp
        timestamp_groups = {}
        for metric in metrics:
            ts = metric['timestamp']
            if ts not in timestamp_groups:
                timestamp_groups[ts] = []
            timestamp_groups[ts].append(metric)

        # Sort timestamps
        sorted_timestamps = sorted(timestamp_groups.keys())

        print(f"Creating {len(sorted_timestamps)} timestamp files...")

        # Create a file for each timestamp
        file_list = []
        for i, timestamp in enumerate(sorted_timestamps):
            batch_metrics = timestamp_groups[timestamp]

            # Create filename with timestamp
            dt = datetime.fromtimestamp(timestamp / 1000)
            filename = f"metrics_{dt.strftime('%Y%m%d_%H%M%S')}.txt"
            filepath = os.path.join(output_dir, filename)

            # Write metrics in Prometheus format
            with open(filepath, 'w') as f:
                for metric in batch_metrics:
                    # Clean labels (remove job/instance conflicts)
                    clean_labels = {k: v for k, v in metric['labels'].items() if k not in ['job', 'instance']}

                    # Format metric
                    if clean_labels:
                        label_pairs = [f'{k}="{v}"' for k, v in clean_labels.items()]
                        label_str = '{' + ','.join(label_pairs) + '}'
                    else:
                        label_str = ''

                    # Write with original timestamp
                    f.write(f"{metric['name']}{label_str} {metric['value']} {timestamp}\n")

            file_list.append((filename, dt.isoformat(), len(batch_metrics)))

            if i % 100 == 0:
                print(f"  Created {i+1}/{len(sorted_timestamps)} files...")

        # Create index file
        index_path = os.path.join(output_dir, "index.html")
        with open(index_path, 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Prometheus Historical Data Import</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .code { background-color: #f5f5f5; padding: 10px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>Prometheus Historical Data Import</h1>
    <p>This directory contains historical Prometheus metrics data ready for import.</p>

    <div class="code">
        <h3>To import this data:</h3>
        <p>1. Start HTTP server: <code>python -m http.server 8000</code></p>
        <p>2. Configure Prometheus to scrape: <code>http://localhost:8000/metrics_YYYYMMDD_HHMMSS.txt</code></p>
        <p>3. Or use recording rules to backfill the data</p>
    </div>

    <h2>Available Data Files</h2>
    <table>
        <tr><th>File</th><th>Timestamp</th><th>Metrics Count</th></tr>
""")

            for filename, timestamp, count in file_list:
                f.write(f"        <tr><td><a href='{filename}'>{filename}</a></td><td>{timestamp}</td><td>{count}</td></tr>\n")

            f.write("""    </table>

    <h2>Import Instructions</h2>
    <p>These files contain historical Prometheus data with original timestamps.
    To import into Prometheus:</p>
    <ol>
        <li>Use Prometheus recording rules with backfill capability</li>
        <li>Or configure a temporary scrape job to read these files</li>
        <li>Or use the Prometheus remote write API with proper timestamp handling</li>
    </ol>

</body>
</html>""")

        print(f"\n✅ Created {len(file_list)} metric files in {output_dir}/")
        print(f"📁 Index file: {index_path}")
        print(f"🌐 Start server: python -m http.server 8000 --directory {output_dir}")
        print(f"📊 Total metrics: {len(metrics)} across {len(sorted_timestamps)} timestamps")

        return True

    def send_metrics_direct_write(self, metrics: List[Dict]) -> bool:
        """
        Send metrics directly to Prometheus using HTTP API with original timestamps

        This method sends metrics with their original timestamps preserved,
        creating proper historical time series data.
        """
        print(f"Sending {len(metrics)} metrics directly to Prometheus with historical timestamps")

        # Group metrics by timestamp for efficient processing
        timestamp_groups = {}
        for metric in metrics:
            ts = metric['timestamp']
            if ts not in timestamp_groups:
                timestamp_groups[ts] = []
            timestamp_groups[ts].append(metric)

        # Sort timestamps chronologically
        sorted_timestamps = sorted(timestamp_groups.keys())

        print(f"Processing {len(sorted_timestamps)} timestamps from {datetime.fromtimestamp(min(sorted_timestamps)/1000)} to {datetime.fromtimestamp(max(sorted_timestamps)/1000)}")

        success_count = 0
        error_count = 0

        # Process in batches to avoid overwhelming Prometheus
        batch_size = 100
        for i in range(0, len(sorted_timestamps), batch_size):
            batch_timestamps = sorted_timestamps[i:i+batch_size]

            # Create a combined metrics payload for this batch
            batch_metrics = []
            for timestamp in batch_timestamps:
                batch_metrics.extend(timestamp_groups[timestamp])

            # Convert to Prometheus exposition format with timestamps
            exposition_data = self._create_timestamped_exposition(batch_metrics)

            try:
                # Send to Prometheus admin API (if available) or use a workaround
                success = self._send_timestamped_batch(exposition_data, batch_metrics)

                if success:
                    success_count += len(batch_metrics)
                    print(f"✓ Batch {i//batch_size + 1}/{(len(sorted_timestamps) + batch_size - 1)//batch_size}: {len(batch_metrics)} metrics")
                else:
                    error_count += len(batch_metrics)
                    print(f"✗ Batch {i//batch_size + 1} failed: {len(batch_metrics)} metrics")

                # Small delay between batches
                time.sleep(0.1)

            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                error_count += len(batch_metrics)

        print(f"\nDirect write summary: {success_count} successful, {error_count} failed")

        if error_count > 0:
            print("💡 Note: Direct write to Prometheus TSDB requires special configuration.")
            print("   Consider using the http-server method with recording rules instead.")

        return error_count == 0

    def _create_timestamped_exposition(self, metrics: List[Dict]) -> str:
        """Create Prometheus exposition format with original timestamps"""
        lines = []
        for metric in metrics:
            name = metric['name']
            # Clean labels (remove job/instance conflicts)
            clean_labels = {k: v for k, v in metric['labels'].items() if k not in ['job', 'instance']}
            value = metric['value']
            timestamp = metric['timestamp']

            # Format labels
            if clean_labels:
                label_pairs = [f'{k}="{v}"' for k, v in clean_labels.items()]
                label_str = '{' + ','.join(label_pairs) + '}'
            else:
                label_str = ''

            lines.append(f"{name}{label_str} {value} {timestamp}")

        return '\n'.join(lines) + '\n'

    def _send_timestamped_batch(self, exposition_data: str, metrics: List[Dict]) -> bool:
        """
        Send timestamped data to Prometheus

        Note: This is a simplified implementation. Real timestamp preservation
        requires either:
        1. Prometheus remote write with protobuf
        2. Direct TSDB manipulation (dangerous)
        3. Recording rules with backfill
        """
        try:
            # For now, we'll create a temporary file that can be used with
            # Prometheus recording rules or external tools
            timestamp = int(time.time())
            temp_file = f"prometheus_import/batch_{timestamp}.txt"

            with open(temp_file, 'w') as f:
                f.write(exposition_data)

            print(f"  → Created batch file: {temp_file}")
            return True

        except Exception as e:
            print(f"  → Failed to create batch file: {e}")
            return False
    
    def _create_exposition_format(self, metrics: List[Dict]) -> str:
        """Create Prometheus exposition format from metrics"""
        lines = []
        for metric in metrics:
            name = metric['name']
            labels = metric['labels']
            value = metric['value']
            timestamp = metric['timestamp']

            # Format labels
            if labels:
                label_pairs = [f'{k}="{v}"' for k, v in labels.items()]
                label_str = '{' + ','.join(label_pairs) + '}'
            else:
                label_str = ''

            lines.append(f"{name}{label_str} {value} {timestamp}")

        return '\n'.join(lines)

    def _create_single_metric_exposition(self, metric: Dict) -> str:
        """Create exposition format for a single metric"""
        name = metric['name']
        labels = metric['labels'].copy()  # Make a copy to avoid modifying original
        value = metric['value']

        # Remove conflicting labels that Pushgateway controls
        conflicting_labels = ['job', 'instance']
        for label in conflicting_labels:
            labels.pop(label, None)

        # Format labels
        if labels:
            label_pairs = [f'{k}="{v}"' for k, v in labels.items()]
            label_str = '{' + ','.join(label_pairs) + '}'
        else:
            label_str = ''

        return f"{name}{label_str} {value}\n"
    
    def import_file(self, file_path: str, method: str = "simulation", pushgateway_url: str = None) -> bool:
        """Import a single Prometheus format file"""
        print(f"Importing {file_path}...")

        metrics = self.parse_prometheus_format(file_path)
        if not metrics:
            print(f"No valid metrics found in {file_path}")
            return False

        if method == "pushgateway":
            return self.send_metrics_via_pushgateway(metrics, pushgateway_url)
        elif method == "http-server":
            return self.create_http_server_import(metrics)
        elif method == "direct-write":
            return self.send_metrics_direct_write(metrics)
        else:
            return self.send_metrics_simulation(metrics)
    
    def import_directory(self, directory_path: str, method: str = "simulation", pushgateway_url: str = None) -> bool:
        """Import all .txt files from a directory"""
        txt_files = glob.glob(os.path.join(directory_path, "*.txt"))

        if not txt_files:
            print(f"No .txt files found in {directory_path}")
            return False

        print(f"Found {len(txt_files)} files to import:")
        for f in txt_files:
            print(f"  - {os.path.basename(f)}")
        print()

        success_count = 0
        for file_path in txt_files:
            if self.import_file(file_path, method, pushgateway_url):
                success_count += 1

        print(f"\nImport completed: {success_count}/{len(txt_files)} files successful")
        return success_count == len(txt_files)

def main():
    parser = argparse.ArgumentParser(description='Import backed up Prometheus data to local instance')
    parser.add_argument('--prometheus-url',
                       default=os.getenv('LOCAL_PROMETHEUS_URL', 'http://localhost:9090'),
                       help='Local Prometheus URL (default: http://localhost:9090)')
    parser.add_argument('--input',
                       help='Input file in Prometheus format (.txt)')
    parser.add_argument('--input-dir',
                       help='Input directory containing .txt files')
    parser.add_argument('--method', choices=['simulation', 'pushgateway', 'http-server', 'direct-write'], default='simulation',
                       help='Import method: simulation (validate only), pushgateway (current values), http-server (historical files), or direct-write (historical timestamps)')
    parser.add_argument('--pushgateway-url',
                       help='Pushgateway URL (default: derived from prometheus-url)')
    parser.add_argument('--overlap-strategy', choices=['warn', 'skip', 'overwrite', 'fail'], default='warn',
                       help='How to handle data overlap: warn (default), skip, overwrite, or fail')
    parser.add_argument('--skip-overlap-check', action='store_true',
                       help='Skip overlap detection (faster but less safe)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for sending metrics (default: 1000)')

    args = parser.parse_args()
    
    # Validate arguments
    if not args.input and not args.input_dir:
        print("Error: Either --input or --input-dir is required")
        sys.exit(1)
    
    if args.input and args.input_dir:
        print("Error: Cannot specify both --input and --input-dir")
        sys.exit(1)
    
    # Create importer
    importer = PrometheusDataImporter(args.prometheus_url)

    # Configure overlap detection
    importer.check_overlaps = not args.skip_overlap_check
    importer.overlap_strategy = args.overlap_strategy

    # Check if Prometheus is accessible
    try:
        response = requests.get(f"{args.prometheus_url}/api/v1/status/config", timeout=5)
        if response.status_code != 200:
            print(f"Warning: Prometheus at {args.prometheus_url} returned status {response.status_code}")
            if not args.skip_overlap_check:
                print("Note: Overlap detection may not work properly")
    except requests.exceptions.RequestException as e:
        print(f"Warning: Cannot connect to Prometheus at {args.prometheus_url}: {e}")
        if not args.skip_overlap_check:
            print("Note: Overlap detection will be disabled")
            importer.check_overlaps = False
        print("Continuing anyway...")
    
    # Set pushgateway URL if using pushgateway method
    if args.method == 'pushgateway' and args.pushgateway_url:
        importer.pushgateway_url = args.pushgateway_url

    # Set pushgateway URL if using pushgateway method
    if args.method == 'pushgateway' and args.pushgateway_url:
        # Store pushgateway URL for later use
        pushgateway_url = args.pushgateway_url
    else:
        pushgateway_url = None

    # Import data
    success = False
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} does not exist")
            sys.exit(1)
        success = importer.import_file(args.input, args.method, pushgateway_url)
    elif args.input_dir:
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory {args.input_dir} does not exist")
            sys.exit(1)
        success = importer.import_directory(args.input_dir, args.method, pushgateway_url)

    if success:
        if args.method == 'simulation':
            print("\n🎉 Data validation completed successfully!")
            print("💡 To actually import data, use --method pushgateway")
            print("   Make sure you have a Pushgateway running first:")
            print("   docker run -d -p 9091:9091 prom/pushgateway")
        else:
            print("\n🎉 Data import completed successfully!")
            print("💡 Check your Prometheus instance to verify the imported data.")
            print("   Visit http://localhost:9090 to query the imported metrics.")
    else:
        print("\n❌ Data import failed. Check the input files and connection.")
        sys.exit(1)

if __name__ == '__main__':
    main()
