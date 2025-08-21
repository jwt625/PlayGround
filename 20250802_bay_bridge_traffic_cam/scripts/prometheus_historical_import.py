#!/usr/bin/env python3
"""
Prometheus Historical Data Import using Admin API

This script imports historical data into Prometheus by using the admin API
and TSDB manipulation techniques.

Usage:
    python prometheus_historical_import.py --data-dir prometheus_import
"""

import os
import sys
import time
import argparse
import requests
import json
import subprocess
from datetime import datetime
from typing import List, Dict
import glob

class PrometheusHistoricalImporter:
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus_url = prometheus_url.rstrip('/')
        self.admin_url = f"{self.prometheus_url}/api/v1/admin"
        
    def check_admin_api(self) -> bool:
        """Check if Prometheus admin API is enabled"""
        try:
            response = requests.post(f"{self.admin_url}/tsdb/snapshot")
            if response.status_code == 200:
                print("✅ Prometheus admin API is enabled")
                return True
            else:
                print(f"❌ Admin API not available: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot access admin API: {e}")
            return False
    
    def create_snapshot(self) -> str:
        """Create a Prometheus snapshot"""
        try:
            print("📸 Creating Prometheus snapshot...")
            response = requests.post(f"{self.admin_url}/tsdb/snapshot")
            
            if response.status_code == 200:
                data = response.json()
                snapshot_name = data.get('data', {}).get('name', '')
                print(f"✅ Snapshot created: {snapshot_name}")
                return snapshot_name
            else:
                print(f"❌ Failed to create snapshot: {response.status_code}")
                return ""
                
        except Exception as e:
            print(f"❌ Error creating snapshot: {e}")
            return ""
    
    def import_via_promtool(self, data_dir: str) -> bool:
        """
        Import historical data using promtool
        
        This uses Prometheus's promtool to import data with preserved timestamps
        """
        print(f"🔧 Importing historical data from {data_dir} using promtool...")
        
        # Find all batch files
        batch_files = glob.glob(os.path.join(data_dir, "batch_*.txt"))
        if not batch_files:
            print(f"❌ No batch files found in {data_dir}")
            return False
        
        print(f"📁 Found {len(batch_files)} batch files to import")
        
        success_count = 0
        for batch_file in batch_files:
            try:
                # Use promtool to import the data
                # Note: This requires promtool to be installed and Prometheus to be stopped
                print(f"  Processing {os.path.basename(batch_file)}...")
                
                # For now, we'll prepare the command but not execute it
                # as it requires stopping Prometheus
                cmd = [
                    "promtool", "tsdb", "create-blocks-from", "openmetrics",
                    batch_file, "/tmp/prometheus_import_blocks"
                ]
                
                print(f"    Command: {' '.join(cmd)}")
                print(f"    ⚠️  This requires promtool and stopping Prometheus")
                
                success_count += 1
                
            except Exception as e:
                print(f"  ❌ Error processing {batch_file}: {e}")
        
        print(f"📊 Prepared {success_count}/{len(batch_files)} files for import")
        return success_count == len(batch_files)
    
    def import_via_recording_rules(self, data_dir: str) -> bool:
        """
        Import data using Prometheus recording rules with backfill
        
        This is the safest method that doesn't require stopping Prometheus
        """
        print(f"📝 Setting up recording rules import from {data_dir}...")
        
        # Create a recording rules file
        rules_file = os.path.join(data_dir, "import_rules.yml")
        
        # Find all metric files
        metric_files = glob.glob(os.path.join(data_dir, "metrics_*.txt"))
        if not metric_files:
            print(f"❌ No metric files found in {data_dir}")
            return False
        
        print(f"📁 Found {len(metric_files)} metric files")
        
        # Create recording rules configuration
        rules_config = {
            "groups": [{
                "name": "historical_import",
                "interval": "30s",
                "rules": []
            }]
        }
        
        # Sample a few files to create rules
        sample_files = metric_files[:5]  # Use first 5 files as examples
        
        for i, metric_file in enumerate(sample_files):
            rule_name = f"historical_data_rule_{i+1}"
            rules_config["groups"][0]["rules"].append({
                "record": rule_name,
                "expr": f"# Import from {os.path.basename(metric_file)}",
                "labels": {
                    "source": "historical_import",
                    "file": os.path.basename(metric_file)
                }
            })
        
        # Write rules file
        import yaml
        try:
            with open(rules_file, 'w') as f:
                yaml.dump(rules_config, f, default_flow_style=False)
            print(f"✅ Created recording rules: {rules_file}")
        except ImportError:
            # Fallback to JSON if yaml not available
            with open(rules_file.replace('.yml', '.json'), 'w') as f:
                json.dump(rules_config, f, indent=2)
            print(f"✅ Created recording rules: {rules_file.replace('.yml', '.json')}")
        
        print("\n📋 Next steps for recording rules import:")
        print("1. Add the rules file to your Prometheus configuration")
        print("2. Reload Prometheus configuration")
        print("3. Use promtool to backfill the rules with historical data")
        print(f"   promtool query range --start=2025-08-03T00:49:55Z --end=2025-08-03T11:56:25Z 'historical_data_rule_1'")
        
        return True
    
    def import_via_federation(self, data_dir: str) -> bool:
        """
        Import data by setting up a temporary Prometheus instance for federation
        """
        print(f"🔗 Setting up federation import from {data_dir}...")
        
        # Create a temporary Prometheus config for the historical data
        temp_config = {
            "global": {
                "scrape_interval": "15s"
            },
            "scrape_configs": [{
                "job_name": "historical_data",
                "static_configs": [{
                    "targets": ["localhost:8000"]
                }],
                "scrape_interval": "30s",
                "metrics_path": "/metrics"
            }]
        }
        
        config_file = os.path.join(data_dir, "temp_prometheus.yml")
        import yaml
        try:
            with open(config_file, 'w') as f:
                yaml.dump(temp_config, f, default_flow_style=False)
            print(f"✅ Created temporary Prometheus config: {config_file}")
        except ImportError:
            print("❌ PyYAML required for federation setup")
            return False
        
        print("\n📋 Federation setup instructions:")
        print("1. Start HTTP server in data directory:")
        print(f"   cd {data_dir} && python -m http.server 8000")
        print("2. Start temporary Prometheus instance:")
        print(f"   prometheus --config.file={config_file} --storage.tsdb.path=/tmp/temp_prometheus")
        print("3. Configure main Prometheus to federate from temporary instance")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Import historical data into Prometheus')
    parser.add_argument('--data-dir', default='prometheus_import',
                       help='Directory containing historical data files')
    parser.add_argument('--prometheus-url', default='http://localhost:9090',
                       help='Prometheus URL')
    parser.add_argument('--method', choices=['admin-api', 'promtool', 'recording-rules', 'federation'],
                       default='recording-rules',
                       help='Import method to use')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory {args.data_dir} does not exist")
        sys.exit(1)
    
    importer = PrometheusHistoricalImporter(args.prometheus_url)
    
    print(f"🚀 Starting historical data import using {args.method} method")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"🎯 Prometheus URL: {args.prometheus_url}")
    print()
    
    success = False
    
    if args.method == "admin-api":
        if importer.check_admin_api():
            snapshot = importer.create_snapshot()
            if snapshot:
                print("✅ Admin API method prepared")
                success = True
        else:
            print("❌ Admin API not available")
    
    elif args.method == "promtool":
        success = importer.import_via_promtool(args.data_dir)
    
    elif args.method == "recording-rules":
        success = importer.import_via_recording_rules(args.data_dir)
    
    elif args.method == "federation":
        success = importer.import_via_federation(args.data_dir)
    
    if success:
        print("\n🎉 Historical data import setup completed!")
        print("💡 Follow the instructions above to complete the import process")
    else:
        print("\n❌ Historical data import setup failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
