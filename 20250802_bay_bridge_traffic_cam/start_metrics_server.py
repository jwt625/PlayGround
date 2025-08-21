#!/usr/bin/env python3
"""
Standalone Metrics Server for Bay Bridge Traffic Detection System

This script starts a standalone metrics server for testing and development purposes.
It simulates traffic detection events and exposes metrics via HTTP for Prometheus scraping.

Usage:
    python start_metrics_server.py [--port PORT] [--simulate]

Options:
    --port PORT     HTTP server port (default: 9091)
    --simulate      Enable traffic simulation for testing
    --help          Show this help message

The server exposes metrics at http://localhost:PORT/metrics
"""

import os
import sys
import time
import random
import argparse
import threading
import signal
from typing import Optional
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prometheus_metrics import initialize_metrics, get_metrics, shutdown_metrics, MetricsConfig

# Load environment variables
load_dotenv()


class TrafficSimulator:
    """Simulates traffic detection events for testing purposes."""
    
    def __init__(self, metrics_instance):
        self.metrics = metrics_instance
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start traffic simulation."""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._simulation_worker, daemon=True)
        self.thread.start()
        print("🚗 Traffic simulation started")
        
    def stop(self):
        """Stop traffic simulation."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print("🛑 Traffic simulation stopped")
        
    def _simulation_worker(self):
        """Background worker that simulates traffic events."""
        while self.running:
            try:
                # Simulate vehicle detection (random direction)
                direction = random.choice(['left', 'right'])
                self.metrics.record_vehicle_count(direction)
                
                # Simulate system health updates (mostly healthy)
                for component in ['webcam', 'detector', 'tracker']:
                    healthy = random.random() > 0.05  # 95% healthy
                    self.metrics.update_system_status(component, healthy)
                
                # Simulate performance metrics
                fps = random.uniform(25, 35)  # 25-35 FPS
                active_objects = random.randint(0, 8)  # 0-8 tracked objects
                processing_time = random.uniform(0.02, 0.08)  # 20-80ms processing
                
                self.metrics.update_performance_metrics(fps, active_objects, processing_time)
                
                # Random delay between vehicles (1-10 seconds)
                delay = random.uniform(1, 10)
                time.sleep(delay)
                
            except Exception as e:
                print(f"❌ Simulation error: {e}")
                time.sleep(1)


class MetricsServer:
    """Standalone metrics server for testing and development."""
    
    def __init__(self, port: int = 9091, simulate: bool = False):
        self.port = port
        self.simulate = simulate
        self.metrics: Optional = None
        self.simulator: Optional[TrafficSimulator] = None
        self.running = False
        
    def start(self):
        """Start the metrics server."""
        try:
            print(f"🚀 Starting Bay Bridge Traffic Metrics Server")
            print(f"📊 Port: {self.port}")
            print(f"🎭 Simulation: {'Enabled' if self.simulate else 'Disabled'}")
            print()
            
            # Initialize metrics with custom port
            config = MetricsConfig.from_env()
            config.http_server_port = self.port
            config.debug = True
            
            self.metrics = initialize_metrics(config)
            
            # Start traffic simulation if requested
            if self.simulate:
                self.simulator = TrafficSimulator(self.metrics)
                self.simulator.start()
            
            self.running = True
            
            print(f"✅ Metrics server started successfully!")
            print(f"📈 Metrics endpoint: http://localhost:{self.port}/metrics")
            print(f"🔍 Prometheus config: http://localhost:9090 (if running)")
            print()
            print("📋 Available metrics:")
            print("  - traffic_vehicles_total (counter)")
            print("  - traffic_flow_rate_per_minute (gauge)")
            print("  - system_status (gauge)")
            print("  - motion_detector_fps (gauge)")
            print("  - tracked_objects_active (gauge)")
            print("  - frame_processing_time_seconds (histogram)")
            print()
            
            if self.simulate:
                print("🎯 Simulation running - generating random traffic events")
                print("   Use Ctrl+C to stop")
            else:
                print("💡 To test with real data, run: python main.py")
                print("   Use Ctrl+C to stop server")
            
            # Keep server running
            self._wait_for_shutdown()
            
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested by user")
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            sys.exit(1)
        finally:
            self.stop()
    
    def stop(self):
        """Stop the metrics server."""
        if not self.running:
            return
            
        print("\n🔄 Shutting down metrics server...")
        
        if self.simulator:
            self.simulator.stop()
            
        shutdown_metrics()
        self.running = False
        
        print("✅ Metrics server stopped")
    
    def _wait_for_shutdown(self):
        """Wait for shutdown signal."""
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


def setup_signal_handlers(server: MetricsServer):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        print(f"\n📡 Received signal {signum}")
        server.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def print_metrics_sample():
    """Print a sample of what metrics look like."""
    print("📄 Sample metrics output:")
    print("=" * 50)
    sample_metrics = """# HELP traffic_vehicles_total Total vehicles detected crossing counting lines
# TYPE traffic_vehicles_total counter
traffic_vehicles_total{app="bay-bridge-traffic-detector",direction="left",instance="main"} 5.0
traffic_vehicles_total{app="bay-bridge-traffic-detector",direction="right",instance="main"} 3.0

# HELP traffic_flow_rate_per_minute Vehicles per minute by direction
# TYPE traffic_flow_rate_per_minute gauge
traffic_flow_rate_per_minute{app="bay-bridge-traffic-detector",direction="left",instance="main"} 2.5
traffic_flow_rate_per_minute{app="bay-bridge-traffic-detector",direction="right",instance="main"} 1.8

# HELP system_status System component health status (1=healthy, 0=error)
# TYPE system_status gauge
system_status{app="bay-bridge-traffic-detector",component="webcam",instance="main"} 1.0
system_status{app="bay-bridge-traffic-detector",component="detector",instance="main"} 1.0
system_status{app="bay-bridge-traffic-detector",component="tracker",instance="main"} 1.0"""
    print(sample_metrics)
    print("=" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Standalone metrics server for Bay Bridge traffic detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_metrics_server.py                    # Start server on default port
  python start_metrics_server.py --port 8080        # Start server on port 8080
  python start_metrics_server.py --simulate         # Start with traffic simulation
  python start_metrics_server.py --sample           # Show sample metrics output

The server will expose metrics at http://localhost:PORT/metrics
Configure Prometheus to scrape this endpoint for monitoring.
        """
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=9091,
        help='HTTP server port (default: 9091)'
    )
    
    parser.add_argument(
        '--simulate',
        action='store_true',
        help='Enable traffic simulation for testing'
    )
    
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Show sample metrics output and exit'
    )
    
    args = parser.parse_args()
    
    if args.sample:
        print_metrics_sample()
        return
    
    # Validate port
    if not (1024 <= args.port <= 65535):
        print(f"❌ Invalid port {args.port}. Must be between 1024 and 65535.")
        sys.exit(1)
    
    # Create and start server
    server = MetricsServer(port=args.port, simulate=args.simulate)
    setup_signal_handlers(server)
    server.start()


if __name__ == '__main__':
    main()
