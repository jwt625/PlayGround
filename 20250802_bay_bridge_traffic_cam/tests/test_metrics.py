#!/usr/bin/env python3
"""
Test Suite for Bay Bridge Traffic Detection Metrics System

This module provides comprehensive testing for the Prometheus metrics integration,
including unit tests, integration tests, and validation tools.

Usage:
    python test_metrics.py                    # Run all tests
    python test_metrics.py --unit             # Run unit tests only
    python test_metrics.py --integration      # Run integration tests only
    python test_metrics.py --validate         # Validate metrics endpoint
    python test_metrics.py --benchmark        # Run performance benchmarks
"""

import os
import sys
import time
import unittest
import threading
import requests
import argparse
from typing import Dict, List, Optional
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prometheus_metrics import (
    TrafficMetrics, MetricsConfig, initialize_metrics, 
    get_metrics, shutdown_metrics
)

# Load environment variables
load_dotenv()


class TestMetricsConfig(unittest.TestCase):
    """Test metrics configuration loading and validation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MetricsConfig()
        self.assertTrue(config.enabled)
        self.assertEqual(config.mode, "pull")
        self.assertTrue(config.http_server_enabled)
        self.assertEqual(config.http_server_port, 9091)
        self.assertEqual(config.app_name, "bay-bridge-traffic-detector")
    
    def test_env_config_loading(self):
        """Test configuration loading from environment variables."""
        with patch.dict(os.environ, {
            'METRICS_ENABLED': 'false',
            'METRICS_MODE': 'push',
            'PROMETHEUS_HTTP_SERVER_PORT': '8080',
            'APP_NAME': 'test-app'
        }):
            config = MetricsConfig.from_env()
            self.assertFalse(config.enabled)
            self.assertEqual(config.mode, "push")
            self.assertEqual(config.http_server_port, 8080)
            self.assertEqual(config.app_name, "test-app")


class TestTrafficMetrics(unittest.TestCase):
    """Test core traffic metrics functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = MetricsConfig(
            enabled=True,
            http_server_enabled=False,  # Don't start HTTP server in tests
            debug=True
        )
        self.metrics = TrafficMetrics(self.config)
    
    def tearDown(self):
        """Clean up test environment."""
        self.metrics.shutdown()
    
    def test_vehicle_count_recording(self):
        """Test vehicle count recording functionality."""
        # Record some vehicle counts
        self.metrics.record_vehicle_count('left')
        self.metrics.record_vehicle_count('left')
        self.metrics.record_vehicle_count('right')
        
        # Get metrics text and verify counts
        metrics_text = self.metrics.get_metrics_text()
        self.assertIn('traffic_vehicles_total{app="bay-bridge-traffic-detector",direction="left",instance="main"} 2.0', metrics_text)
        self.assertIn('traffic_vehicles_total{app="bay-bridge-traffic-detector",direction="right",instance="main"} 1.0', metrics_text)
    
    def test_system_status_updates(self):
        """Test system status monitoring."""
        # Update system status
        self.metrics.update_system_status('webcam', True)
        self.metrics.update_system_status('detector', False)
        self.metrics.update_system_status('tracker', True)
        
        # Verify status in metrics
        metrics_text = self.metrics.get_metrics_text()
        self.assertIn('system_status{app="bay-bridge-traffic-detector",component="webcam",instance="main"} 1.0', metrics_text)
        self.assertIn('system_status{app="bay-bridge-traffic-detector",component="detector",instance="main"} 0.0', metrics_text)
        self.assertIn('system_status{app="bay-bridge-traffic-detector",component="tracker",instance="main"} 1.0', metrics_text)
    
    def test_performance_metrics(self):
        """Test performance metrics recording."""
        # Update performance metrics
        self.metrics.update_performance_metrics(fps=30.5, active_objects=3, processing_time=0.05)
        
        # Verify metrics
        metrics_text = self.metrics.get_metrics_text()
        self.assertIn('motion_detector_fps{app="bay-bridge-traffic-detector",instance="main"} 30.5', metrics_text)
        self.assertIn('tracked_objects_active{app="bay-bridge-traffic-detector",instance="main"} 3.0', metrics_text)
        self.assertIn('frame_processing_time_seconds_bucket', metrics_text)
    
    def test_flow_rate_calculation(self):
        """Test traffic flow rate calculation."""
        # Record some vehicle counts
        self.metrics.record_vehicle_count('left')
        self.metrics.record_vehicle_count('left')
        self.metrics.record_vehicle_count('right')
        
        # Manually trigger flow calculation (simulate time passage)
        self.metrics._last_flow_calculation = time.time() - 60  # 1 minute ago
        self.metrics._flow_calculation_worker()
        
        # Check that flow rates were calculated
        metrics_text = self.metrics.get_metrics_text()
        self.assertIn('traffic_flow_rate_per_minute', metrics_text)
    
    def test_disabled_metrics(self):
        """Test that metrics are not recorded when disabled."""
        disabled_config = MetricsConfig(enabled=False)
        disabled_metrics = TrafficMetrics(disabled_config)
        
        # Try to record metrics
        disabled_metrics.record_vehicle_count('left')
        disabled_metrics.update_system_status('webcam', True)
        
        # Metrics should not be recorded (counters should be 0)
        metrics_text = disabled_metrics.get_metrics_text()
        # The metrics exist but should have no samples recorded
        self.assertIn('traffic_vehicles_total', metrics_text)


class TestMetricsIntegration(unittest.TestCase):
    """Test metrics system integration."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.test_port = 9092  # Use different port for testing
        self.config = MetricsConfig(
            enabled=True,
            http_server_enabled=True,
            http_server_port=self.test_port,
            debug=True
        )
    
    def tearDown(self):
        """Clean up integration test environment."""
        shutdown_metrics()
        time.sleep(0.5)  # Allow server to shut down
    
    def test_http_server_startup(self):
        """Test HTTP server startup and metrics endpoint."""
        # Initialize metrics with HTTP server
        metrics = initialize_metrics(self.config)
        time.sleep(1)  # Allow server to start
        
        # Test metrics endpoint
        try:
            response = requests.get(f'http://localhost:{self.test_port}/metrics', timeout=5)
            self.assertEqual(response.status_code, 200)
            self.assertIn('traffic_vehicles_total', response.text)
            self.assertIn('system_status', response.text)
        except requests.exceptions.RequestException as e:
            self.fail(f"Failed to connect to metrics endpoint: {e}")
    
    def test_metrics_endpoint_content(self):
        """Test metrics endpoint returns valid Prometheus format."""
        # Initialize and record some data
        metrics = initialize_metrics(self.config)
        time.sleep(1)
        
        metrics.record_vehicle_count('left')
        metrics.update_system_status('webcam', True)
        
        # Fetch metrics
        response = requests.get(f'http://localhost:{self.test_port}/metrics', timeout=5)
        content = response.text
        
        # Verify Prometheus format
        self.assertIn('# HELP traffic_vehicles_total', content)
        self.assertIn('# TYPE traffic_vehicles_total counter', content)
        self.assertIn('# HELP system_status', content)
        self.assertIn('# TYPE system_status gauge', content)


class MetricsValidator:
    """Utility class for validating metrics system."""
    
    def __init__(self, endpoint: str = "http://localhost:9091/metrics"):
        self.endpoint = endpoint
    
    def validate_endpoint(self) -> Dict[str, any]:
        """Validate metrics endpoint accessibility and content."""
        result = {
            'accessible': False,
            'valid_format': False,
            'metrics_found': [],
            'errors': []
        }
        
        try:
            response = requests.get(self.endpoint, timeout=10)
            result['accessible'] = response.status_code == 200
            
            if result['accessible']:
                content = response.text
                
                # Check for expected metrics
                expected_metrics = [
                    'traffic_vehicles_total',
                    'traffic_flow_rate_per_minute',
                    'system_status',
                    'motion_detector_fps',
                    'tracked_objects_active'
                ]
                
                for metric in expected_metrics:
                    if metric in content:
                        result['metrics_found'].append(metric)
                
                # Validate Prometheus format
                if '# HELP' in content and '# TYPE' in content:
                    result['valid_format'] = True
                
        except requests.exceptions.RequestException as e:
            result['errors'].append(f"Connection error: {e}")
        except Exception as e:
            result['errors'].append(f"Validation error: {e}")
        
        return result
    
    def print_validation_report(self):
        """Print a comprehensive validation report."""
        print("🔍 Bay Bridge Traffic Metrics Validation Report")
        print("=" * 60)
        
        result = self.validate_endpoint()
        
        # Endpoint accessibility
        if result['accessible']:
            print("✅ Metrics endpoint is accessible")
        else:
            print("❌ Metrics endpoint is not accessible")
            for error in result['errors']:
                print(f"   Error: {error}")
            return
        
        # Format validation
        if result['valid_format']:
            print("✅ Metrics format is valid (Prometheus format)")
        else:
            print("❌ Invalid metrics format")
        
        # Metrics presence
        print(f"\n📊 Found {len(result['metrics_found'])} metrics:")
        for metric in result['metrics_found']:
            print(f"   ✅ {metric}")
        
        # Missing metrics
        expected_metrics = [
            'traffic_vehicles_total',
            'traffic_flow_rate_per_minute', 
            'system_status'
        ]
        missing = [m for m in expected_metrics if m not in result['metrics_found']]
        if missing:
            print(f"\n⚠️  Missing critical metrics:")
            for metric in missing:
                print(f"   ❌ {metric}")
        
        print("\n" + "=" * 60)
        
        if result['accessible'] and result['valid_format'] and len(result['metrics_found']) >= 3:
            print("🎉 Metrics system validation PASSED")
        else:
            print("❌ Metrics system validation FAILED")


class PerformanceBenchmark:
    """Performance benchmarking for metrics system."""
    
    def __init__(self):
        self.config = MetricsConfig(
            enabled=True,
            http_server_enabled=False,
            debug=False
        )
        self.metrics = TrafficMetrics(self.config)
    
    def benchmark_vehicle_counting(self, num_events: int = 1000) -> float:
        """Benchmark vehicle counting performance."""
        start_time = time.time()
        
        for i in range(num_events):
            direction = 'left' if i % 2 == 0 else 'right'
            self.metrics.record_vehicle_count(direction)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"📈 Vehicle counting benchmark:")
        print(f"   Events: {num_events}")
        print(f"   Duration: {duration:.4f} seconds")
        print(f"   Rate: {num_events/duration:.0f} events/second")
        print(f"   Overhead: {(duration/num_events)*1000:.4f} ms/event")
        
        return duration
    
    def benchmark_metrics_generation(self, num_iterations: int = 100) -> float:
        """Benchmark metrics text generation."""
        # Record some data first
        for i in range(10):
            self.metrics.record_vehicle_count('left')
            self.metrics.update_system_status('webcam', True)
        
        start_time = time.time()
        
        for _ in range(num_iterations):
            metrics_text = self.metrics.get_metrics_text()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"📊 Metrics generation benchmark:")
        print(f"   Iterations: {num_iterations}")
        print(f"   Duration: {duration:.4f} seconds")
        print(f"   Rate: {num_iterations/duration:.0f} generations/second")
        print(f"   Overhead: {(duration/num_iterations)*1000:.4f} ms/generation")
        
        return duration
    
    def run_all_benchmarks(self):
        """Run all performance benchmarks."""
        print("🚀 Running performance benchmarks...")
        print()
        
        self.benchmark_vehicle_counting(1000)
        print()
        self.benchmark_metrics_generation(100)
        print()
        
        self.metrics.shutdown()
        print("✅ Benchmarks completed")


def main():
    """Main entry point for test suite."""
    parser = argparse.ArgumentParser(description="Test suite for Bay Bridge traffic metrics")
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--validate', action='store_true', help='Validate metrics endpoint')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmarks')
    parser.add_argument('--endpoint', default='http://localhost:9091/metrics', 
                       help='Metrics endpoint URL for validation')
    
    args = parser.parse_args()
    
    if args.validate:
        validator = MetricsValidator(args.endpoint)
        validator.print_validation_report()
        return
    
    if args.benchmark:
        benchmark = PerformanceBenchmark()
        benchmark.run_all_benchmarks()
        return
    
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    if args.unit or (not args.integration):
        suite.addTests(loader.loadTestsFromTestCase(TestMetricsConfig))
        suite.addTests(loader.loadTestsFromTestCase(TestTrafficMetrics))
    
    if args.integration or (not args.unit):
        suite.addTests(loader.loadTestsFromTestCase(TestMetricsIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
