#!/usr/bin/env python3
"""
Test Counter Persistence for Bay Bridge Traffic Detection System

This script tests the counter persistence functionality to ensure
traffic_vehicles_total counters are restored after application restarts.

Usage:
    python test_counter_persistence.py
"""

import os
import time
import json
from prometheus_metrics import initialize_metrics, shutdown_metrics, MetricsConfig


def test_counter_persistence():
    """Test that traffic counters persist across application restarts."""
    print("🔄 TESTING COUNTER PERSISTENCE")
    print("=" * 60)
    
    state_file = "test_traffic_metrics_state.json"
    
    # Clean up any existing test state file
    if os.path.exists(state_file):
        os.remove(state_file)
        print("🧹 Cleaned up existing test state file")
    
    print("\n📊 Phase 1: Create initial metrics and record vehicles")
    
    # Create first metrics instance
    config1 = MetricsConfig.from_env()
    config1.enabled = True
    config1.http_server_enabled = False  # Don't start HTTP server for test
    config1.persist_state = True
    config1.state_file = state_file
    config1.debug = True
    
    metrics1 = initialize_metrics(config1)
    
    # Record some vehicle counts
    print("Recording vehicles: 5 left, 3 right")
    for i in range(5):
        metrics1.record_vehicle_count('left')
    for i in range(3):
        metrics1.record_vehicle_count('right')
    
    # Check initial values
    metrics_text1 = metrics1.get_metrics_text()
    print("\nInitial counter values:")
    for line in metrics_text1.split('\n'):
        if 'traffic_vehicles_total' in line and not line.startswith('#'):
            print(f"  {line}")
    
    # Verify state file was created
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
        print(f"\n✅ State file created: {state['counters']}")
    else:
        print("\n❌ State file was not created!")
        return False
    
    # Shutdown first instance
    print("\n🔄 Phase 2: Shutdown and restart")
    shutdown_metrics()
    print("First instance shut down")
    
    # Wait a moment
    time.sleep(1)
    
    # Create second metrics instance (simulating restart)
    print("Creating new metrics instance...")
    config2 = MetricsConfig.from_env()
    config2.enabled = True
    config2.http_server_enabled = False
    config2.persist_state = True
    config2.state_file = state_file
    config2.debug = True
    
    metrics2 = initialize_metrics(config2)
    
    # Check restored values
    metrics_text2 = metrics2.get_metrics_text()
    print("\nRestored counter values:")
    restored_left = 0
    restored_right = 0
    for line in metrics_text2.split('\n'):
        if 'traffic_vehicles_total' in line and not line.startswith('#'):
            print(f"  {line}")
            if 'direction="left"' in line:
                restored_left = float(line.split()[-1])
            elif 'direction="right"' in line:
                restored_right = float(line.split()[-1])
    
    # Verify restoration
    success = True
    if restored_left == 5.0:
        print("✅ Left counter correctly restored")
    else:
        print(f"❌ Left counter incorrect: expected 5.0, got {restored_left}")
        success = False
    
    if restored_right == 3.0:
        print("✅ Right counter correctly restored")
    else:
        print(f"❌ Right counter incorrect: expected 3.0, got {restored_right}")
        success = False
    
    print("\n📊 Phase 3: Add more vehicles and verify continuity")
    
    # Add more vehicles
    print("Adding 2 more left vehicles...")
    for i in range(2):
        metrics2.record_vehicle_count('left')
    
    # Check final values
    metrics_text3 = metrics2.get_metrics_text()
    print("\nFinal counter values:")
    final_left = 0
    for line in metrics_text3.split('\n'):
        if 'traffic_vehicles_total' in line and not line.startswith('#'):
            print(f"  {line}")
            if 'direction="left"' in line:
                final_left = float(line.split()[-1])
    
    if final_left == 7.0:
        print("✅ Counter continuity verified (5 + 2 = 7)")
    else:
        print(f"❌ Counter continuity failed: expected 7.0, got {final_left}")
        success = False
    
    # Cleanup
    shutdown_metrics()
    if os.path.exists(state_file):
        os.remove(state_file)
        print("\n🧹 Test state file cleaned up")
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 COUNTER PERSISTENCE TEST PASSED!")
        print("✅ Counters are successfully restored after restart")
    else:
        print("❌ COUNTER PERSISTENCE TEST FAILED!")
        print("⚠️  Counters are NOT properly restored")
    print("=" * 60)
    
    return success


def test_state_file_validation():
    """Test state file validation (age, app name, etc.)."""
    print("\n🔍 TESTING STATE FILE VALIDATION")
    print("=" * 40)
    
    state_file = "test_invalid_state.json"
    
    # Test 1: Old state file
    print("Test 1: Old state file (should be ignored)")
    old_state = {
        'timestamp': time.time() - (25 * 3600),  # 25 hours ago
        'app_name': 'bay-bridge-traffic-detector',
        'app_instance': 'main',
        'counters': {'left': 100, 'right': 50}
    }
    
    with open(state_file, 'w') as f:
        json.dump(old_state, f)
    
    config = MetricsConfig.from_env()
    config.enabled = True
    config.http_server_enabled = False
    config.persist_state = True
    config.state_file = state_file
    
    metrics = initialize_metrics(config)
    
    # Check that counters are 0 (old state ignored)
    metrics_text = metrics.get_metrics_text()
    left_value = 0
    for line in metrics_text.split('\n'):
        if 'traffic_vehicles_total' in line and 'direction="left"' in line and not line.startswith('#'):
            left_value = float(line.split()[-1])
    
    if left_value == 0:
        print("✅ Old state file correctly ignored")
    else:
        print(f"❌ Old state file was used: {left_value}")
    
    shutdown_metrics()
    os.remove(state_file)
    
    print("State file validation tests completed")


if __name__ == '__main__':
    success = test_counter_persistence()
    test_state_file_validation()
    
    if success:
        print("\n🚀 Counter persistence is working correctly!")
        print("Your traffic counters will now survive application restarts.")
    else:
        print("\n⚠️  Counter persistence needs debugging.")
