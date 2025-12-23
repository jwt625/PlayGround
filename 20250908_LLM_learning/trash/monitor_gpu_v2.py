#!/usr/bin/env python3
"""
High-resolution GPU power and temperature monitoring using updated NVML Field Values API.
Uses NVML_FI_DEV_POWER_INSTANT for true transient power (not 1-sec averaged).
Logs metrics at ~10ms resolution to CSV file.

Key improvements over v1:
- Uses nvmlDeviceGetFieldValues() with NVML_FI_DEV_POWER_INSTANT for instant power
- Also logs NVML_FI_DEV_POWER_AVERAGE for comparison (1-sec averaged on H100)
- Uses nvmlDeviceGetTemperature() (still supported, TemperatureV requires more complex setup)
- Batches field queries per GPU for efficiency
"""

import pynvml
import time
import csv
import sys
import signal
from datetime import datetime

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    global running
    print('\nStopping monitoring...')
    running = False

signal.signal(signal.SIGINT, signal_handler)

def get_gpu_metrics(handle):
    """
    Query GPU metrics using Field Values API for power and legacy API for temperature.
    Returns: (power_instant_w, power_avg_w, temp_c)
    """
    try:
        # Define field IDs to query - batch them for efficiency
        field_ids = [
            pynvml.NVML_FI_DEV_POWER_INSTANT,
            pynvml.NVML_FI_DEV_POWER_AVERAGE,
        ]

        # Query field values - Python wrapper handles ctypes conversion
        field_values = pynvml.nvmlDeviceGetFieldValues(handle, field_ids)

        # Extract power values (convert from milliwatts to watts)
        power_instant_w = None
        power_avg_w = None

        for fv in field_values:
            if fv.nvmlReturn == pynvml.NVML_SUCCESS:
                if fv.fieldId == pynvml.NVML_FI_DEV_POWER_INSTANT:
                    # Field value is in milliwatts, convert to watts
                    power_instant_w = fv.value.uiVal / 1000.0
                elif fv.fieldId == pynvml.NVML_FI_DEV_POWER_AVERAGE:
                    power_avg_w = fv.value.uiVal / 1000.0

        # Get temperature (using legacy API - still works fine)
        temp_c = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        return power_instant_w, power_avg_w, temp_c

    except Exception as e:
        return None, None, None

def main():
    # Parse command line arguments
    interval_ms = 10  # default 10ms
    output_file = None  # Will be auto-generated with timestamp

    if len(sys.argv) > 1:
        interval_ms = int(sys.argv[1])
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    # Auto-generate filename with timestamp if not provided
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'gpu_metrics_v2_{timestamp}.csv'
    
    interval_sec = interval_ms / 1000.0
    
    print(f"Starting GPU monitoring (v2 - Field Values API):")
    print(f"  Sampling interval: {interval_ms}ms ({1000/interval_ms:.1f} Hz)")
    print(f"  Output file: {output_file}")
    print(f"  Power: INSTANT + AVERAGE (1-sec) for comparison")
    print(f"  Press Ctrl+C to stop")
    print()
    
    # Initialize NVML
    try:
        pynvml.nvmlInit()
    except Exception as e:
        print(f"Failed to initialize NVML: {e}")
        sys.exit(1)
    
    # Get GPU handles
    try:
        num_gpus = pynvml.nvmlDeviceGetCount()
        print(f"Detected {num_gpus} GPU(s)")
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(num_gpus)]
        
        # Print GPU names
        for i, handle in enumerate(handles):
            name = pynvml.nvmlDeviceGetName(handle)
            print(f"  GPU {i}: {name}")
        print()
    except Exception as e:
        print(f"Failed to get GPU handles: {e}")
        pynvml.nvmlShutdown()
        sys.exit(1)
    
    # Open CSV file and write header
    try:
        f = open(output_file, 'w', newline='')
        writer = csv.writer(f)
        
        # Build header - now includes both instant and averaged power
        header = ['timestamp', 'elapsed_sec']
        for i in range(num_gpus):
            header.extend([
                f'gpu{i}_power_instant_w',
                f'gpu{i}_power_avg_w',
                f'gpu{i}_temp_c'
            ])
        writer.writerow(header)
        f.flush()
        
        # Monitoring loop
        start_time = time.perf_counter()
        sample_count = 0
        
        while running:
            loop_start = time.perf_counter()
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # millisecond precision
            elapsed = loop_start - start_time
            
            row = [timestamp, f'{elapsed:.3f}']
            
            # Query each GPU using Field Values API
            for handle in handles:
                power_instant, power_avg, temp = get_gpu_metrics(handle)
                
                # Format values or N/A if query failed
                row.extend([
                    f'{power_instant:.2f}' if power_instant is not None else 'N/A',
                    f'{power_avg:.2f}' if power_avg is not None else 'N/A',
                    temp if temp is not None else 'N/A'
                ])

            writer.writerow(row)
            f.flush()

            sample_count += 1

            # Print status every 100 samples
            if sample_count % 100 == 0:
                actual_rate = sample_count / elapsed
                print(f"Samples: {sample_count}, Actual rate: {actual_rate:.1f} Hz, Elapsed: {elapsed:.1f}s")

            # Sleep for remaining time to maintain interval
            loop_duration = time.perf_counter() - loop_start
            sleep_time = max(0, interval_sec - loop_duration)
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Cleanup
        f.close()
        pynvml.nvmlShutdown()

        # Final statistics
        total_time = time.perf_counter() - start_time
        actual_rate = sample_count / total_time
        print(f"\nMonitoring stopped:")
        print(f"  Total samples: {sample_count}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Actual sampling rate: {actual_rate:.2f} Hz")
        print(f"  Data saved to: {output_file}")

    except Exception as e:
        print(f"Error during monitoring: {e}")
        f.close()
        pynvml.nvmlShutdown()
        sys.exit(1)

if __name__ == '__main__':
    main()


