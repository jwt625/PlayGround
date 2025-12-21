#!/usr/bin/env python3
"""
High-resolution GPU power and temperature monitoring using pynvml.
Logs metrics at ~10-50ms resolution to CSV file.
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
        output_file = f'gpu_metrics_{timestamp}.csv'
    
    interval_sec = interval_ms / 1000.0
    
    print(f"Starting GPU monitoring:")
    print(f"  Sampling interval: {interval_ms}ms ({1000/interval_ms:.1f} Hz)")
    print(f"  Output file: {output_file}")
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
        
        # Build header
        header = ['timestamp', 'elapsed_sec']
        for i in range(num_gpus):
            header.extend([f'gpu{i}_power_w', f'gpu{i}_temp_c'])
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
            
            # Query each GPU
            for handle in handles:
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    row.extend([f'{power:.2f}', temp])
                except Exception as e:
                    # If query fails, write N/A
                    row.extend(['N/A', 'N/A'])
            
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

