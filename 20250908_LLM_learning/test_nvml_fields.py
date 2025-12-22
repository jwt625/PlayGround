#!/usr/bin/env python3
"""
Minimal test to verify NVML Field Values API access.
Tests both legacy and new APIs to see what works.
"""

import pynvml
import sys

def test_legacy_api(handle, gpu_id):
    """Test the old/legacy API that we know works."""
    print(f"\n=== GPU {gpu_id}: Testing Legacy API ===")
    try:
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        power_w = power_mw / 1000.0
        print(f"  Power (legacy): {power_w:.2f} W")
    except Exception as e:
        print(f"  Power (legacy) FAILED: {e}")

    try:
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        print(f"  Temperature (legacy): {temp} C")
    except Exception as e:
        print(f"  Temperature (legacy) FAILED: {e}")

def test_temperature_api(handle, gpu_id):
    """Test temperature APIs - both legacy and new."""
    print(f"\n=== GPU {gpu_id}: Testing Temperature APIs ===")

    # Test legacy nvmlDeviceGetTemperature
    try:
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        print(f"  nvmlDeviceGetTemperature (deprecated): {temp} C")
    except Exception as e:
        print(f"  nvmlDeviceGetTemperature FAILED: {e}")

    # Check if nvmlDeviceGetTemperatureV exists
    if hasattr(pynvml, 'nvmlDeviceGetTemperatureV'):
        print(f"  nvmlDeviceGetTemperatureV: EXISTS")
        try:
            # Try to call it - need to figure out the signature
            import inspect
            sig = inspect.signature(pynvml.nvmlDeviceGetTemperatureV)
            print(f"    Signature: {sig}")

            # Try calling it
            temp_v = pynvml.nvmlDeviceGetTemperatureV(handle, pynvml.NVML_TEMPERATURE_GPU)
            print(f"    Temperature (V): {temp_v} C")
        except Exception as e:
            print(f"    nvmlDeviceGetTemperatureV call FAILED: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  nvmlDeviceGetTemperatureV: NOT FOUND")

    # Try temperature via Field Values API
    print(f"\n  Trying temperature via Field Values API:")
    try:
        # Check if memory temp field exists
        if hasattr(pynvml, 'NVML_FI_DEV_MEMORY_TEMP'):
            field_ids = [pynvml.NVML_FI_DEV_MEMORY_TEMP]
            print(f"    NVML_FI_DEV_MEMORY_TEMP = {pynvml.NVML_FI_DEV_MEMORY_TEMP}")

            results = pynvml.nvmlDeviceGetFieldValues(handle, field_ids)
            for result in results:
                if result.nvmlReturn == pynvml.NVML_SUCCESS:
                    temp = result.value.uiVal
                    print(f"    Memory Temperature: {temp} C")
                else:
                    print(f"    Memory temp query FAILED: {result.nvmlReturn}")
        else:
            print(f"    NVML_FI_DEV_MEMORY_TEMP: NOT FOUND")
    except Exception as e:
        print(f"    Field Values temperature FAILED: {e}")

def test_field_values_api(handle, gpu_id):
    """Test the Field Values API for instant/average power."""
    print(f"\n=== GPU {gpu_id}: Testing Field Values API ===")

    # Try using the Python wrapper for nvmlDeviceGetFieldValues
    # Signature: nvmlDeviceGetFieldValues(handle, fieldIds)
    try:
        # Create list of field IDs to query
        field_ids = [
            pynvml.NVML_FI_DEV_POWER_INSTANT,
            pynvml.NVML_FI_DEV_POWER_AVERAGE,
        ]

        print(f"  Querying field IDs:")
        print(f"    NVML_FI_DEV_POWER_INSTANT = {pynvml.NVML_FI_DEV_POWER_INSTANT}")
        print(f"    NVML_FI_DEV_POWER_AVERAGE = {pynvml.NVML_FI_DEV_POWER_AVERAGE}")

        # Call the function - Python wrapper handles the ctypes conversion
        results = pynvml.nvmlDeviceGetFieldValues(handle, field_ids)
        print(f"  nvmlDeviceGetFieldValues returned: {type(results)}")
        print(f"  Number of results: {len(results)}")

        # Parse results
        for i, result in enumerate(results):
            field_name = "INSTANT" if result.fieldId == pynvml.NVML_FI_DEV_POWER_INSTANT else "AVERAGE"
            print(f"\n  Result {i} - Power {field_name}:")
            print(f"    fieldId: {result.fieldId}")
            print(f"    nvmlReturn: {result.nvmlReturn} ({'SUCCESS' if result.nvmlReturn == 0 else 'FAILED'})")
            print(f"    timestamp: {result.timestamp}")
            print(f"    latencyUsec: {result.latencyUsec}")
            print(f"    valueType: {result.valueType}")

            if result.nvmlReturn == pynvml.NVML_SUCCESS:
                # valueType 1 = unsigned int
                # The value is a union, try to access uiVal
                try:
                    power_mw = result.value.uiVal
                    power_w = power_mw / 1000.0
                    print(f"    Power: {power_mw} mW = {power_w:.2f} W")
                except Exception as e:
                    print(f"    Failed to extract value: {e}")

    except Exception as e:
        print(f"  Field Values API test FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Try alternative: check for total energy consumption
    print(f"\n  Alternative power APIs:")
    try:
        total_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        print(f"    nvmlDeviceGetTotalEnergyConsumption: {total_energy} mJ")
    except AttributeError:
        print(f"    nvmlDeviceGetTotalEnergyConsumption: NOT AVAILABLE")
    except Exception as e:
        print(f"    nvmlDeviceGetTotalEnergyConsumption: ERROR - {e}")

def main():
    print("Initializing NVML...")
    try:
        pynvml.nvmlInit()
        print("NVML initialized successfully")
    except Exception as e:
        print(f"Failed to initialize NVML: {e}")
        sys.exit(1)
    
    try:
        num_gpus = pynvml.nvmlDeviceGetCount()
        print(f"\nDetected {num_gpus} GPU(s)")
        
        for i in range(num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            print(f"\nGPU {i}: {name}")
            
            # Test all APIs
            test_legacy_api(handle, i)
            test_temperature_api(handle, i)
            test_field_values_api(handle, i)
        
        pynvml.nvmlShutdown()
        print("\n\nNVML shutdown successfully")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        pynvml.nvmlShutdown()
        sys.exit(1)

if __name__ == '__main__':
    main()

