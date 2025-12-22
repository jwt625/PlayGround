# DevLog-003: GPU Metrics Monitoring - NVML API Update

**Date**: 2025-12-22
**Author**: Wentao
**Status**: Completed - C Implementation Recommended

## Overview

Investigated and validated NVML Field Values API for accurate instant power measurements on H100 GPUs. The legacy `nvmlDeviceGetPowerUsage()` returns 1-second averaged power on Ampere and newer architectures, which is insufficient for capturing transient power behavior during training.

**Key Finding:** PyNVML does not expose `nvmlDeviceGetTemperatureV()`, the recommended non-deprecated temperature API. C implementation required for complete NVML API access.

## Background

### Problem Statement

The original monitoring script (`monitor_gpu.py`) used legacy NVML APIs:
- `nvmlDeviceGetPowerUsage()` - Returns 1-second averaged power on H100
- `nvmlDeviceGetTemperature()` - Deprecated in NVML C API

For high-resolution monitoring at 100 Hz (10ms intervals), the 1-second averaging obscures transient power spikes and variations that occur during model training.

### NVML Documentation (H100 Behavior)

From NVIDIA NVML documentation:

**Power Usage Getter:**
> "On Ampere (except GA100) or newer GPUs, the API returns power averaged over 1 sec interval."

**Recommended Power Fields:**
- `NVML_FI_DEV_POWER_INSTANT` - Current GPU power (true instantaneous reading)
- `NVML_FI_DEV_POWER_AVERAGE` - Averaged over 1 sec interval (Ampere or newer)

**Temperature Getter:**
> "Deprecated - Use nvmlDeviceGetTemperatureV instead."

However, `nvmlDeviceGetTemperatureV` is not exposed in PyNVML bindings.

## Investigation Process

### API Availability Testing

Created `test_nvml_fields.py` to verify NVML Field Values API availability in PyNVML:

**Key Findings:**
1. Field constants are accessible:
   - `NVML_FI_DEV_POWER_INSTANT = 186`
   - `NVML_FI_DEV_POWER_AVERAGE = 185`
   - `NVML_FI_DEV_MEMORY_TEMP = 82`

2. Python wrapper signature:
   ```python
   nvmlDeviceGetFieldValues(handle, fieldIds)
   ```
   - Takes device handle and list of field IDs
   - Returns array of `c_nvmlFieldValue_t` structures

3. Field value structure:
   ```python
   c_nvmlFieldValue_t(
       fieldId: int,
       scopeId: int,
       timestamp: int,
       latencyUsec: int,
       valueType: int,  # 1 = unsigned int
       nvmlReturn: int,  # 0 = NVML_SUCCESS
       value: c_nvmlValue_t  # Union type
   )
   ```

4. Power values are in milliwatts, accessed via `value.uiVal`

### Measured Power Comparison

Example readings from H100 GPU during active training:

| Metric | GPU 0 | GPU 1 |
|--------|-------|-------|
| Power INSTANT | 615.79 W | 648.69 W |
| Power AVERAGE | 690.48 W | 685.56 W |
| Legacy API | 690.48 W | 685.56 W |

**Observations:**
- Legacy API matches AVERAGE field (confirms 1-sec averaging)
- INSTANT shows ~75W difference from AVERAGE (transient variation)
- This validates the need for instant power readings

### Temperature API Status

**Tested APIs:**
- `nvmlDeviceGetTemperature()` - Works (69°C GPU, despite deprecation notice)
- `nvmlDeviceGetTemperatureV()` - Not available in PyNVML
- `NVML_FI_DEV_MEMORY_TEMP` - Works (64°C memory via Field Values API)

**Decision:** Continue using `nvmlDeviceGetTemperature()` for GPU temperature since the V2 API is not exposed in Python bindings.

## Implementation

### New Script: `monitor_gpu_v2.py`

**Key Changes:**

1. **Power Measurement** - Field Values API:
   ```python
   field_ids = [
       pynvml.NVML_FI_DEV_POWER_INSTANT,
       pynvml.NVML_FI_DEV_POWER_AVERAGE,
   ]
   field_values = pynvml.nvmlDeviceGetFieldValues(handle, field_ids)
   
   for fv in field_values:
       if fv.nvmlReturn == pynvml.NVML_SUCCESS:
           if fv.fieldId == pynvml.NVML_FI_DEV_POWER_INSTANT:
               power_instant_w = fv.value.uiVal / 1000.0
           elif fv.fieldId == pynvml.NVML_FI_DEV_POWER_AVERAGE:
               power_avg_w = fv.value.uiVal / 1000.0
   ```

2. **Batched Queries** - Single API call per GPU for both power fields (reduces overhead)

3. **Enhanced CSV Output** - Three columns per GPU:
   - `gpu{i}_power_instant_w` - Instantaneous power
   - `gpu{i}_power_avg_w` - 1-second averaged power
   - `gpu{i}_temp_c` - GPU temperature

4. **Temperature** - Retained legacy API (still functional):
   ```python
   temp_c = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
   ```

### Usage

```bash
# Default 10ms sampling
./monitor_gpu_v2.py

# Custom interval (50ms)
./monitor_gpu_v2.py 50

# Custom interval and output file
./monitor_gpu_v2.py 10 my_metrics.csv
```

Output file naming: `gpu_metrics_v2_YYYYMMDD_HHMMSS.csv`

## Validation Script

### `test_nvml_fields.py`

Comprehensive test script that validates:
1. Legacy API functionality (baseline)
2. Temperature API availability and alternatives
3. Field Values API for instant/averaged power
4. Memory temperature via Field Values API
5. Alternative power APIs (total energy consumption)

**Test Output Structure:**
- Legacy API readings (power, temperature)
- Temperature API exploration (deprecated vs V2 vs field values)
- Field Values API detailed results (timestamps, latency, value extraction)
- Alternative power metrics

This script serves as reference implementation for correct NVML Field Values API usage in PyNVML.

## Results

### Visualization Update

Updated `visualize_gpu_metrics_v2.py`:
- Changed `downsample_factor` from 100 to 1 (no point skipping)
- All data points now included in visualization
- Enables accurate analysis of transient power behavior

### Benefits

1. **True Transient Capture** - 10ms instant power readings vs 1-sec averaged
2. **Comparison Channel** - Both instant and averaged power logged for validation
3. **Reduced Overhead** - Batched field queries minimize API call overhead
4. **Future-Proof** - Uses recommended NVML Field Values API

## References

### NVML Documentation
- Power usage getter behavior on Ampere/H100 architectures
- Field Values API for explicit power semantics selection
- Temperature API deprecation notice

### Related Files
- `monitor_gpu.py` - Original implementation (legacy APIs)
- `monitor_gpu_v2.py` - Updated implementation (Field Values API, Python)
- `monitor_gpu_v2.c` - **Production C implementation (recommended)**
- `test_nvml_fields.py` - API validation and reference implementation (Python)
- `test_nvml_fields.c` - C implementation validating full NVML API access
- `Makefile` - Build configuration for C programs
- `visualize_gpu_metrics_v2.py` - Visualization script (updated for full resolution)

## C Implementation Investigation

### PyNVML API Limitation Discovered

After validating the Python implementation, a critical API gap was identified:

**Problem:** `nvmlDeviceGetTemperature()` is deprecated, but the recommended replacement `nvmlDeviceGetTemperatureV()` is **NOT exposed in PyNVML bindings**.

**Evidence:**
- Python test shows: `nvmlDeviceGetTemperatureV: NOT FOUND`
- No GPU core temperature field exists in Field Values API
- Only available fields: `NVML_FI_DEV_MEMORY_TEMP` (82) and temperature limit thresholds
- No `NVML_FI_DEV_GPU_TEMP` or equivalent

### C Implementation Created

Created `test_nvml_fields.c` - equivalent C implementation to verify NVML C API availability.

**Key Findings:**

1. **Temperature API - WORKS in C:**
   ```c
   nvmlTemperature_v1_t temp_v1;
   temp_v1.version = nvmlTemperature_v1;
   temp_v1.sensorType = NVML_TEMPERATURE_GPU;
   nvmlDeviceGetTemperatureV(device, (nvmlTemperature_t*)&temp_v1);
   // Returns: temp_v1.temperature = 69°C (actual reading)
   ```
   - `nvmlDeviceGetTemperatureV()` is fully functional in C API
   - Returns GPU core temperature correctly
   - This is the recommended, non-deprecated API

2. **Field Values API - WORKS in C:**
   ```c
   nvmlFieldValue_t field_values[2] = {0};
   field_values[0].fieldId = NVML_FI_DEV_POWER_INSTANT;
   field_values[0].scopeId = 0;
   field_values[1].fieldId = NVML_FI_DEV_POWER_AVERAGE;
   field_values[1].scopeId = 0;
   nvmlDeviceGetFieldValues(device, 2, field_values);
   ```
   - Proper initialization (zero-init + scopeId) required
   - Returns instant and averaged power correctly
   - Memory temperature via `NVML_FI_DEV_MEMORY_TEMP` also works

3. **Test Results (H100 GPUs):**
   - GPU 0: Temp 69°C, Power Instant 672.95W, Power Average 689.81W
   - GPU 1: Temp 63°C, Power Instant 774.05W, Power Average 688.80W
   - All APIs functional and returning valid data

### Recommendation Update

**Switch to C implementation** for the following reasons:

1. **API Completeness:** Access to `nvmlDeviceGetTemperatureV()` - the proper, non-deprecated temperature API
2. **Future-Proof:** Using recommended APIs instead of deprecated ones
3. **Performance:** Lower overhead for 100+ Hz polling, no GIL contention
4. **Correctness:** Avoid relying on deprecated APIs that may be removed

The Python implementation successfully validated the approach, but the C implementation provides access to the complete, non-deprecated NVML API surface.

### C Monitor Implementation - COMPLETED

Created `monitor_gpu_v2.c` - production-ready C implementation with full API support.

**Implementation Details:**

1. **Temperature API:**
   ```c
   nvmlTemperature_v1_t temp_v1;
   temp_v1.version = nvmlTemperature_v1;
   temp_v1.sensorType = NVML_TEMPERATURE_GPU;
   nvmlDeviceGetTemperatureV(device, (nvmlTemperature_t*)&temp_v1);
   // Returns: temp_v1.temperature (non-deprecated API)
   ```

2. **Power API (Field Values):**
   ```c
   nvmlFieldValue_t field_values[2] = {0};
   field_values[0].fieldId = NVML_FI_DEV_POWER_INSTANT;
   field_values[0].scopeId = 0;
   field_values[1].fieldId = NVML_FI_DEV_POWER_AVERAGE;
   field_values[1].scopeId = 0;
   nvmlDeviceGetFieldValues(device, 2, field_values);
   ```

3. **CSV Output Format:**
   - Identical to Python v2: `timestamp,elapsed_sec,gpu0_power_instant_w,gpu0_power_avg_w,gpu0_temp_c,...`
   - Compatible with existing visualization tools (`visualize_gpu_metrics_v2.py`)

4. **Performance Validation:**
   - Tested at 10ms intervals (100 Hz)
   - Achieved 99-100 Hz actual sampling rate
   - Low overhead, no GIL contention
   - Suitable for sustained high-frequency monitoring

**Usage:**
```bash
# Compile
make monitor_gpu_v2

# Run with 10ms interval (auto-generated filename)
./monitor_gpu_v2 10

# Run with custom interval and filename
./monitor_gpu_v2 10 gpu_metrics_output.csv

# Run in tmux session for long-term monitoring
tmux new -s gpu_monitor_c -d "cd /home/ubuntu/GitHub/PlayGround/20250908_LLM_learning && ./monitor_gpu_v2 10"
```

**Test Results:**
- 296 samples in 2.97 seconds = 99.7 Hz
- Power instant values show transient behavior (e.g., 376W to 847W spikes)
- Power average values remain stable (1-sec averaged)
- Temperature readings accurate and responsive

## Future Considerations

### Potential C/Go Port

~~If sustained multi-GPU polling at 100+ Hz with minimal jitter becomes necessary:~~
~~- Python adds interpreter overhead and GIL contention~~
~~- C/Go would reduce per-call FFI cost and improve timing precision~~
~~- Current Python implementation serves as validated reference~~

~~**Recommendation:** Validate signal quality with Python v2 first, then port to C/Go only if performance requirements demand it.~~

**UPDATE:** C implementation is now recommended due to PyNVML API gap (missing `nvmlDeviceGetTemperatureV()`). See C Implementation Investigation section above.

### Additional Metrics

Optional enhancements via Field Values API:
- `NVML_FI_DEV_MEMORY_TEMP` - Memory thermal monitoring
- Power limit tracking
- Energy consumption deltas

## Conclusion

Successfully implemented high-resolution GPU monitoring using NVML Field Values API for accurate instant power measurements on H100 GPUs.

**Python v2 (`monitor_gpu_v2.py`):**
- Validated the Field Values API approach
- Captures true transient power behavior at 10ms resolution
- Limited by PyNVML missing `nvmlDeviceGetTemperatureV()` - forced to use deprecated API

**C Implementation (`monitor_gpu_v2.c`) - RECOMMENDED:**
- Full access to non-deprecated NVML APIs
- Uses `nvmlDeviceGetTemperatureV()` for temperature (proper API)
- Uses `nvmlDeviceGetFieldValues()` for instant + averaged power
- Validated at 100 Hz with 99.7% accuracy
- CSV format compatible with existing visualization tools
- Lower overhead, no GIL contention

**Production Usage:**
```bash
tmux new -s gpu_monitor_c -d "cd /home/ubuntu/GitHub/PlayGround/20250908_LLM_learning && ./monitor_gpu_v2 10"
```

The C implementation is now the recommended solution for production GPU monitoring.

