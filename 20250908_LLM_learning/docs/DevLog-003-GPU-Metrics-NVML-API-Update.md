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

Created `visualize_gpu_metrics_v2_enhanced.py` for v2 CSV format:
- Plots both instant and average power curves (solid and dotted lines respectively)
- Includes all datapoints without downsampling (downsample_factor=1)
- Extracts timestamp from CSV filename for output files
- Generates interactive HTML plot: `gpu_metrics_plot-v2_<timestamp>.html`
- Exports text summary: `gpu_metrics_v2_summary_<timestamp>.txt`
- Supports idle period detection and automatic trimming with configurable buffer
- Compatible with both Python and C monitor implementations

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
- `visualize_gpu_metrics_v2_enhanced.py` - Visualization script for v2 format (instant + average power)

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

## Visualization Script Enhancement and Data Processing

### Command-Line Argument Support

Both visualization scripts (`visualize_gpu_metrics_v2.py` for v1 format and `visualize_gpu_metrics_v2_enhanced.py` for v2 format) were enhanced to accept command-line arguments for improved usability. The scripts now support positional arguments for CSV filename and downsampling factor, along with optional flags for power threshold and buffer minutes. This eliminates the need to edit hardcoded values in the script for each processing run. The default downsampling factor is set to 10 for v2 format based on empirical analysis showing that NVML metrics update at approximately 10 Hz despite 100 Hz sampling, meaning 90% of consecutive samples are duplicates. For v1 format, analysis revealed even higher redundancy with 327 Hz sampling but only 10 Hz actual updates, requiring a downsampling factor of 32 to eliminate the 97% duplicate samples.

### Data Quality Analysis and Processing Results

Analysis of the actual metric update frequencies revealed significant differences between v1 and v2 implementations. The v1 Python implementation using pynvml sampled at 326.6 Hz but power values only changed at 9.9 Hz (3% unique samples), resulting in massive data redundancy. The v2 C implementation improved efficiency by sampling at 99.2 Hz with the same 9.9 Hz update rate, reducing redundancy from 97% to 90%. Two major datasets were processed and visualized: a 98 MB v2 format file capturing 4.24 hours of training with 1.5M samples (downsampled to 142K points), and a 1.2 GB v1 format file capturing 19.0 hours of training with 22.6M samples (downsampled to 698K points). The longer v1 run showed higher average power consumption (657W vs 571W) but lower peak power (784W vs 915W), suggesting different training workload characteristics. Both datasets had idle periods automatically detected and trimmed, with interactive HTML visualizations and summary statistics generated for analysis.

### Visualization Output Files

The processed datasets generated the following visualization artifacts: `gpu_metrics_plot-v2_20251222_210600.html` (300 MB, 4.24-hour run with instant and averaged power curves), `gpu_metrics_v2_summary_20251222_210600.txt` (summary statistics for v2 run), `gpu_metrics_plot_20251222_060524.html` (19.0-hour run with single power readings), and corresponding summary files. The v2 visualization includes both instant and averaged power as separate traces, allowing comparison of transient behavior versus 1-second averaged values. All visualizations use Plotly for interactive exploration with pan, zoom, and hover capabilities, making it easy to identify power spikes, thermal throttling events, and training phase transitions across multi-hour runs.

## GPU Voltage and Current Metrics Investigation

### Problem Statement

Investigated whether GPU core voltage and current (amperage) can be monitored via software on Linux, similar to how power and temperature are accessible through NVML.

### Summary: Voltage/Current NOT Available for Standard GPUs

Comprehensive research confirms that NVIDIA does not expose voltage or current metrics for discrete GPUs through any standard software interface.

| Method | Voltage | Current | Notes |
|--------|---------|---------|-------|
| NVML API | No | No | Not implemented for GPUs |
| nvidia-smi | Deprecated (returns N/A) | No | `--query-gpu=voltage.gpu` removed |
| DCGM | NVSwitch only | NVSwitch only | Field IDs 701-704 are for NVSwitches, not GPUs |
| Linux sysfs/hwmon | No | No | NVIDIA driver does not expose hwmon interface |
| I2C/SMBus | Blocked | Blocked | VRM I2C bus not exposed to userspace |

### What IS Available

| Metric | API | Notes |
|--------|-----|-------|
| Power (Watts) | `nvmlDeviceGetPowerUsage()`, Field Values API | Instant and averaged |
| Total Energy (mJ) | `nvmlDeviceGetTotalEnergyConsumption()` | Cumulative since driver load |
| NVSwitch Voltage | DCGM `DCGM_FI_DEV_NVSWITCH_VOLTAGE_MVOLT` (701) | NVSwitch only |
| NVSwitch Current | DCGM `DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ` (702-704) | NVSwitch only |
| Jetson Voltage/Current | INA3221 via sysfs | Embedded platforms only |

### Validation: Energy Consumption via C API

Created `nvml_test.c` to verify energy consumption availability:

```c
unsigned long long energy;
result = nvmlDeviceGetTotalEnergyConsumption(device, &energy);
// Returns cumulative energy in millijoules since driver load
```

**Test Results (H100 GPUs):**
- GPU 0: Total Energy 865,830 kJ (cumulative)
- GPU 1: Total Energy 648,324 kJ (cumulative)
- Power Instant: 70.79 W, 70.58 W (idle)
- Power Average: 70.79 W, 70.58 W (idle)

Note: `nvidia-smi --query-gpu=energy.counter` does not work; the field is only accessible via C API.

### I2C Investigation

The NVIDIA driver exposes I2C adapters, but they are for DDC (monitor communication), not VRM access:

```
i2c-1  i2c  NVIDIA i2c adapter 2 at 7:00.0  I2C adapter
i2c-2  i2c  NVIDIA i2c adapter 2 at 8:00.0  I2C adapter
```

Scanning these buses with `i2cdetect` returns no devices. The VRM I2C bus is internal and not exposed to userspace.

### Technical Background

1. **NVIDIA has intentionally locked I2C access to VRM** on consumer and datacenter GPUs from Pascal onwards (confirmed by HWiNFO developer Martin).

2. **Windows tools (HWiNFO, GPU-Z)** also cannot read actual voltage on Pascal+ GPUs. They only display VID (Voltage ID) values reported by the driver, not measured output voltage.

3. **DCGM voltage/current fields exist** but only for NVSwitches in datacenter configurations, not for GPU cores.

4. **Jetson platforms differ** because they include INA3221 power monitors accessible via I2C sysfs nodes.

### Workarounds for Voltage/Current Measurement

If voltage/current measurement is required:

1. **External power meter** on PCIe power cables (Kill-A-Watt or similar)
2. **Shunt resistor + ADC** on power rails (hardware modification)
3. **Server BMC/IPMI** if the server exposes GPU rail monitoring via baseboard management
4. **Derive from Power**: With known power P and estimated voltage V (0.8-1.1V typical), estimate current I = P/V

### Related Files

- `nvml_test.c` - C program testing NVML power and energy APIs
- Compiled with: `gcc -o nvml_test nvml_test.c -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lnvidia-ml`

### Conclusion

GPU voltage and current metrics are not accessible via software on standard NVIDIA GPUs. Power consumption (Watts) and cumulative energy (millijoules) are the closest available metrics. For precise voltage/current analysis, external measurement hardware is required.

