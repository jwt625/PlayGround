# DevLog-001: 2-GPU Training Configuration and Monitoring Setup

## Metadata
- **Date**: 2025-12-21
- **Author**: Wentao (jwt625)
- **Project**: nanochat LLM Training
- **Hardware**: 2× NVIDIA H100 80GB HBM3
- **Environment**: KVM Cloud VM

---

## Overview

Adapted the nanochat training pipeline from 8-GPU to 2-GPU configuration and implemented high-resolution GPU monitoring for power and thermal metrics.

---

## Hardware Configuration

**GPUs:**
- Count: 2× NVIDIA H100 80GB HBM3
- Total VRAM: 160 GB
- Compute capacity: 1/4 of original 8-GPU setup

**Virtualization:**
- Platform: KVM
- GPU passthrough enabled

---

## Training Configuration

### Script: `run_2gpu.sh`

**Key modifications from `run1000.sh`:**

1. **GPU count**: 8 → 2
   ```bash
   NPROC_PER_NODE=2
   ```

2. **Model depth**: 32 → 16
   ```bash
   --depth=16
   ```

3. **Dataset shards**: 800 → 200
   ```bash
   python -m nanochat.dataset -n 200 &
   ```

4. **Batch size**: Maintained at 8
   ```bash
   --device_batch_size=8
   ```

### Expected Model Specifications

- Parameters: ~470M (vs. 1.9B in original)
- Training tokens: ~9.4B (Chinchilla scaling: 20 × params)
- Dataset requirement: ~200 shards
- Training time: ~4.3 hours (base training only)
- Total pipeline time: ~5.5-6.5 hours

### Observed Performance Metrics

**Training step performance:**
- Time per step: ~1,217 ms
- Throughput: ~430,600 tokens/sec
- MFU: ~43.8%
- Total steps: 12,800
- Estimated completion: 4 hours 19 minutes

**MFU Analysis:**
- GPU utilization: 100%
- MFU: 43.8% (expected for 470M parameter model on 2 GPUs)
- Bottleneck: Memory bandwidth (typical for smaller models)
- Performance: Acceptable for this configuration

---

## GPU Monitoring System

### Implementation: `monitor_gpu.py`

**Technology:**
- Library: pynvml (Python NVML bindings)
- Sampling rate: 100 Hz (10ms intervals)
- Resolution: Highest available without additional software installation

**Metrics captured:**
- GPU power consumption (Watts)
- GPU temperature (Celsius)
- Timestamp with millisecond precision
- Per-GPU granularity

**File output:**
- Format: CSV
- Naming: `gpu_metrics_YYYYMMDD_HHMMSS.csv`
- Size estimate: ~147 MB for 8-hour run
- Columns: timestamp, elapsed_sec, gpu0_power_w, gpu0_temp_c, gpu1_power_w, gpu1_temp_c

**Verified performance:**
- Actual sampling rate: 99.2 Hz
- Overhead: Negligible
- Runs independently of training process

### Usage

**Start monitoring:**
```bash
python3 monitor_gpu.py 10
```

**Run in tmux:**
```bash
tmux new -s gpu_monitor -d "cd /home/ubuntu/GitHub/PlayGround/20250908_LLM_learning && python3 monitor_gpu.py 10"
```

**Combined training and monitoring:**
```bash
tmux new -s training -d "cd /home/ubuntu/GitHub/PlayGround/20250908_LLM_learning/nanochat && ./run_2gpu.sh" \; split-window -h "cd /home/ubuntu/GitHub/PlayGround/20250908_LLM_learning && python3 monitor_gpu.py 10" \; attach
```

---

## Technical Notes

### MFU vs GPU Utilization

- GPU utilization at 100% indicates GPU is always busy
- MFU at 43.8% indicates actual compute efficiency
- Gap explained by:
  - Memory bandwidth bottleneck (primary factor)
  - Non-compute operations (LayerNorm, softmax, activations)
  - Distributed training overhead (all-reduce, gradient sync)
  - Smaller model size (less arithmetic intensity)

### Monitoring Resolution Limitations

**Available on KVM cloud VM:**
- nvidia-smi: 1 second minimum
- pynvml: 10-100ms practical limit
- CUDA events: Microsecond precision (requires code modification)

**Not available without installation:**
- DCGM: Sub-second resolution
- Nsight Systems: Nanosecond profiling

**Selected approach:**
- pynvml at 10ms intervals provides optimal balance of resolution and overhead

---

## Files Created

1. `20250908_LLM_learning/nanochat/run_2gpu.sh` - Training script for 2-GPU configuration
2. `20250908_LLM_learning/monitor_gpu.py` - GPU monitoring script
3. `DevLog/DevLog-001-2GPU-Training-Setup.md` - This document

---

## Next Steps

- Monitor training progress and validate loss convergence
- Analyze GPU power and thermal data post-training
- Evaluate model performance metrics (CORE, validation bpb)
- Document final training results

