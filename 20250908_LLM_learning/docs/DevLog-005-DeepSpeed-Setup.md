# DevLog-005: DeepSpeed Setup and Machine Evaluation

**Date:** 2025-01-13

## Overview

Setting up DeepSpeed for distributed training experiments. This log documents the machine evaluation and setup process for running DeepSpeed training while monitoring GPU metrics with `monitor_gpu_v2.c`.

## Machine Specifications

| Component | Value |
|-----------|-------|
| GPU | 2x NVIDIA H100 80GB HBM3 |
| GPU Memory | 81559 MiB per GPU (163GB total) |
| CUDA Version | 12.8 |
| Driver Version | 570.124.06 |
| CPU | Intel Xeon Platinum 8480+ (52 vCPUs, 26 cores, HT enabled) |
| RAM | 442GB (432GB available) |
| PyTorch | 2.8.0+cu128 |
| OS | Ubuntu 22.04 (kernel 6.8.0-52-generic) |

## What is DeepSpeed?

DeepSpeed is Microsoft's deep learning optimization library for distributed training at scale.

Key features:
- **ZeRO (Zero Redundancy Optimizer)**: Memory-efficient training for massive models
- **3D Parallelism**: Data, pipeline, and tensor parallelism
- **Mixed precision training**: FP16/BF16 support
- **Gradient checkpointing**: Trade compute for memory
- **CPU/NVMe Offloading**: Extend training capacity beyond GPU memory

Notable models trained with DeepSpeed:
- Megatron-Turing NLG (530B)
- BLOOM (176B)
- GLM (130B)
- GPT-NeoX (20B)

## Machine Fitness Evaluation

**Result: EXCELLENT**

This machine is very well suited for DeepSpeed:

1. **H100 GPUs**: Top-tier datacenter GPUs with massive memory (80GB each)
2. **CUDA 12.8**: Fully compatible with DeepSpeed
3. **PyTorch 2.8**: Already working with CUDA (2 GPUs detected)
4. **442GB RAM**: Sufficient for CPU offloading (ZeRO-Offload)
5. **Multi-GPU**: 2 GPUs enable data parallelism or ZeRO-2/ZeRO-3

## Repository

DeepSpeed cloned to: `DeepSpeed/`

Source: https://github.com/deepspeedai/DeepSpeed

## TODOs

- [x] Install DeepSpeed package
  ```bash
  uv pip install deepspeed
  ```
  Installed: deepspeed==0.18.4

- [x] Compile monitor_gpu_v2.c binary
  ```bash
  cd gpu_fast_metrics
  gcc -o monitor_gpu_v2 monitor_gpu_v2.c -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lnvidia-ml -O2
  ```

- [x] Verify DeepSpeed installation
  ```bash
  ds_report
  ```
  Result: Most ops show [OKAY] compatibility, will JIT compile when needed.

- [x] Create training script for benchmarking
  Created: `deepspeed_benchmark/train_simple.py` (124.44M parameter GPT-like model)

- [x] Create DeepSpeed configuration JSON (ZeRO stage 2)
  Created: `deepspeed_benchmark/ds_config.json`

- [x] Run training with GPU monitoring
  Created: `deepspeed_benchmark/run_benchmark.sh`
  ```bash
  cd deepspeed_benchmark && ./run_benchmark.sh 20
  ```

- [ ] Run longer benchmark for detailed GPU metrics analysis
  - Compare instant vs average power readings
  - Examine power fluctuations during forward/backward passes
  - Check temperature behavior under sustained load

## Initial Test Results (20 steps)

- Model: 124.44M parameters (12 layers, 768 embedding)
- Batch size: 8 per GPU, sequence length: 1024
- ZeRO Stage 2, FP16 enabled
- Throughput: ~153K tokens/sec
- Peak GPU power: GPU0=529W, GPU1=540W (out of 700W TDP)
- GPU temperature: rose from 28-29C to 30-31C

## DeepSpeed Requirements

From `DeepSpeed/requirements/requirements.txt`:
- einops
- hjson
- msgpack
- ninja
- numpy
- packaging>=20.0
- psutil
- py-cpuinfo
- pydantic>=2.0.0
- torch
- tqdm

## Notes

- The DeepSpeed examples directory contains only a README pointing to external resources
- Training examples are typically found in HuggingFace repos or DeepSpeedExamples repo
- The `ds_report` command will show compatibility status after installation

