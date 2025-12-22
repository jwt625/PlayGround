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

## Training Issues and Resolutions

### Issue #1: HTTP 403 Error on eval_bundle.zip Download

**Problem:**
Training failed at step 2000 (15.6% complete) when attempting to download evaluation bundle:
```
Downloading https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip...
urllib.error.HTTPError: HTTP Error 403: Forbidden
```

**Root Cause:**
- The S3 bucket `karpathy-public.s3.us-west-2.amazonaws.com` is no longer accessible
- Both `eval_bundle.zip` and `identity_conversations.jsonl` files return 403 Forbidden
- This is a known issue tracked in [nanochat issue #379](https://github.com/karpathy/nanochat/issues/379)

**Impact:**
- Base training crashed at first evaluation checkpoint (step 2000)
- Subsequent pipeline stages failed due to missing checkpoint directories:
  - `/home/ubuntu/.cache/nanochat/base_checkpoints`
  - `/home/ubuntu/.cache/nanochat/mid_checkpoints`
  - `/home/ubuntu/.cache/nanochat/chatsft_checkpoints`

**Resolution:**
Downloaded workaround files from GitHub issue #379 (provided by user @ddudek):

1. **eval_bundle.zip** (24.87 MB):
   ```bash
   cd /home/ubuntu/.cache/nanochat
   wget -O eval_bundle.zip "https://github.com/user-attachments/files/24214174/eval_bundle.zip"
   unzip -o eval_bundle.zip
   ```

2. **identity_conversations.jsonl** (2.29 MB):
   ```bash
   cd /home/ubuntu/.cache/nanochat
   wget -O identity_conversations.jsonl.zip "https://github.com/user-attachments/files/24214240/identity_conversations.jsonl.zip"
   unzip -o identity_conversations.jsonl.zip
   ```

**Verification:**
```bash
ls -la /home/ubuntu/.cache/nanochat/eval_bundle/
ls -la /home/ubuntu/.cache/nanochat/identity_conversations.jsonl
```

**Files cached:**
- `eval_bundle/` directory with CORE evaluation datasets
  - 7 evaluation categories (symbolic_problem_solving, world_knowledge, commonsense_reasoning, safety, reading_comprehension, programming, language_understanding)
  - `core.yaml` configuration
  - `eval_meta_data.csv` metadata
  - Reference GPT-2 model results
- `identity_conversations.jsonl` for mid-training identity learning

**Status:** Resolved - All required files cached locally

---

---

## GPU Resource Utilization Analysis

### Current Training Run Metrics (2025-12-21 08:16)

**GPU Memory Usage:**
- GPU 0: 28,481 MiB / 81,559 MiB (34.9% utilized)
- GPU 1: 20,772 MiB / 81,559 MiB (25.5% utilized)
- Average: ~30% memory utilization

**GPU Compute Utilization:**
- GPU 0: 89% compute utilization, 681W / 700W power
- GPU 1: 99% compute utilization, 673W / 700W power
- Average: ~94% compute utilization

**Current Model Configuration:**
- Depth: 16 layers
- Model dimension: 1024 (depth × 64)
- Number of heads: 8
- Parameters: ~470M
- Device batch size: 8
- Sequence length: 2048

**Memory Breakdown (per GPU, estimated):**
- Model parameters (BF16): ~940 MB (470M params × 2 bytes)
- Optimizer states (AdamW): ~3,760 MB (2 buffers × 4 bytes × 470M params)
- Gradients (BF16): ~940 MB
- Activations (batch_size=8, seq_len=2048): ~14-16 GB
- **Total estimated**: ~20 GB per GPU
- **Actual usage**: ~20-28 GB per GPU (matches estimate)

### Analysis: Compute-Bound vs Memory-Bound

**Key Finding:** Training is **compute-bound**, not memory-bound.

- Compute utilization: ~94% (GPUs working at full capacity)
- Memory utilization: ~30% (significant unused VRAM)

**Implications:**
1. **Can we train a larger model?** Yes, we have ~50-60 GB unused VRAM per GPU
2. **Should we train a larger model?** It depends on training time tolerance
3. **What would change?** Larger model = slower training steps, but better final quality

### Proposed Larger Model Configurations

#### Option 1: Conservative Increase (depth=24)
**Configuration:**
- Depth: 24 layers (vs. current 16)
- Model dimension: 1536 (24 × 64)
- Number of heads: 12
- Parameters: ~1.06B (vs. current 470M)
- Device batch size: 8 (unchanged)

**Expected Resource Usage:**
- Memory per GPU: ~45 GB (vs. current ~25 GB)
- Memory headroom: ~35 GB remaining
- Training time per step: ~1.8× slower (proportional to FLOPs increase)
- Total training time: ~7.7 hours (vs. current ~4.3 hours)

**Chinchilla-optimal training:**
- Training tokens: ~21B (20 × 1.06B params)
- Dataset requirement: ~400 shards (vs. current 200)

#### Option 2: Aggressive Increase (depth=32, original 8×H100 config)
**Configuration:**
- Depth: 32 layers (original configuration)
- Model dimension: 2048 (32 × 64)
- Number of heads: 16
- Parameters: ~1.9B (original target)
- Device batch size: 6-8 (may need reduction)

**Expected Resource Usage:**
- Memory per GPU: ~70-75 GB (approaching limit)
- Memory headroom: ~5-10 GB remaining (tight)
- Training time per step: ~4× slower (proportional to FLOPs increase)
- Total training time: ~17 hours (vs. current ~4.3 hours)

**Chinchilla-optimal training:**
- Training tokens: ~38B (20 × 1.9B params)
- Dataset requirement: ~800 shards (original target)

#### Option 3: Balanced Increase (depth=20)
**Configuration:**
- Depth: 20 layers
- Model dimension: 1280 (20 × 64)
- Number of heads: 10
- Parameters: ~738M
- Device batch size: 8 (unchanged)

**Expected Resource Usage:**
- Memory per GPU: ~33 GB (vs. current ~25 GB)
- Memory headroom: ~47 GB remaining (very safe)
- Training time per step: ~1.4× slower
- Total training time: ~6 hours (vs. current ~4.3 hours)

**Chinchilla-optimal training:**
- Training tokens: ~14.8B (20 × 738M params)
- Dataset requirement: ~280 shards

### Recommendation

**Recommended: Option 3 (depth=20)** for the next training run:

**Rationale:**
1. **Significant quality improvement**: 57% more parameters (738M vs. 470M)
2. **Manageable time increase**: Only ~40% longer training (~6 hrs vs. 4.3 hrs)
3. **Safe memory margins**: Uses only ~40% of available VRAM
4. **Better hardware utilization**: Increases memory usage from 30% to 40%
5. **Reasonable dataset requirement**: 280 shards (vs. 200 current, 800 original)

**To implement:**
```bash
# Modify run_2gpu.sh line 57:
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=20 --device_batch_size=8 --run=$WANDB_RUN

# Download additional shards (line 27):
python -m nanochat.dataset -n 300 &  # Extra margin beyond 280 required
```

**Expected outcomes:**
- Model size: 738M parameters (vs. GPT-2 Medium's 350M, GPT-2 Large's 774M)
- Training time: ~6 hours base training + ~1.5 hours mid/sft = ~7.5 hours total
- Memory usage: ~33 GB per GPU (~40% utilization)
- Compute usage: ~95% (unchanged, still compute-bound)

---

---

## Training Run #2: 2025-12-21 (depth=16, batch_size=8)

### Status: Base Training Complete, Mid-Training Failed

**Base Training Results:**
- Status: Completed successfully
- Total steps: 12,800/12,800 (100%)
- Training time: 259 minutes (4.3 hours)
- Final validation bpb: 0.8669
- CORE metric: 0.1614
- Model saved: `/home/ubuntu/.cache/nanochat/base_checkpoints/d16`

**Performance Metrics:**
- Average time per step: ~1,217 ms
- Throughput: ~430,600 tokens/sec
- MFU: ~43.8%
- Final loss: 2.904

**Evaluation Results:**
- hellaswag_zeroshot: 38.77% accuracy
- arc_easy: 54.04% accuracy
- arc_challenge: 25.68% accuracy
- winogrande: 52.96% accuracy
- piqa: 68.72% accuracy
- boolq: 46.80% accuracy

### Issue #2: Mid-Training Data File Corruption

**Problem:**
Mid-training phase failed with JSON decode error:
```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Root Cause:**
The `identity_conversations.jsonl` file was corrupted - it contained an HTTP 403 XML error response (278 bytes) instead of the actual JSONL data (2.2MB). This occurred because the AWS S3 bucket hosting the file (https://karpathy-public.s3.us-west-2.amazonaws.com/) returned 403 Forbidden errors.

**File Content (Corrupted):**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<Error><Code>AccessDenied</Code><Message>Access Denied</Message>...</Error>
```

**Resolution:**
1. **Immediate fix**: Downloaded correct file from GitHub issue #379 workaround:
   - Source: https://github.com/user-attachments/files/24214240/identity_conversations.jsonl.zip
   - File size: 2.2MB (correct) vs 278 bytes (corrupted)
   - Location: `/home/ubuntu/.cache/nanochat/identity_conversations.jsonl`
   - Verified: Valid JSONL format with 996 identity conversation entries

2. **Permanent fix**: Modified `run_2gpu.sh` to prevent re-downloading:
   - Changed line 38 from unconditional `curl` to conditional download
   - Now checks if file exists and is valid (not 278 bytes)
   - Downloads from GitHub workaround instead of broken AWS S3 URL
   - Only downloads if file is missing or corrupted

**Modified code in `run_2gpu.sh` (lines 39-49):**
```bash
# Download identity_conversations.jsonl only if it doesn't exist or is corrupted (278 bytes = AWS 403 error)
IDENTITY_FILE="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
if [ ! -f "$IDENTITY_FILE" ] || [ $(stat -c%s "$IDENTITY_FILE") -eq 278 ]; then
    echo "Downloading identity_conversations.jsonl from GitHub workaround (AWS S3 returns 403)..."
    curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl.zip" "https://github.com/user-attachments/files/24214240/identity_conversations.jsonl.zip"
    unzip -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl.zip" -d "$NANOCHAT_BASE_DIR"
    rm -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl.zip"
    echo "Downloaded identity_conversations.jsonl ($(stat -c%s "$IDENTITY_FILE") bytes)"
else
    echo "identity_conversations.jsonl already exists ($(stat -c%s "$IDENTITY_FILE") bytes), skipping download"
fi
```

**Impact:**
- Mid-training phase failed during data loading
- Subsequent pipeline stages also failed (cascading failure)
- Base training completed successfully and checkpoints are intact

**Status:** Resolved (2025-12-21 17:23)
- File re-downloaded and verified: 2.2 MB, 996 lines
- Script modified to prevent future re-downloads
- Ready to resume mid-training phase

**Next Steps:**
1. Resume pipeline from mid-training phase (skip base training)
2. Monitor mid-training progress
3. Complete full pipeline (mid_train -> chat_sft -> chat_eval)
4. Consider training a larger model (depth=20 or depth=24) to better utilize available GPU memory

---

## Training Log Analysis and Loss Curve Extraction

### Log Parser Implementation

Created `parse_training_logs.py` to extract training metrics from terminal output logs for post-hoc analysis.

**Functionality:**
- Parses training step logs using regex pattern matching
- Extracts metrics: step number, loss, gradient norm, learning rate multiplier, time per step, throughput, MFU, cumulative training time
- Outputs structured CSV files for analysis
- Generates multi-panel visualization plots

**Usage:**
```bash
python parse_training_logs.py --csv
python parse_training_logs.py --log-dir /path/to/logs --output custom_plot.png
```

**Output files:**
- CSV format: `training_YYYYMMDD_HHMMSS.csv` with columns: step, loss, grad_norm, lrm, dt_ms, tok_per_sec, mfu, total_time_min
- Visualization: `loss_curves.png` with 4 subplots (loss vs step, loss vs time, MFU vs step, throughput vs step)

### Parsed Training Runs

**Run 1: training_20251221_070319 (Base Training - Complete)**
- Steps: 12,800 (0 to 12,799)
- Loss: 11.09 to 2.77 (8.32 reduction)
- Training time: 259.23 minutes (4.3 hours)
- Average throughput: 431,260 tokens/sec
- Average MFU: 43.90%
- Status: Complete base training run

**Run 2: training_20251221_172457 (Mid-Training)**
- Steps: 813 (1 to 813)
- Loss: 2.08 to 1.26 (0.82 reduction)
- Training time: 15.98 minutes
- Status: Mid-training phase

**Run 3: training_20251221_183935 (Base Training - Incomplete)**
- Steps: 11,939 (0 to 11,938)
- Loss: 11.09 to 2.68 (8.41 reduction)
- Training time: 586.16 minutes (9.8 hours)
- Status: Stopped before completion at step 11,938/12,800 (93.3%)

**Runs 4-5: training_20251221_171757, training_20251221_172138**
- Status: Failed or aborted early (no training steps logged)

### Key Observations

**Loss convergence:**
- Initial loss: ~11.09 (random initialization)
- Rapid descent in first 1,000 steps: 11.09 to 3.46 (67% reduction)
- Slower convergence after step 5,000: 3.07 to 2.90 (5.5% reduction)
- Final loss: 2.77-2.90 depending on run completion

**Training stability:**
- Consistent throughput: ~430K tokens/sec across all runs
- Stable MFU: ~43.8-43.9% throughout training
- No significant performance degradation over time

**Data availability:**
- All training metrics now available in CSV format for further analysis
- Loss curves plotted against both training steps and wall-clock time
- Enables comparison across different training runs and configurations

### Interactive Visualization Update

**Enhanced plotting capabilities:**
- Migrated from static matplotlib plots to interactive Plotly HTML dashboards
- Custom hover tooltips showing full metric details without abbreviation
- Script now defaults to `nanochat/` directory when run from parent folder
- Generates `loss_curves.html` (6.1 MB) with zoom, pan, and hover interactions
- Static PNG output still available via `--static` flag

**Usage:**
```bash
python parse_training_logs.py              # Interactive HTML (default)
python parse_training_logs.py --static     # Static PNG
python parse_training_logs.py --csv        # Also export CSV files
```

---

## Training Run #3: 2025-12-21 (depth=24, batch_size=8)

### Status: In Progress (37.7% complete)

**Model Configuration:**
- Depth: 24 layers
- Parameters: 880,803,840 (~881M, 2.6× larger than depth=16)
- Target steps: 33,600 (Chinchilla-optimal: 20:1 token:param ratio)
- Target tokens: 17.6B (vs 6.7B for depth=16)

**Current Progress (as of step 12,678):**
- Steps completed: 12,678 / 33,600 (37.7%)
- Training time: 622 minutes (10.4 hours)
- Current loss: 2.70-2.84 range
- Time per step: ~2.94 seconds (45% slower than depth=16)
- Throughput: ~178K tokens/sec
- MFU: ~50.3%

**Loss Comparison with depth=16 model:**
```
Step   | depth=16 (336M) | depth=24 (881M) | Improvement
-------|-----------------|-----------------|------------
     0 |        11.0904  |        11.0904  |    0.0000
  1000 |         3.4627  |         3.3553  |   -0.1074
  5000 |         3.0704  |         2.9279  |   -0.1425
 10000 |         2.9700  |         2.8135  |   -0.1565
 12000 |         2.9072  |         2.8153  |   -0.0918
```

**Key findings:**
- Larger model achieves 3-5% lower loss at equivalent step counts
- Loss advantage most pronounced at steps 5,000-10,000
- Already outperforms fully-trained depth=16 model despite being only 37% complete

**Estimated completion:**
- Remaining time: ~16 hours
- Total training time: ~26 hours
- Expected final loss: <2.7 (significantly better than depth=16's 2.90)

### Issue #3: Missing Intermediate Checkpoints

**Problem:**
Training runs do not save intermediate checkpoints by default. The `save_every` parameter defaults to `-1`, meaning checkpoints are only saved at the final training step.

**Impact:**
- Training interruption results in complete loss of progress
- No recovery points for long-running training jobs (26+ hours)
- Current depth=24 run at 37% completion has no checkpoint to resume from

**Root Cause:**
Default configuration inherited from original `run1000.sh` script, which assumes uninterrupted 41.6-hour training runs on 8×H100 nodes. Design prioritizes disk space savings over fault tolerance.

**Resolution:**
Modified `run_2gpu.sh` to include `--save_every=3000` parameter:
```bash
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
    --depth=16 --device_batch_size=8 --save_every=3000 --run=$WANDB_RUN
```

**Checkpoint policy:**
- Saves checkpoint every 3,000 steps
- For depth=24 (33,600 steps): checkpoints at steps 3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000, 33600
- Checkpoint size: ~3.5 GB per checkpoint for 881M parameter model
- Total disk usage: ~38.5 GB for all checkpoints

**Status:** Fixed for future runs. Current depth=24 run will continue without intermediate checkpoints until completion.

**Recommendation:** For production runs exceeding 10 hours, always set `save_every` to a reasonable interval (2000-5000 steps depending on model size and training duration).

