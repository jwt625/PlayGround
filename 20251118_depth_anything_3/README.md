# Depth Anything V3 - READY TO USE

## 🎉 Installation Complete

The **biggest and best** DA3 model is installed and ready for your dense image sets!

### ✅ What's Installed

- **Model:** DA3 NESTED-GIANT-LARGE
- **Size:** 1.4B parameters (12.6 GB on disk)
- **Type:** Multi-view depth estimation + camera pose prediction
- **Quality:** Maximum - best available model
- **VRAM:** ~24-32 GB (you have 70+ GB free)
- **Cache:** Local at `./models/` (no re-download needed)

### 🚀 Quick Start

```bash
# 1. Activate environment
source setup_env.sh

# 2. Process your dense image set (replaces COLMAP!)
./run_da3.sh /path/to/your/images

# Logs will be saved to: logs/da3_YYYYMMDD_HHMMSS.log
# Output will be in: workspace/gallery/scene/
```

### 📁 Directory Structure

```
20251118_depth_anything_3/
├── .venv/                  # Python virtual environment
├── models/                 # Model cache (12.6 GB)
├── logs/                   # Processing logs with timestamps
├── depth-anything-3/       # DA3 source code
├── setup_env.sh           # Environment setup script
├── run_da3.sh             # Wrapper with logging
└── README.md              # This file
```

### 🎯 Your Use Case: Dense Images Where COLMAP Fails

DA3 is **perfect** for your scenario because:

✅ **No COLMAP needed** - Predicts camera poses directly from images  
✅ **Handles challenging scenes** - Low texture, reflections, repetitive patterns  
✅ **Dense image sets** - Designed for 100-1000+ images  
✅ **Better accuracy** - 35.7% better pose estimation than previous SOTA  
✅ **3DGS ready** - Outputs depth maps, poses, and point clouds  

### 📊 System Status

**GPU Status:**
- GPU 0: 73 GB free
- GPU 1: 81 GB free
- vLLM process: ✅ Stopped (freed up VRAM)

**Environment:**
- Python: 3.10.12
- PyTorch: 2.9.0 with CUDA 12.1
- Package manager: uv (fast!)

### 📖 Documentation

- **Quick reference:** See `USAGE.md`
- **Planning doc:** See `planning.md`
- **CLI help:** Run `da3 --help`

### 🔧 Common Commands

```bash
# Process images with logging (recommended)
./run_da3.sh /path/to/images

# Or use da3 directly
da3 images /path/to/images

# Process with custom output directory
./run_da3.sh /path/to/images --export-dir ./my_output

# Process with higher resolution
./run_da3.sh /path/to/images --process-res 1024

# Use specific GPU
CUDA_VISIBLE_DEVICES=1 ./run_da3.sh /path/to/images

# Launch web UI for testing
da3 gradio

# Get help
da3 --help
da3 images --help
```

### 📋 Logging

All runs using `./run_da3.sh` are automatically logged to:
- **Location:** `logs/da3_YYYYMMDD_HHMMSS.log`
- **Format:** Timestamped entries with full command output
- **Includes:** Model loading, processing progress, errors, timing

### 📤 Output Files

After processing, you'll get:

- **Depth maps** - Dense depth for each image
- **Camera poses** - Relative camera positions (COLMAP replacement!)
- **Point cloud** - 3D reconstruction (.ply format)
- **GLB export** - 3D scene for visualization
- **Ray maps** - For advanced reconstruction

All outputs are ready to use with 3D Gaussian Splatting!

### 🎨 Example Workflow

```bash
# 1. Setup
source setup_env.sh

# 2. Process your dense image set (with logging)
./run_da3.sh ~/my_project/dense_images

# 3. Monitor progress
tail -f logs/da3_*.log

# 4. Check outputs
ls -lh workspace/gallery/scene/

# 5. Use with 3DGS
# - Point cloud: workspace/gallery/scene/*.ply
# - Camera poses: workspace/gallery/scene/cameras/
# - Depth maps: workspace/gallery/scene/depth/
```

### 📊 Advanced Usage

**Process with custom resolution:**
```bash
./run_da3.sh /path/to/images --process-res 1024
```

**Use existing COLMAP data (pose-conditioned depth):**
```bash
source setup_env.sh
da3 colmap /path/to/colmap/sparse/0 2>&1 | tee logs/da3_colmap_$(date +%Y%m%d_%H%M%S).log
```

**Launch Gradio web UI:**
```bash
source setup_env.sh
da3 gradio
# Open http://localhost:7860
```

### 💡 Tips

1. **First run** - Model is already cached, so it will start immediately
2. **Processing time** - Depends on number of images and resolution
3. **GPU selection** - Use `CUDA_VISIBLE_DEVICES=1 ./run_da3.sh ...` to select GPU
4. **Resolution** - Default is 504px, increase with `--process-res 1024` for more detail
5. **Logs** - Always check logs for errors: `tail -f logs/da3_*.log`
6. **Monitoring** - Watch GPU usage: `watch -n 1 nvidia-smi`

### 🆘 Troubleshooting

**Out of memory:**
- Reduce resolution: `--process-res 512`
- Use specific GPU: `CUDA_VISIBLE_DEVICES=1`
- Check GPU memory: `nvidia-smi`

**Slow processing:**
- Check logs for bottlenecks
- Monitor GPU utilization: `nvidia-smi`
- Ensure images are on fast storage (SSD)

**Check logs:**
```bash
# View latest log
tail -f logs/da3_*.log

# Search for errors
grep -i error logs/da3_*.log
```

---

**Status:** ✅ Ready to process your dense image sets!

**Next step:** Point DA3 at your images and let it replace COLMAP! 🚀

