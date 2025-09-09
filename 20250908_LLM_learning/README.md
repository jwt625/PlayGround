# LLM Training & Inference Learning Journey

A comprehensive 3-6 month learning project for mastering Large Language Model training and inference, optimized for high-performance computing with dual NVIDIA H100 GPUs.

## Machine Specifications
- **CPU**: Intel Xeon Platinum 8480+ (52 cores)
- **Memory**: 442GB RAM
- **GPUs**: 2x NVIDIA H100 80GB HBM3 (160GB total VRAM)
- **Optimized for**: Large model training and inference

## Project Structure

```
20250908_LLM_learning/
├── README.md                    # This file
├── pyproject.toml              # Project dependencies (uv)
├── .venv/                      # Virtual environment
├── docs/                       # Documentation and planning
│   └── DevLog-000-planning.md  # Detailed learning plan
└── week1-attention/            # Phase 1, Week 1: Attention mechanisms
    ├── annotated_transformer.py
    ├── simple_training.py
    └── README.md
```

## Learning Plan Overview

### Phase 1: Fundamentals (Weeks 1-4)
- **Week 1**: ✅ Attention mechanisms and Annotated Transformer
- **Week 2**: minGPT and multi-GPU training experiments
- **Week 3**: BERT and HuggingFace transformers
- **Week 4**: nanoGPT with 1B+ parameter models

### Phase 2: Scaling Up (Weeks 5-8)
- Scaling laws and compute-optimal training
- Advanced distributed training (DeepSpeed, ZeRO)
- T5 and text-to-text transformers
- 3-7B parameter model training

### Phase 3: Inference & Optimization (Weeks 9-10)
- vLLM, TensorRT-LLM, and inference frameworks
- Quantization (FP8, GPTQ, AWQ)
- Large model deployment (13B-34B parameters)

### Phase 4: Finetuning & Alignment (Weeks 11-14)
- LoRA and QLoRA techniques
- Full-precision vs quantized finetuning
- RLHF and DPO alignment methods

### Phase 5: Advanced Topics (Weeks 15-20)
- Mixture of Experts (MoE) architectures
- Retrieval-Augmented Generation (RAG)
- Multimodal LLMs
- Production-scale systems

## Quick Start

### Environment Setup
```bash
# Clone and navigate to project
cd 20250908_LLM_learning

# Activate virtual environment
source .venv/bin/activate

# Verify GPU setup
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### Week 1: Test Transformer Implementation
```bash
cd week1-attention
python annotated_transformer.py  # Test basic functionality
python simple_training.py        # Train on copy task
```

## Dependencies

Managed with `uv` for fast, reliable package management:
- PyTorch 2.8.0 with CUDA support
- NumPy, Matplotlib for data handling and visualization
- Jupyter for interactive development

## Hardware Utilization Strategy

This project is designed to fully exploit the dual H100 setup:

- **160GB VRAM**: Enables training/inference of 30B+ parameter models
- **52-core CPU**: Massive parallel data preprocessing and loading
- **442GB RAM**: Load entire large datasets in memory
- **Advanced techniques**: Model parallelism, pipeline parallelism, ZeRO-3

## Progress Tracking

- [x] **Week 1**: Annotated Transformer implementation complete
- [ ] **Week 2**: minGPT and multi-GPU experiments
- [ ] **Week 3**: BERT and tokenizer experiments
- [ ] **Week 4**: nanoGPT large model training

## Key Learning Outcomes

By the end of this journey, you will have:

1. **Deep understanding** of Transformer architectures
2. **Hands-on experience** with large model training (3-7B+ parameters)
3. **Production skills** for model deployment and optimization
4. **Advanced techniques** for alignment and finetuning
5. **Complete pipeline** from training to serving

## Resources

- **Planning**: See `docs/DevLog-000-planning.md` for detailed weekly breakdown
- **Papers**: Key papers referenced throughout the learning plan
- **Code**: Clean, educational implementations optimized for learning

## Next Steps

Continue with Week 2: minGPT implementation and multi-GPU training experiments.
