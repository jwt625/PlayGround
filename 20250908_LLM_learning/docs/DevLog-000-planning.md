# LLM Training & Inference Learning Plan (3–6 Months)
**Machine Specs:** Intel Xeon Platinum 8480+ (52 cores), 442GB RAM, 2x NVIDIA H100 80GB HBM3
**Revised for High-Performance Computing Setup**

## Phase 1: Fundamentals (Weeks 1–4)
**Goal:** Understand Transformer basics and build a small GPT.

- **Week 1**
  - Read: *Attention Is All You Need* (Vaswani et al., 2017)
  - Code: [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
  - [newer version](https://nlp.seas.harvard.edu/annotated-transformer/)

- **Week 2**
  - Read: *Language Models are Unsupervised Multitask Learners (GPT-2)*
  - Code: [minGPT](https://github.com/karpathy/minGPT)
  - Train GPT on small dataset (Shakespeare, toy wiki subset).
  - **REVISED:** With dual H100s, experiment with larger datasets and multi-GPU training from the start.

- **Week 3**
  - Read: *BERT: Pre-training of Deep Bidirectional Transformers* (2018)
  - Code: HuggingFace `transformers` tutorials (tokenizers, model training).

- **Week 4**
  - Project: Reproduce GPT training on a medium corpus using [nanoGPT](https://github.com/karpathy/nanoGPT).
  - **REVISED:** Train larger models (1B+ parameters) using both H100s with data parallelism.
  - Deliverable: A working GPT-like model generating coherent text (significantly larger than originally planned).

---

## Phase 2: Scaling Up (Weeks 5–8)
**Goal:** Learn distributed training and scaling laws.

- **Week 5**
  - Read: *Scaling Laws for Neural Language Models* (Kaplan et al., 2020).
  - **REVISED:** Experiment with scaling laws using 2-7B parameter models on dual H100s.
  - Test compute-optimal training ratios with your available VRAM (160GB total).

- **Week 6**
  - Tools: [HuggingFace Accelerate](https://github.com/huggingface/accelerate), [DeepSpeed](https://github.com/microsoft/DeepSpeed).
  - **REVISED:** Master advanced multi-GPU techniques: ZeRO-2/3, gradient checkpointing, mixed precision.
  - Experiment with model parallelism for models that exceed single GPU memory.

- **Week 7**
  - Read: *T5: Exploring Transfer Learning with a Unified Text-to-Text Transformer*.
  - Experiment: Train a small T5 variant with HuggingFace.

- **Week 8**
  - **REVISED:** Deliverable: Train a 3-7B parameter model on full datasets (The Pile, RedPajama).
  - Utilize full dual H100 capacity for training larger, more capable models.

---

## Phase 3: Inference & Optimization (Weeks 9–10)
**Goal:** Efficient inference.

- **Week 9**
  - Explore inference frameworks:
    - [vLLM](https://github.com/vllm-project/vllm) - optimize for dual H100 setup
    - [text-generation-inference](https://github.com/huggingface/text-generation-inference)
    - [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - H100-optimized inference
  - **REVISED:** Benchmark throughput with larger models (13B-30B) that can fit in your 160GB VRAM.

- **Week 10**
  - Study quantization methods:
    - [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
    - GPTQ, AWQ, FP8 quantization (H100 native support)
  - **REVISED:** Deploy and serve larger models (LLaMA-2 13B/30B, CodeLlama 34B) with optimized inference.

---

## Phase 4: Finetuning & Alignment (Weeks 11–14)
**Goal:** Apply LoRA, QLoRA, and RLHF/DPO.

- **Week 11**
  - Read: *LoRA: Low-Rank Adaptation of Large Language Models*.
  - Code: [PEFT](https://github.com/huggingface/peft).
  - **REVISED:** Fine-tune larger models (LLaMA-2 13B/30B, CodeLlama 34B) using LoRA on dual H100s.

- **Week 12**
  - Read: *QLoRA: Efficient Finetuning of Quantized LLMs*.
  - **REVISED:** Experiment with full-precision finetuning of large models (your VRAM allows it).
  - Compare QLoRA vs full finetuning performance and efficiency.

- **Week 13**
  - Read: *InstructGPT* (2022), *DPO* (2023).
  - Code: [TRL](https://github.com/huggingface/trl).
  - Implement preference optimization.

- **Week 14**
  - **REVISED:** Deliverable: Multiple fine-tuned large instruct-style LLMs (13B-34B parameters).
  - Compare different finetuning approaches and model sizes.

---

## Phase 5: Advanced Topics (Weeks 15–20)
**Goal:** Explore state-of-the-art directions.

- **Week 15**
  - Read: *Switch Transformers: Scaling to Trillion Parameters* (MoE).
  - **REVISED:** Explore and potentially train smaller MoE models (Mixtral-style).
  - Your dual H100 setup can handle larger MoE architectures than typical consumer setups.

- **Week 16**
  - Study Retrieval-Augmented Generation:
    - Papers: REALM, RAG, RETRO.
    - Tools: [LlamaIndex](https://github.com/jerryjliu/llama_index), [LangChain](https://github.com/hwchase17/langchain).

- **Week 17–18**
  - Multimodal LLMs:
    - Read: Flamingo, LLaVA.
    - Code: [LLaVA](https://github.com/haotian-liu/LLaVA).

- **Week 19–20**
  - **REVISED:** Project: Build a production-scale RAG pipeline with large fine-tuned models (13B-34B).
  - Leverage your compute power for real-time inference of larger, more capable models.
  - Deliverable: High-performance end-to-end system (retrieval + large LLM inference).

---

## Continuous Habits
- Follow new papers: [arXiv-sanity](https://arxiv-sanity-lite.com/), HuggingFace blog.
- Engage in communities: HuggingFace Discord, EleutherAI.
- Track leaderboards: [Open LLM Leaderboard](https://huggingface.co/leaderboards).

---

## Deliverables by End of Journey (REVISED FOR DUAL H100 SETUP)
1. **Large GPT-like model trained from scratch** (1B+ parameters).
2. **Multi-billion parameter model** trained with advanced distributed techniques (3-7B).
3. **High-performance inference deployment** with large models (13B-34B) using TensorRT-LLM/vLLM.
4. **Full-precision and LoRA finetuned large instruct models** (13B-34B parameters).
5. **Advanced alignment experiments** (DPO/RLHF) on large models.
6. **Production-scale RAG + large LLM pipeline** with real-time inference.
7. **BONUS:** MoE architecture experiments and custom model architectures.

## Hardware Utilization Strategy
- **Dual H100 80GB**: 160GB total VRAM enables training/inference of 30B+ parameter models
- **52-core Xeon**: Massive CPU power for data preprocessing, tokenization, and parallel data loading
- **442GB RAM**: Enables loading entire large datasets in memory for faster training
- **Advanced techniques**: Model parallelism, pipeline parallelism, and ZeRO-3 for maximum efficiency

---

## Recommended Repository Structure

```
20250908_LLM_learning/
├── README.md                          # Project overview and setup instructions
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment specification
├── .gitignore                        # Git ignore patterns
├── .gitattributes                    # Git LFS configuration for large files
│
├── docs/                             # Documentation and learning materials
│   ├── DevLog-000-planning.md       # This planning document
│   ├── weekly-logs/                 # Weekly progress logs
│   │   ├── week-01.md
│   │   ├── week-02.md
│   │   └── ...
│   ├── papers/                      # Key papers and notes
│   │   ├── attention-is-all-you-need.md
│   │   ├── scaling-laws.md
│   │   └── ...
│   └── benchmarks/                  # Performance benchmarks and results
│
├── phase1-fundamentals/             # Phase 1: Transformer basics (Weeks 1-4)
│   ├── week1-attention/
│   │   ├── annotated_transformer.py
│   │   ├── attention_mechanisms.py
│   │   └── experiments/
│   ├── week2-gpt/
│   │   ├── minGPT/                  # Fork/clone of minGPT
│   │   ├── shakespeare_training/
│   │   └── multi_gpu_experiments/
│   ├── week3-bert/
│   │   ├── tokenizer_experiments/
│   │   ├── huggingface_tutorials/
│   │   └── bert_training/
│   └── week4-nanogpt/
│       ├── nanoGPT/                 # Fork/clone of nanoGPT
│       ├── large_model_training/    # 1B+ parameter experiments
│       └── distributed_training/
│
├── phase2-scaling/                  # Phase 2: Scaling up (Weeks 5-8)
│   ├── week5-scaling-laws/
│   │   ├── scaling_experiments/
│   │   ├── compute_optimal_ratios/
│   │   └── model_size_analysis/
│   ├── week6-distributed/
│   │   ├── deepspeed_configs/
│   │   ├── accelerate_configs/
│   │   ├── zero_experiments/
│   │   └── model_parallelism/
│   ├── week7-t5/
│   │   ├── t5_variants/
│   │   └── text2text_experiments/
│   └── week8-large-models/
│       ├── 3b_model_training/
│       ├── 7b_model_training/
│       └── full_dataset_experiments/
│
├── phase3-inference/                # Phase 3: Inference optimization (Weeks 9-10)
│   ├── week9-frameworks/
│   │   ├── vllm_experiments/
│   │   ├── tgi_experiments/
│   │   ├── tensorrt_llm/
│   │   └── benchmarks/
│   └── week10-quantization/
│       ├── fp8_quantization/
│       ├── gptq_experiments/
│       ├── awq_experiments/
│       └── deployment_configs/
│
├── phase4-finetuning/              # Phase 4: Finetuning & alignment (Weeks 11-14)
│   ├── week11-lora/
│   │   ├── peft_experiments/
│   │   ├── large_model_lora/
│   │   └── lora_configs/
│   ├── week12-qlora/
│   │   ├── full_precision_vs_qlora/
│   │   ├── efficiency_comparisons/
│   │   └── memory_optimization/
│   ├── week13-alignment/
│   │   ├── dpo_experiments/
│   │   ├── rlhf_experiments/
│   │   └── preference_optimization/
│   └── week14-instruct-models/
│       ├── 13b_instruct_models/
│       ├── 34b_instruct_models/
│       └── model_comparisons/
│
├── phase5-advanced/                # Phase 5: Advanced topics (Weeks 15-20)
│   ├── week15-moe/
│   │   ├── mixtral_experiments/
│   │   ├── custom_moe/
│   │   └── moe_training/
│   ├── week16-rag/
│   │   ├── llamaindex_experiments/
│   │   ├── langchain_experiments/
│   │   └── retrieval_systems/
│   ├── week17-18-multimodal/
│   │   ├── llava_experiments/
│   │   ├── flamingo_studies/
│   │   └── vision_language_models/
│   └── week19-20-production-rag/
│       ├── production_pipeline/
│       ├── large_model_serving/
│       └── end_to_end_system/
│
├── shared/                         # Shared utilities and common code
│   ├── __init__.py
│   ├── data/                       # Data processing utilities
│   │   ├── __init__.py
│   │   ├── tokenizers.py
│   │   ├── datasets.py
│   │   └── preprocessing.py
│   ├── models/                     # Model architectures and utilities
│   │   ├── __init__.py
│   │   ├── transformer_blocks.py
│   │   ├── attention.py
│   │   └── model_utils.py
│   ├── training/                   # Training utilities
│   │   ├── __init__.py
│   │   ├── distributed_utils.py
│   │   ├── optimization.py
│   │   └── checkpointing.py
│   ├── inference/                  # Inference utilities
│   │   ├── __init__.py
│   │   ├── generation.py
│   │   ├── quantization.py
│   │   └── serving.py
│   └── evaluation/                 # Evaluation and benchmarking
│       ├── __init__.py
│       ├── metrics.py
│       ├── benchmarks.py
│       └── analysis.py
│
├── datasets/                       # Dataset storage and management
│   ├── raw/                        # Raw datasets
│   ├── processed/                  # Preprocessed datasets
│   ├── tokenized/                  # Tokenized datasets
│   └── README.md                   # Dataset documentation
│
├── models/                         # Trained models and checkpoints
│   ├── phase1/                     # Models from Phase 1
│   ├── phase2/                     # Models from Phase 2
│   ├── phase3/                     # Optimized models from Phase 3
│   ├── phase4/                     # Fine-tuned models from Phase 4
│   ├── phase5/                     # Advanced models from Phase 5
│   └── README.md                   # Model documentation
│
├── configs/                        # Configuration files
│   ├── training/                   # Training configurations
│   ├── inference/                  # Inference configurations
│   ├── distributed/                # Distributed training configs
│   └── hardware/                   # Hardware-specific configs
│
├── scripts/                        # Utility scripts
│   ├── setup/                      # Environment setup scripts
│   ├── data/                       # Data download and processing
│   ├── training/                   # Training scripts
│   ├── inference/                  # Inference scripts
│   └── evaluation/                 # Evaluation scripts
│
├── notebooks/                      # Jupyter notebooks for exploration
│   ├── exploratory/                # Data exploration notebooks
│   ├── experiments/                # Experimental notebooks
│   └── analysis/                   # Analysis and visualization
│
└── tests/                          # Unit tests and integration tests
    ├── unit/                       # Unit tests
    ├── integration/                # Integration tests
    └── benchmarks/                 # Performance tests
```

## Key Design Principles

1. **Phase-based organization**: Each phase has its own directory for clear progression
2. **Shared utilities**: Common code in `shared/` to avoid duplication
3. **Separation of concerns**: Models, datasets, configs, and code are separated
4. **Scalability**: Structure supports large models and datasets
5. **Documentation**: Comprehensive docs and weekly logs for tracking progress
6. **Git LFS ready**: Structure anticipates large files (models, datasets)
7. **Hardware optimization**: Configs directory for H100-specific optimizations
