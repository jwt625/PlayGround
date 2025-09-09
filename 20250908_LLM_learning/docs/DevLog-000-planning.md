# LLM Training & Inference Learning Plan (3–6 Months)

## Phase 1: Fundamentals (Weeks 1–4)
**Goal:** Understand Transformer basics and build a small GPT.

- **Week 1**
  - Read: *Attention Is All You Need* (Vaswani et al., 2017)
  - Code: [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

- **Week 2**
  - Read: *Language Models are Unsupervised Multitask Learners (GPT-2)*  
  - Code: [minGPT](https://github.com/karpathy/minGPT)
  - Train GPT on small dataset (Shakespeare, toy wiki subset).

- **Week 3**
  - Read: *BERT: Pre-training of Deep Bidirectional Transformers* (2018)
  - Code: HuggingFace `transformers` tutorials (tokenizers, model training).

- **Week 4**
  - Project: Reproduce GPT training on a medium corpus using [nanoGPT](https://github.com/karpathy/nanoGPT).
  - Deliverable: A working GPT-like model generating coherent text.

---

## Phase 2: Scaling Up (Weeks 5–8)
**Goal:** Learn distributed training and scaling laws.

- **Week 5**
  - Read: *Scaling Laws for Neural Language Models* (Kaplan et al., 2020).
  - Experiment: Train progressively larger nanoGPT models.

- **Week 6**
  - Tools: [HuggingFace Accelerate](https://github.com/huggingface/accelerate), [DeepSpeed](https://github.com/microsoft/DeepSpeed).
  - Practice multi-GPU training.

- **Week 7**
  - Read: *T5: Exploring Transfer Learning with a Unified Text-to-Text Transformer*.
  - Experiment: Train a small T5 variant with HuggingFace.

- **Week 8**
  - Deliverable: Train a ~100M–500M parameter model on a curated dataset (subset of The Pile or RedPajama).

---

## Phase 3: Inference & Optimization (Weeks 9–10)
**Goal:** Efficient inference.

- **Week 9**
  - Explore inference frameworks:
    - [vLLM](https://github.com/vllm-project/vllm)
    - [text-generation-inference](https://github.com/huggingface/text-generation-inference)
  - Benchmark throughput and latency.

- **Week 10**
  - Study quantization methods:
    - [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
    - GPTQ, AWQ
  - Deliverable: Quantize and deploy an open model (LLaMA-2 7B, Mistral 7B).

---

## Phase 4: Finetuning & Alignment (Weeks 11–14)
**Goal:** Apply LoRA, QLoRA, and RLHF/DPO.

- **Week 11**
  - Read: *LoRA: Low-Rank Adaptation of Large Language Models*.
  - Code: [PEFT](https://github.com/huggingface/peft).
  - Fine-tune LLaMA-2 or Mistral using LoRA.

- **Week 12**
  - Read: *QLoRA: Efficient Finetuning of Quantized LLMs*.
  - Reproduce QLoRA finetuning on consumer GPU.

- **Week 13**
  - Read: *InstructGPT* (2022), *DPO* (2023).
  - Code: [TRL](https://github.com/huggingface/trl).
  - Implement preference optimization.

- **Week 14**
  - Deliverable: A fine-tuned instruct-style LLM.

---

## Phase 5: Advanced Topics (Weeks 15–20)
**Goal:** Explore state-of-the-art directions.

- **Week 15**
  - Read: *Switch Transformers: Scaling to Trillion Parameters* (MoE).
  - Explore [Mixtral](https://mistral.ai).

- **Week 16**
  - Study Retrieval-Augmented Generation:
    - Papers: REALM, RAG, RETRO.
    - Tools: [LlamaIndex](https://github.com/jerryjliu/llama_index), [LangChain](https://github.com/hwchase17/langchain).

- **Week 17–18**
  - Multimodal LLMs:
    - Read: Flamingo, LLaVA.
    - Code: [LLaVA](https://github.com/haotian-liu/LLaVA).

- **Week 19–20**
  - Project: Build a small RAG pipeline with a fine-tuned LLaMA/Mistral.
  - Deliverable: End-to-end system (retrieval + LLM inference).

---

## Continuous Habits
- Follow new papers: [arXiv-sanity](https://arxiv-sanity-lite.com/), HuggingFace blog.
- Engage in communities: HuggingFace Discord, EleutherAI.
- Track leaderboards: [Open LLM Leaderboard](https://huggingface.co/leaderboards).

---

## Deliverables by End of Journey
1. Small GPT-like model trained from scratch.
2. Mid-size model trained with distributed setup.
3. Inference-optimized deployment (quantized, vLLM).
4. LoRA/QLoRA finetuned instruct model.
5. Alignment experiments (DPO/RLHF).
6. End-to-end RAG + LLM pipeline.
