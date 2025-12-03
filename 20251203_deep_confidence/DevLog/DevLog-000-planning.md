# Evaluation of Endpoint Capabilities and Client-Side Implementability of Deep Confidence

This report summarizes the feasibility of implementing the Deep Confidence (DeepConf) method described in the paper *“Deep Confidence: Confidence Estimation for Large Language Model Reasoning via Token Distribution Entropy” (arXiv:2508.15260)* using the tested vLLM-based inference endpoints. It integrates an assessment of endpoint capabilities with an explanation of which portions of DeepConf can be performed entirely on the client side.

---

## Background: What DeepConf Requires
DeepConf improves reliability in multi-sample LLM reasoning by:
1. Sampling multiple reasoning traces.
2. Computing per-token confidence from top-k log-probabilities.
3. Aggregating token confidences into trace-level stability measures (tail confidence, lowest-group confidence, bottom-10%).
4. Filtering low-quality traces and applying confidence-weighted majority voting.
5. Optionally implementing online early stopping (requires engine-level integration).

All steps except online early stopping can be executed client-side with standard vLLM APIs providing multiple completions and token logprobs.

---

# Endpoint Evaluation

## First Endpoint: [REDACTED_INTERNAL_ENDPOINT_1]

### Model 1: Llama-4-Maverick-17B-128E-Instruct-FP8
- Standard vLLM V1 API.
- Supports multiple completions and logprobs.
- No reasoning or thinking traces.
- High throughput for multiple completions.
- Suitable for majority voting or DeepConf-lite (logprob-based scoring only).

### Model 2: GLM-4.6-FP8
- Reasoning-aware vLLM V1 API.
- Exposes internal structured reasoning through `<think>` XML tags.
- Supports logprobs and multiple completions, but multi-sample generation is slow (practically limited to 2–3 completions).
- Useful for high-quality reasoning inspection and per-trace confidence analysis.
- Not ideal for high-volume voting due to latency constraints.

## Second Endpoint: [REDACTED_INTERNAL_ENDPOINT_2]

### Model: kimi-k2 (moonshotai/Kimi-K2-Thinking)
- Reasoning-aware vLLM V1 API.
- Provides separated `reasoning` and `reasoning_content` fields (cleanest structured reasoning among tested models).
- Fast multi-sample generation (5–10 completions).
- Supports logprobs and high-performance sampling.
- Large context window (262K tokens).
- Most suitable endpoint for production-grade majority voting and full client-side DeepConf.

---

# Implementability of DeepConf on the Client Side

## 1. Token-Level Confidence
All three models support `logprobs` and `top_logprobs`, allowing computation of token-level confidence:
\[
C_i = -\frac{1}{k}\sum_{j=1}^k \log P_i(j)
\]
This requires no server changes.

## 2. Trace-Level Confidence Metrics
Client code can compute:
- Tail confidence.
- Lowest group confidence (sliding windows).
- Bottom-10% confidence.

These metrics mirror the paper’s definitions and rely solely on returned logprobs.

## 3. Confidence-Based Filtering and Weighted Voting
With per-trace metrics, the client can:
- Filter to the top η percent of traces.
- Apply confidence-weighted majority voting on extracted answers.

All endpoints support this workflow.

---

# What Cannot Be Implemented Without Server Modifications

## Online Early Stopping (DeepConf-Low/High Online Mode)
Stopping low-confidence traces mid-generation requires the server to evaluate confidence per token during decoding. This demands modifications inside vLLM’s decoding loop, as described in the DeepConf paper. None of the tested endpoints expose such functionality.

---

# Summary and Recommendations

- All tested endpoints support the core requirements for **offline Deep Confidence**: multi-completion sampling, reasoning traces (where available), and token-level logprobs.
- **kimi-k2** is the best endpoint for production-scale majority voting and client-side DeepConf due to speed, clean reasoning separation, and scalability.
- **GLM-4.6** provides the richest structured reasoning but is slower for multi-sample use.
- **Llama-4-Maverick** is suitable for simple voting and confidence scoring but lacks reasoning fields.
- The only DeepConf feature not implementable client-side is **engine-level online early stopping**, which requires server modification.

Client-side DeepConf offers nearly all practical benefits described in the original paper and can be deployed immediately on top of current endpoints.


# DeepConf Paper — Direct Answers

## 1. Token-Level Confidence
- **Recommended k:** The paper never states a recommended k.  
  However, implementation uses `top_logprobs = 20`, meaning **k = 19**.
- **Is k=3 sufficient?** The paper does not test this. Its effective k≈20 suggests that **k=3 is noisier** and deviates from their setup.
- **Normalization/clipping:** None. Confidence is just the negative mean of top-k logprobs.

## 2. Trace-Level Metrics — Exact Definitions
### Tail Confidence
- Defined as average confidence over the **last 2048 tokens**.
- Not a percentage (e.g., not “last 20%”); it is a **fixed window**.

### Lowest Group Confidence
- Sliding window size: **2048 tokens**.  
- Stride: **1 token** (fully overlapping windows).  
- Metric: the **minimum** confidence across all sliding windows.

### Bottom-10% Confidence
- Compute group confidences over all 2048-token windows.
- Take the **lowest 10%** of these windows.
- Metric = **mean of those lowest 10%**, not the 10th percentile.

## 3. Filtering Strategy
- Only two η values are used:
  - **η = 10%** (keep top 10% most-confident traces; aggressive)
  - **η = 90%** (keep top 90%; conservative)
- No other η values are recommended.
- No metric combining is proposed.  
  - **Offline:** Tail Confidence or Bottom-10% Confidence perform best.  
  - **Online:** Lowest Group Confidence is used.

## 4. Confidence-Weighted Majority Voting
- Vote weight formula:
  \[
  V(a) = \sum_{t} C_t \cdot \mathbf{1}[answer(t)=a]
  \]
- **Weight = C_t**, not `C_t * vote_count`; the summation naturally accounts for counts.
- No alternate weighting scheme is described.

## 5. Answer Extraction
- Final answer must appear inside **`\boxed{...}`**.
- Extract the boxed content.
- Paper does **not** specify any normalization (e.g., numeric vs. textual formats). It assumes numeric answers inside the box.
