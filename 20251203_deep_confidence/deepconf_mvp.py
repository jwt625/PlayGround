#!/usr/bin/env python3
"""
DeepConf MVP Implementation
Implements the Deep Confidence method for LLM reasoning as described in arXiv:2508.15260
"""

import os
import re
import json
import math
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"deepconf_run_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
KIMI_API_BASE = os.getenv("KIMI_API_BASE")
KIMI_API_KEY = os.getenv("KIMI_API_KEY", "")
KIMI_MODEL_ID = os.getenv("KIMI_MODEL_ID", "kimi-k2")

if not KIMI_API_BASE:
    raise ValueError("KIMI_API_BASE environment variable is required")

# DeepConf parameters (from paper)
K = 19  # top_logprobs = 20, so k = 19
TOP_LOGPROBS = 20
N_COMPLETIONS = 10
TEMPERATURE = 0.8
MAX_TOKENS = 64000
WINDOW_SIZE = 2048  # For trace-level metrics
FILTER_THRESHOLDS = [0.1, 0.9]  # η = 10% and 90%


@dataclass
class Trace:
    """Represents a single reasoning trace with its confidence metrics"""
    content: str
    reasoning: str
    answer: Optional[str]
    token_confidences: List[float]
    tail_confidence: float
    lowest_group_confidence: float
    bottom_10_confidence: float
    
    def get_confidence(self, metric: str) -> float:
        """Get confidence by metric name"""
        if metric == "tail":
            return self.tail_confidence
        elif metric == "lowest_group":
            return self.lowest_group_confidence
        elif metric == "bottom_10":
            return self.bottom_10_confidence
        else:
            raise ValueError(f"Unknown metric: {metric}")


def load_aime_dataset() -> List[Dict[str, Any]]:
    """Load AIME 2025 dataset from HuggingFace (both I and II)"""
    logger.info("Loading AIME 2025 dataset...")
    problems = []

    # Load both AIME2025-I and AIME2025-II
    for config in ["AIME2025-I", "AIME2025-II"]:
        logger.info(f"Loading {config}...")
        dataset = load_dataset("opencompass/AIME2025", config, split="test")
        for idx, item in enumerate(dataset):
            problems.append({
                "problem": item["question"],  # Field is 'question' not 'problem'
                "answer": item["answer"],
                "id": f"{config}-{idx+1}"
            })

    logger.info(f"Loaded {len(problems)} problems total")
    return problems


def generate_traces(problem: str, n: int = N_COMPLETIONS) -> List[Dict[str, Any]]:
    """Generate n reasoning traces with logprobs for a given problem"""
    # Set a long timeout for large reasoning traces (30 minutes)
    # kimi-k2 with 64K max_tokens and n=10 can take a very long time
    client = OpenAI(
        api_key=KIMI_API_KEY or "dummy",
        base_url=KIMI_API_BASE,
        timeout=1800.0  # 30 minutes - adjust based on your server performance
    )

    prompt = f"""Solve the following math problem. Show your reasoning and provide the final answer in \\boxed{{}} format.

Problem: {problem}"""

    logger.debug(f"Generating {n} completions...")

    # Track timing
    start_time = time.time()
    start_timestamp = datetime.now().isoformat()

    response = client.chat.completions.create(
        model=KIMI_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        n=n,
        logprobs=True,
        top_logprobs=TOP_LOGPROBS
    )

    end_time = time.time()
    end_timestamp = datetime.now().isoformat()
    elapsed_time = end_time - start_time

    traces = []
    for choice in response.choices:
        content = choice.message.content or ""
        reasoning = getattr(choice.message, 'reasoning', '') or ""
        logprobs_data = choice.logprobs

        # Extract token-level logprobs
        token_logprobs = []
        if logprobs_data and hasattr(logprobs_data, 'content') and logprobs_data.content:
            for token_data in logprobs_data.content:
                if hasattr(token_data, 'top_logprobs') and token_data.top_logprobs:
                    # Get top-k logprobs (k=19, but we have 20 total)
                    top_k = token_data.top_logprobs[:K]
                    token_logprobs.append([t.logprob for t in top_k])

        traces.append({
            "content": content,
            "reasoning": reasoning,
            "token_logprobs": token_logprobs
        })

    # Add timing metadata to all traces
    timing_info = {
        "api_call_start": start_timestamp,
        "api_call_end": end_timestamp,
        "api_call_duration_seconds": elapsed_time
    }

    for trace in traces:
        trace.update(timing_info)

    logger.debug(f"Generated {len(traces)} traces in {elapsed_time:.2f}s")
    return traces


def compute_token_confidence(top_k_logprobs: List[float]) -> float:
    """
    Compute token-level confidence from top-k logprobs
    Formula: C_i = -1/k * Σ log P_i(j) for j=1 to k
    """
    if not top_k_logprobs:
        return 0.0
    return -sum(top_k_logprobs) / len(top_k_logprobs)


def compute_trace_metrics(token_confidences: List[float]) -> Tuple[float, float, float]:
    """
    Compute all three trace-level confidence metrics
    Returns: (tail_confidence, lowest_group_confidence, bottom_10_confidence)
    """
    if not token_confidences:
        return 0.0, 0.0, 0.0
    
    # Tail Confidence: average over last 2048 tokens
    tail_window = min(WINDOW_SIZE, len(token_confidences))
    tail_confidence = sum(token_confidences[-tail_window:]) / tail_window
    
    # Lowest Group Confidence: minimum over all sliding windows
    if len(token_confidences) <= WINDOW_SIZE:
        lowest_group_confidence = sum(token_confidences) / len(token_confidences)
    else:
        window_confidences = []
        for i in range(len(token_confidences) - WINDOW_SIZE + 1):
            window = token_confidences[i:i + WINDOW_SIZE]
            window_confidences.append(sum(window) / WINDOW_SIZE)
        lowest_group_confidence = min(window_confidences)
    
    # Bottom-10% Confidence: mean of lowest 10% of window confidences
    if len(token_confidences) <= WINDOW_SIZE:
        bottom_10_confidence = sum(token_confidences) / len(token_confidences)
    else:
        window_confidences = []
        for i in range(len(token_confidences) - WINDOW_SIZE + 1):
            window = token_confidences[i:i + WINDOW_SIZE]
            window_confidences.append(sum(window) / WINDOW_SIZE)
        # Get bottom 10%
        num_bottom = max(1, int(len(window_confidences) * 0.1))
        sorted_windows = sorted(window_confidences)
        bottom_10_confidence = sum(sorted_windows[:num_bottom]) / num_bottom

    return tail_confidence, lowest_group_confidence, bottom_10_confidence


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} format"""
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()  # Return last boxed answer
    return None


def process_traces(raw_traces: List[Dict[str, Any]]) -> List[Trace]:
    """Process raw traces to compute confidences and extract answers"""
    traces = []
    for raw_trace in raw_traces:
        # Compute token-level confidences
        token_confidences = []
        for top_k_logprobs in raw_trace["token_logprobs"]:
            conf = compute_token_confidence(top_k_logprobs)
            token_confidences.append(conf)

        # Compute trace-level metrics
        tail_conf, lowest_group_conf, bottom_10_conf = compute_trace_metrics(token_confidences)

        # Extract answer
        answer = extract_boxed_answer(raw_trace["content"])

        trace = Trace(
            content=raw_trace["content"],
            reasoning=raw_trace["reasoning"],
            answer=answer,
            token_confidences=token_confidences,
            tail_confidence=tail_conf,
            lowest_group_confidence=lowest_group_conf,
            bottom_10_confidence=bottom_10_conf
        )
        traces.append(trace)

    return traces


def filter_traces(traces: List[Trace], threshold: float, metric: str) -> List[Trace]:
    """
    Filter traces by confidence threshold
    threshold: η (e.g., 0.1 = keep top 10%, 0.9 = keep top 90%)
    metric: which confidence metric to use
    """
    if not traces:
        return []

    # Sort by confidence (descending)
    sorted_traces = sorted(traces, key=lambda t: t.get_confidence(metric), reverse=True)

    # Keep top η percent
    num_keep = max(1, int(len(sorted_traces) * threshold))
    return sorted_traces[:num_keep]


def majority_voting(traces: List[Trace]) -> Tuple[Optional[str], Dict[str, int]]:
    """Simple majority voting (baseline)"""
    if not traces:
        return None, {}

    vote_counts = defaultdict(int)
    for trace in traces:
        if trace.answer:
            vote_counts[trace.answer] += 1

    if not vote_counts:
        return None, {}

    winner = max(vote_counts.items(), key=lambda x: x[1])[0]
    return winner, dict(vote_counts)


def confidence_weighted_voting(traces: List[Trace], metric: str) -> Tuple[Optional[str], Dict[str, float]]:
    """
    Confidence-weighted majority voting
    Formula: V(a) = Σ C_t · 1[answer(t)=a]
    """
    if not traces:
        return None, {}

    vote_weights = defaultdict(float)
    for trace in traces:
        if trace.answer:
            confidence = trace.get_confidence(metric)
            vote_weights[trace.answer] += confidence

    if not vote_weights:
        return None, {}

    winner = max(vote_weights.items(), key=lambda x: x[1])[0]
    return winner, dict(vote_weights)


def evaluate_problem(problem: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a single problem with DeepConf"""
    problem_start_time = time.time()
    problem_start_timestamp = datetime.now().isoformat()

    logger.info(f"Evaluating problem {problem['id']}")
    logger.debug(f"Problem text: {problem['problem'][:100]}...")

    # Single-shot baseline (n=1)
    logger.debug("Generating single-shot baseline...")
    raw_single = generate_traces(problem["problem"], n=1)
    single_trace = process_traces(raw_single)[0] if raw_single else None

    # Extract timing info from raw trace
    single_timing = {}
    single_duration = None
    if raw_single:
        single_timing = {
            "api_call_start": raw_single[0].get("api_call_start"),
            "api_call_end": raw_single[0].get("api_call_end"),
            "api_call_duration_seconds": raw_single[0].get("api_call_duration_seconds")
        }
        single_duration = raw_single[0].get("api_call_duration_seconds")

    num_tokens_single = len(single_trace.token_confidences) if single_trace else 0

    single_shot_result = {
        "answer": single_trace.answer if single_trace else None,
        "correct": (single_trace.answer == problem["answer"]) if single_trace and single_trace.answer else False,
        "reasoning": single_trace.reasoning if single_trace else "",
        "content": single_trace.content if single_trace else "",
        "tail_confidence": single_trace.tail_confidence if single_trace else 0.0,
        "lowest_group_confidence": single_trace.lowest_group_confidence if single_trace else 0.0,
        "bottom_10_confidence": single_trace.bottom_10_confidence if single_trace else 0.0,
        "num_tokens": num_tokens_single,
        "token_confidences": single_trace.token_confidences if single_trace else [],
        "tokens_per_second": (num_tokens_single / single_duration) if single_duration and single_duration > 0 else None,
        **single_timing
    }

    # Multi-sample generation (n=10)
    logger.debug(f"Generating {N_COMPLETIONS} completions...")
    raw_traces = generate_traces(problem["problem"])
    traces = process_traces(raw_traces)

    num_with_answers = sum(1 for t in traces if t.answer)
    logger.info(f"Problem {problem['id']}: Single-shot={'✓' if single_shot_result['correct'] else '✗'}, Multi-sample: {num_with_answers}/{len(traces)} traces with answers")

    # Baseline: simple majority voting with full details
    baseline_answer, baseline_votes = majority_voting(traces)

    # Build detailed voting breakdown for baseline
    baseline_voting_details = []
    for i, trace in enumerate(traces):
        baseline_voting_details.append({
            "trace_index": i,
            "voted_for": trace.answer,
            "has_answer": trace.answer is not None
        })

    # Extract timing info from multi-sample traces (all have same API call timing)
    multi_timing = {}
    if raw_traces:
        multi_timing = {
            "api_call_start": raw_traces[0].get("api_call_start"),
            "api_call_end": raw_traces[0].get("api_call_end"),
            "api_call_duration_seconds": raw_traces[0].get("api_call_duration_seconds")
        }

    # Calculate tokens per second safely
    duration = multi_timing.get("api_call_duration_seconds")

    results = {
        "problem_id": problem["id"],
        "problem_text": problem["problem"],
        "ground_truth": problem["answer"],
        "num_traces": len(traces),
        "num_with_answers": sum(1 for t in traces if t.answer),
        "single_shot_baseline": single_shot_result,
        "traces": [
            {
                "trace_index": i,
                "reasoning": t.reasoning,
                "content": t.content,
                "extracted_answer": t.answer,
                "tail_confidence": t.tail_confidence,
                "lowest_group_confidence": t.lowest_group_confidence,
                "bottom_10_confidence": t.bottom_10_confidence,
                "num_tokens": len(t.token_confidences),
                "token_confidences": t.token_confidences,
                "tokens_per_second": (len(t.token_confidences) / duration) if duration and duration > 0 else None
            }
            for i, t in enumerate(traces)
        ],
        "multi_sample_timing": multi_timing,
        "baseline_majority_voting": {
            "answer": baseline_answer,
            "vote_counts": baseline_votes,
            "voting_details": baseline_voting_details,
            "correct": baseline_answer == problem["answer"] if baseline_answer else False
        },
        "deepconf": {}
    }

    # DeepConf with different metrics and thresholds
    for metric in ["tail", "lowest_group", "bottom_10"]:
        results["deepconf"][metric] = {}
        for threshold in FILTER_THRESHOLDS:
            filtered = filter_traces(traces, threshold, metric)
            answer, weights = confidence_weighted_voting(filtered, metric)

            # Build detailed voting breakdown for DeepConf
            deepconf_voting_details = []
            for trace in filtered:
                trace_idx = traces.index(trace)  # Find original index
                deepconf_voting_details.append({
                    "trace_index": trace_idx,
                    "voted_for": trace.answer,
                    "confidence_weight": trace.get_confidence(metric),
                    "has_answer": trace.answer is not None
                })

            results["deepconf"][metric][f"eta_{int(threshold*100)}"] = {
                "answer": answer,
                "vote_weights": weights,
                "voting_details": deepconf_voting_details,
                "num_filtered": len(filtered),
                "num_total": len(traces),
                "filtered_trace_indices": [traces.index(t) for t in filtered],
                "correct": answer == problem["answer"] if answer else False
            }

    # Add problem-level timing
    problem_end_time = time.time()
    problem_end_timestamp = datetime.now().isoformat()
    problem_duration = problem_end_time - problem_start_time

    results["timing"] = {
        "problem_start": problem_start_timestamp,
        "problem_end": problem_end_timestamp,
        "problem_duration_seconds": problem_duration
    }

    return results


def run_evaluation(num_problems: Optional[int] = None) -> Dict[str, Any]:
    """Run full evaluation on AIME 2025 dataset"""
    logger.info("="*80)
    logger.info("DeepConf MVP Evaluation")
    logger.info("="*80)
    logger.info(f"Log file: {log_filename}")
    logger.info(f"Configuration: n={N_COMPLETIONS}, temp={TEMPERATURE}, max_tokens={MAX_TOKENS}, top_logprobs={TOP_LOGPROBS}")

    # Load dataset
    problems = load_aime_dataset()
    if num_problems:
        problems = problems[:num_problems]
        logger.info(f"Evaluating on first {num_problems} problems")
    else:
        logger.info(f"Evaluating on all {len(problems)} problems")

    # Results file with timestamp
    results_file = f"deepconf_results_{timestamp}.json"
    logger.info(f"Results will be saved to: {results_file}")

    # Evaluate each problem with progress bar
    all_results = []
    with tqdm(total=len(problems), desc="Evaluating problems", unit="problem") as pbar:
        for problem in problems:
            try:
                result = evaluate_problem(problem)
                all_results.append(result)

                # Save results incrementally after each problem
                summary = {
                    "config": {
                        "n_completions": N_COMPLETIONS,
                        "temperature": TEMPERATURE,
                        "max_tokens": MAX_TOKENS,
                        "top_logprobs": TOP_LOGPROBS,
                        "k": K,
                        "window_size": WINDOW_SIZE,
                        "filter_thresholds": FILTER_THRESHOLDS
                    },
                    "num_problems_evaluated": len(all_results),
                    "num_problems_total": len(problems),
                    "results": all_results
                }
                with open(results_file, "w") as f:
                    json.dump(summary, f, indent=2)

                pbar.update(1)
                pbar.set_postfix({"correct": sum(1 for r in all_results if r["baseline_majority_voting"]["correct"])})
            except Exception as e:
                logger.error(f"Error evaluating problem {problem['id']}: {e}", exc_info=True)
                pbar.update(1)
                continue

    # Compute aggregate statistics
    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)

    total = len(all_results)

    # Single-shot baseline
    single_shot_correct = sum(1 for r in all_results if r["single_shot_baseline"]["correct"])
    logger.info(f"\nSingle-Shot Baseline (n=1):")
    logger.info(f"  Accuracy: {single_shot_correct}/{total} = {single_shot_correct/total*100:.1f}%")

    # Multi-sample majority voting baseline
    baseline_correct = sum(1 for r in all_results if r["baseline_majority_voting"]["correct"])
    logger.info(f"\nMulti-Sample Baseline (Simple Majority Voting, n={N_COMPLETIONS}):")
    logger.info(f"  Accuracy: {baseline_correct}/{total} = {baseline_correct/total*100:.1f}%")

    logger.info(f"\nDeepConf (Confidence-Weighted Voting, n={N_COMPLETIONS}):")
    for metric in ["tail", "lowest_group", "bottom_10"]:
        logger.info(f"\n  Metric: {metric}")
        for threshold in FILTER_THRESHOLDS:
            key = f"eta_{int(threshold*100)}"
            correct = sum(1 for r in all_results if r["deepconf"][metric][key]["correct"])
            avg_filtered = sum(r["deepconf"][metric][key]["num_filtered"] for r in all_results) / total
            logger.info(f"    η={int(threshold*100)}%: {correct}/{total} = {correct/total*100:.1f}% (avg {avg_filtered:.1f} traces)")

    # Detailed results (already saved incrementally)
    summary = {
        "total_problems": total,
        "single_shot_accuracy": single_shot_correct / total if total > 0 else 0,
        "baseline_majority_accuracy": baseline_correct / total if total > 0 else 0,
        "deepconf_accuracy": {},
        "per_problem_results": all_results
    }

    for metric in ["tail", "lowest_group", "bottom_10"]:
        summary["deepconf_accuracy"][metric] = {}
        for threshold in FILTER_THRESHOLDS:
            key = f"eta_{int(threshold*100)}"
            correct = sum(1 for r in all_results if r["deepconf"][metric][key]["correct"])
            summary["deepconf_accuracy"][metric][key] = correct / total if total > 0 else 0

    # Save final version with aggregate stats
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    """Main entry point"""
    import sys

    # Parse command line args
    num_problems = None
    if len(sys.argv) > 1:
        try:
            num_problems = int(sys.argv[1])
        except ValueError:
            logger.error(f"Usage: {sys.argv[0]} [num_problems]")
            sys.exit(1)

    # Run evaluation (results saved incrementally)
    summary = run_evaluation(num_problems)

    # Results already saved incrementally
    results_file = f"deepconf_results_{timestamp}.json"

    logger.info(f"\n" + "="*80)
    logger.info(f"Results saved to {results_file}")
    logger.info(f"Log saved to {log_filename}")
    logger.info("="*80)


if __name__ == "__main__":
    main()

