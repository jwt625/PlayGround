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
MAX_TOKENS = 10000
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
    client = OpenAI(api_key=KIMI_API_KEY or "dummy", base_url=KIMI_API_BASE)

    prompt = f"""Solve the following math problem. Show your reasoning and provide the final answer in \\boxed{{}} format.

Problem: {problem}"""

    logger.debug(f"Generating {n} completions...")
    response = client.chat.completions.create(
        model=KIMI_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        n=n,
        logprobs=True,
        top_logprobs=TOP_LOGPROBS
    )

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

    logger.debug(f"Generated {len(traces)} traces")
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
    logger.info(f"Evaluating problem {problem['id']}")
    logger.debug(f"Problem text: {problem['problem'][:100]}...")

    # Generate traces
    raw_traces = generate_traces(problem["problem"])
    traces = process_traces(raw_traces)

    num_with_answers = sum(1 for t in traces if t.answer)
    logger.info(f"Problem {problem['id']}: Extracted answers from {num_with_answers}/{len(traces)} traces")

    # Baseline: simple majority voting
    baseline_answer, baseline_votes = majority_voting(traces)

    results = {
        "problem_id": problem["id"],
        "ground_truth": problem["answer"],
        "num_traces": len(traces),
        "num_with_answers": sum(1 for t in traces if t.answer),
        "baseline": {
            "answer": baseline_answer,
            "votes": baseline_votes,
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

            results["deepconf"][metric][f"eta_{int(threshold*100)}"] = {
                "answer": answer,
                "weights": weights,
                "num_filtered": len(filtered),
                "correct": answer == problem["answer"] if answer else False
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

    # Evaluate each problem with progress bar
    all_results = []
    with tqdm(total=len(problems), desc="Evaluating problems", unit="problem") as pbar:
        for problem in problems:
            try:
                result = evaluate_problem(problem)
                all_results.append(result)
                pbar.update(1)
                pbar.set_postfix({"correct": sum(1 for r in all_results if r["baseline"]["correct"])})
            except Exception as e:
                logger.error(f"Error evaluating problem {problem['id']}: {e}", exc_info=True)
                pbar.update(1)
                continue

    # Compute aggregate statistics
    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)

    total = len(all_results)
    baseline_correct = sum(1 for r in all_results if r["baseline"]["correct"])

    logger.info(f"\nBaseline (Simple Majority Voting):")
    logger.info(f"  Accuracy: {baseline_correct}/{total} = {baseline_correct/total*100:.1f}%")

    logger.info(f"\nDeepConf (Confidence-Weighted Voting):")
    for metric in ["tail", "lowest_group", "bottom_10"]:
        logger.info(f"\n  Metric: {metric}")
        for threshold in FILTER_THRESHOLDS:
            key = f"eta_{int(threshold*100)}"
            correct = sum(1 for r in all_results if r["deepconf"][metric][key]["correct"])
            avg_filtered = sum(r["deepconf"][metric][key]["num_filtered"] for r in all_results) / total
            logger.info(f"    η={int(threshold*100)}%: {correct}/{total} = {correct/total*100:.1f}% (avg {avg_filtered:.1f} traces)")

    # Detailed results
    summary = {
        "total_problems": total,
        "baseline_accuracy": baseline_correct / total if total > 0 else 0,
        "deepconf_accuracy": {},
        "per_problem_results": all_results
    }

    for metric in ["tail", "lowest_group", "bottom_10"]:
        summary["deepconf_accuracy"][metric] = {}
        for threshold in FILTER_THRESHOLDS:
            key = f"eta_{int(threshold*100)}"
            correct = sum(1 for r in all_results if r["deepconf"][metric][key]["correct"])
            summary["deepconf_accuracy"][metric][key] = correct / total if total > 0 else 0

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

    # Run evaluation
    summary = run_evaluation(num_problems)

    # Save results with timestamp
    results_file = f"deepconf_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n" + "="*80)
    logger.info(f"Results saved to {results_file}")
    logger.info(f"Log saved to {log_filename}")
    logger.info("="*80)


if __name__ == "__main__":
    main()

