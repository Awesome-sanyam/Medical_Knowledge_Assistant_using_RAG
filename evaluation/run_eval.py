"""
Automated Academic Evaluation Suite
=====================================
Evaluates the Medical Knowledge Assistant against a held-out benchmark
using ROUGE-L scores and optionally BERTScore.

Usage:
    cd medical_assistant_project
    source venv/bin/activate
    pip install rouge-score
    python evaluation/run_eval.py
"""

import json
import os
import sys
import time
from datetime import datetime

# Ensure Django project is importable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core_project.settings")

import django

django.setup()

try:
    from rouge_score import rouge_scorer

    ROUGE_AVAILABLE = True
except ImportError:
    print("[WARNING] rouge-score not installed. Run: pip install rouge-score")
    ROUGE_AVAILABLE = False

from rag_engine.ollama_client import generate_response
from rag_engine.retriever import get_relevant_context

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_QUERIES_PATH = os.path.join(EVAL_DIR, "test_queries.json")
RESULTS_PATH = os.path.join(EVAL_DIR, "eval_results.json")


def evaluate_system(verbose: bool = True) -> dict:
    """
    Run the full evaluation pipeline against the held-out benchmark.

    Returns a summary dict with mean ROUGE-L, per-query scores, and metadata.
    """
    with open(TEST_QUERIES_PATH, "r") as f:
        test_cases = json.load(f)

    if verbose:
        print(f"\n{'='*60}")
        print("  Medical Knowledge Assistant — Evaluation Suite")
        print(f"  Running {len(test_cases)} benchmark queries")
        print(f"{'='*60}\n")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True) if ROUGE_AVAILABLE else None
    results = []

    for i, case in enumerate(test_cases, 1):
        query = case["query"]
        ground_truth = case["expected_answer"]

        if verbose:
            print(f"[{i}/{len(test_cases)}] Evaluating: {query[:65]}…")

        start = time.time()

        # Retrieve context
        try:
            context_chunks, low_confidence = get_relevant_context(query, top_k=3)
            chunk_texts = [c["content"] for c in context_chunks]
        except Exception as e:
            print(f"  ⚠️  Retrieval failed: {e}")
            context_chunks, chunk_texts, low_confidence = [], [], True

        # Generate answer
        try:
            generated_answer = generate_response(query, chunk_texts)
        except Exception as e:
            print(f"  ⚠️  Generation failed: {e}")
            generated_answer = ""

        latency = round(time.time() - start, 2)

        # Score
        rouge1, rougeL = 0.0, 0.0
        if scorer and generated_answer:
            scores = scorer.score(ground_truth, generated_answer)
            rouge1 = round(scores["rouge1"].fmeasure, 4)
            rougeL = round(scores["rougeL"].fmeasure, 4)

        result = {
            "query": query,
            "generated_answer": generated_answer,
            "ground_truth": ground_truth,
            "rouge1": rouge1,
            "rougeL": rougeL,
            "retrieved_chunks_count": len(context_chunks),
            "low_confidence": low_confidence,
            "latency_seconds": latency,
        }
        results.append(result)

        if verbose:
            status = "⚠️" if low_confidence else "✅"
            print(
                f"  {status} ROUGE-1: {rouge1:.4f} | ROUGE-L: {rougeL:.4f} | "
                f"Chunks: {len(context_chunks)} | Latency: {latency}s"
            )

    # Summary statistics
    if results:
        mean_rouge1 = round(sum(r["rouge1"] for r in results) / len(results), 4)
        mean_rougeL = round(sum(r["rougeL"] for r in results) / len(results), 4)
        mean_latency = round(sum(r["latency_seconds"] for r in results) / len(results), 2)
        low_conf_count = sum(1 for r in results if r["low_confidence"])
    else:
        mean_rouge1 = mean_rougeL = mean_latency = 0.0
        low_conf_count = 0

    summary = {
        "evaluation_timestamp": datetime.utcnow().isoformat() + "Z",
        "total_queries": len(results),
        "mean_rouge1": mean_rouge1,
        "mean_rougeL": mean_rougeL,
        "mean_latency_seconds": mean_latency,
        "low_confidence_queries": low_conf_count,
        "results": results,
    }

    # Persist results
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  ✅ Evaluation Complete")
        print(f"  Mean ROUGE-1 : {mean_rouge1:.4f}")
        print(f"  Mean ROUGE-L : {mean_rougeL:.4f}")
        print(f"  Mean Latency : {mean_latency}s")
        print(f"  Low-Conf.    : {low_conf_count}/{len(results)} queries")
        print(f"  Results saved → evaluation/eval_results.json")
        print(f"{'='*60}\n")

    return summary


if __name__ == "__main__":
    evaluate_system(verbose=True)
