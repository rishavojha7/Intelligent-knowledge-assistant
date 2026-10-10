"""
Evaluation script for the Knowledge Assistant.

Runs the evaluation question set against the system and computes
metrics for retrieval quality, answer quality, and workflow routing.

Usage:
    python -m scripts.evaluate [--output evaluation_results.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.agent.graph import run_query
from src.config import get_settings


def load_eval_set(path: Path) -> list[dict]:
    """Load the evaluation question set."""
    with open(path) as f:
        data = json.load(f)
    return data["evaluation_set"]["questions"]


def evaluate_single(question_data: dict) -> dict:
    """Evaluate a single question and return results."""
    question = question_data["question"]
    expected_workflow = question_data.get("expected_workflow", "")
    expected_sources = question_data.get("expected_sources", [])
    expected_contains = question_data.get("expected_answer_contains", [])
    category = question_data.get("category", "unknown")

    start_time = time.time()
    try:
        result = run_query(question=question)
        elapsed = time.time() - start_time

        # Check workflow routing accuracy
        route_correct = result.workflow == expected_workflow if expected_workflow else None

        # Check source coverage
        actual_sources = {s.document for s in result.sources}
        source_hits = sum(1 for s in expected_sources if s in actual_sources)
        source_recall = source_hits / len(expected_sources) if expected_sources else 1.0

        # Check answer content coverage
        answer_lower = result.answer.lower()
        content_hits = sum(1 for term in expected_contains if term.lower() in answer_lower)
        content_recall = content_hits / len(expected_contains) if expected_contains else 1.0

        # Check for hallucination markers (for unanswerable questions)
        graceful_refusal = False
        if category == "unanswerable":
            refusal_phrases = [
                "don't have enough information",
                "not enough information",
                "cannot answer",
                "no relevant information",
                "not covered",
                "outside the scope",
                "not available in",
            ]
            graceful_refusal = any(p in answer_lower for p in refusal_phrases)

        return {
            "id": question_data["id"],
            "category": category,
            "question": question,
            "answer": result.answer[:500],  # Truncate for report
            "workflow_used": result.workflow,
            "workflow_expected": expected_workflow,
            "route_correct": route_correct,
            "sources_found": list(actual_sources),
            "sources_expected": expected_sources,
            "source_recall": round(source_recall, 3),
            "content_recall": round(content_recall, 3),
            "confidence": result.confidence,
            "reasoning_steps": result.reasoning_steps,
            "graceful_refusal": graceful_refusal if category == "unanswerable" else None,
            "latency_seconds": round(elapsed, 2),
            "error": None,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "id": question_data["id"],
            "category": category,
            "question": question,
            "answer": "",
            "error": str(e),
            "latency_seconds": round(elapsed, 2),
            "route_correct": None,
            "source_recall": 0.0,
            "content_recall": 0.0,
            "graceful_refusal": None,
        }


def compute_aggregate_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics across all evaluation results."""
    total = len(results)
    errors = sum(1 for r in results if r.get("error"))
    successful = [r for r in results if not r.get("error")]

    # Routing accuracy
    route_checks = [r for r in successful if r.get("route_correct") is not None]
    route_accuracy = (
        sum(1 for r in route_checks if r["route_correct"]) / len(route_checks)
        if route_checks else 0
    )

    # Average source recall
    source_recalls = [r["source_recall"] for r in successful if "source_recall" in r]
    avg_source_recall = sum(source_recalls) / len(source_recalls) if source_recalls else 0

    # Average content recall
    content_recalls = [r["content_recall"] for r in successful if "content_recall" in r]
    avg_content_recall = sum(content_recalls) / len(content_recalls) if content_recalls else 0

    # Graceful refusal rate for unanswerable questions
    unanswerable = [r for r in successful if r["category"] == "unanswerable"]
    refusal_rate = (
        sum(1 for r in unanswerable if r.get("graceful_refusal"))
        / len(unanswerable)
        if unanswerable else 0
    )

    # Average latency
    latencies = [r["latency_seconds"] for r in successful]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    # Per-category breakdown
    categories = {}
    for r in successful:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"count": 0, "content_recall": [], "source_recall": []}
        categories[cat]["count"] += 1
        categories[cat]["content_recall"].append(r.get("content_recall", 0))
        categories[cat]["source_recall"].append(r.get("source_recall", 0))

    for cat, data in categories.items():
        data["avg_content_recall"] = round(
            sum(data["content_recall"]) / len(data["content_recall"]), 3
        )
        data["avg_source_recall"] = round(
            sum(data["source_recall"]) / len(data["source_recall"]), 3
        )
        del data["content_recall"]
        del data["source_recall"]

    return {
        "total_questions": total,
        "successful": len(successful),
        "errors": errors,
        "routing_accuracy": round(route_accuracy, 3),
        "avg_source_recall": round(avg_source_recall, 3),
        "avg_content_recall": round(avg_content_recall, 3),
        "unanswerable_refusal_rate": round(refusal_rate, 3),
        "avg_latency_seconds": round(avg_latency, 2),
        "per_category": categories,
    }


def print_report(metrics: dict, results: list[dict]) -> None:
    """Print a human-readable evaluation report."""
    print("\n" + "=" * 70)
    print("  EVALUATION REPORT — Intelligent Knowledge Assistant")
    print("=" * 70)

    print(f"\n📊 Aggregate Metrics:")
    print(f"   Total questions     : {metrics['total_questions']}")
    print(f"   Successful          : {metrics['successful']}")
    print(f"   Errors              : {metrics['errors']}")
    print(f"   Routing accuracy    : {metrics['routing_accuracy']:.1%}")
    print(f"   Avg source recall   : {metrics['avg_source_recall']:.1%}")
    print(f"   Avg content recall  : {metrics['avg_content_recall']:.1%}")
    print(f"   Refusal rate (N/A)  : {metrics['unanswerable_refusal_rate']:.1%}")
    print(f"   Avg latency         : {metrics['avg_latency_seconds']:.2f}s")

    print(f"\n📂 Per-Category Breakdown:")
    for cat, data in metrics["per_category"].items():
        print(f"   {cat:15s} : {data['count']} questions | "
              f"content recall: {data['avg_content_recall']:.1%} | "
              f"source recall: {data['avg_source_recall']:.1%}")

    print(f"\n{'─' * 70}")
    print("Per-Question Results:")
    print(f"{'─' * 70}")
    for r in results:
        status = "✅" if not r.get("error") else "❌"
        route_mark = "✓" if r.get("route_correct") else "✗" if r.get("route_correct") is False else "?"
        print(f"  {status} [{r['id']:15s}] route={route_mark} "
              f"src_recall={r.get('source_recall', 0):.0%} "
              f"content={r.get('content_recall', 0):.0%} "
              f"latency={r.get('latency_seconds', 0):.1f}s")
        if r.get("error"):
            print(f"     ⚠️  Error: {r['error'][:80]}")

    print(f"\n{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Knowledge Assistant")
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for detailed results (default: evaluation_results.json)",
    )
    parser.add_argument(
        "--eval-set",
        type=str,
        default="configs/eval_questions.json",
        help="Path to evaluation question set",
    )
    args = parser.parse_args()

    print("Loading evaluation set...")
    eval_set = load_eval_set(Path(args.eval_set))
    print(f"Found {len(eval_set)} evaluation questions\n")

    print("Running evaluations...")
    results = []
    for i, q in enumerate(eval_set, 1):
        print(f"  [{i}/{len(eval_set)}] {q['id']}: {q['question'][:60]}...")
        result = evaluate_single(q)
        results.append(result)

    metrics = compute_aggregate_metrics(results)
    print_report(metrics, results)

    # Save detailed results
    output_path = Path(args.output)
    output = {"metrics": metrics, "results": results}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"💾 Detailed results saved to {output_path}")


if __name__ == "__main__":
    main()
