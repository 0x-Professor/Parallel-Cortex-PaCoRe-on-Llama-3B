#!/usr/bin/env python3
"""
Comprehensive benchmark runner for PaCoRe pipeline.

Evaluates the Parallel Cortex reasoning approach against various configurations
and provides detailed metrics to demonstrate competitive performance.

Usage:
    python examples/run_benchmark.py --samples 10 --verbose
    python examples/run_benchmark.py --full --compare
"""

import argparse
import sys
import os
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pacore_pipeline import PaCoRePipeline
from src.benchmark import (
    PaCoreBenchmark,
    create_math_benchmark_dataset,
    print_comparison_table,
    load_dataset,
)


def run_quick_validation(pipeline) -> bool:
    """Quick validation that the pipeline works."""
    print("\n" + "="*60)
    print("QUICK VALIDATION")
    print("="*60)
    
    test_cases = [
        ("What is 23 * 17?", "391"),
        ("What is 100 - 37?", "63"),
    ]
    
    passed = 0
    for problem, expected in test_cases:
        print(f"\nTest: {problem}")
        result = pipeline.run(problem)
        answer = result.get("final_answer")
        correct = str(answer).strip() == expected
        status = "✓ PASS" if correct else "✗ FAIL"
        print(f"  Expected: {expected}, Got: {answer} -> {status}")
        if correct:
            passed += 1
    
    print(f"\nValidation: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def run_full_benchmark(pipeline, args):
    """Run comprehensive benchmark evaluation."""
    print("\n" + "="*60)
    print("FULL BENCHMARK EVALUATION")
    print("="*60)
    
    benchmark = PaCoreBenchmark(pipeline, verbose=args.verbose)
    
    # Load or create dataset
    dataset = create_math_benchmark_dataset()
    if args.samples:
        dataset = dataset[:args.samples]
    
    print(f"\nDataset size: {len(dataset)} problems")
    
    if args.compare:
        # Compare different configurations
        configurations = [
            {"name": "Baseline (1 round, k=1)", "num_rounds": 1, "self_consistency_k": 1},
            {"name": "Multi-round (2 rounds)", "num_rounds": 2, "self_consistency_k": 1},
            {"name": "Self-consistency (k=3)", "num_rounds": 1, "self_consistency_k": 3},
            {"name": "Combined (2 rounds, k=3)", "num_rounds": 2, "self_consistency_k": 3},
        ]
        
        results = benchmark.compare_configurations(
            dataset,
            configurations,
            max_samples=args.samples,
        )
        print_comparison_table(results)
        
        # Find best configuration
        best_name = max(results.items(), key=lambda x: x[1].accuracy)[0]
        print(f"\n🏆 Best configuration: {best_name}")
        
        return results
    else:
        # Single configuration run
        summary = benchmark.run_dataset(
            dataset,
            num_rounds=args.rounds,
            self_consistency_k=args.consistency,
            max_samples=args.samples,
        )
        
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(f"  Total problems: {summary.total}")
        print(f"  Correct: {summary.correct}")
        print(f"  Accuracy: {summary.accuracy * 100:.1f}%")
        print(f"  Avg Latency: {summary.avg_latency_ms:.0f}ms")
        print(f"  Avg Confidence: {summary.avg_confidence * 100:.1f}%")
        print(f"  Consistency Rate: {summary.consistency_rate * 100:.1f}%")
        
        return summary


def generate_report(results, output_path: str):
    """Generate a detailed benchmark report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "method": "PaCoRe (Parallel Cortex Reasoning)",
        "results": {}
    }
    
    if isinstance(results, dict):
        for name, summary in results.items():
            report["results"][name] = summary.to_dict()
    else:
        report["results"]["default"] = results.to_dict()
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PaCoRe pipeline on math reasoning tasks"
    )
    parser.add_argument(
        "--samples", type=int, default=None,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--rounds", type=int, default=1,
        help="Number of refinement rounds (default: 1)"
    )
    parser.add_argument(
        "--consistency", type=int, default=1,
        help="Self-consistency k value (default: 1)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare multiple configurations"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick validation only"
    )
    parser.add_argument(
        "--report", type=str, default=None,
        help="Output path for JSON report"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("PaCoRe BENCHMARK RUNNER")
    print("Parallel Cortex Reasoning with Llama-3.2-3B")
    print("="*60)
    
    # Initialize pipeline
    print("\nLoading pipeline...")
    pipeline = PaCoRePipeline()
    print("Pipeline ready!")
    
    # Run validation
    if args.quick:
        success = run_quick_validation(pipeline)
        sys.exit(0 if success else 1)
    
    # Quick validation first
    if not run_quick_validation(pipeline):
        print("\n⚠ Validation failed! Check pipeline configuration.")
        sys.exit(1)
    
    # Run full benchmark
    results = run_full_benchmark(pipeline, args)
    
    # Generate report if requested
    if args.report:
        generate_report(results, args.report)
    
    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
