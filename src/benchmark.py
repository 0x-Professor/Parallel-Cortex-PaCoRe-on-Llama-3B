#!/usr/bin/env python3
"""
Benchmark module for evaluating PaCoRe pipeline on standard datasets.

Supports:
- GSM8K-style math problems
- Custom evaluation datasets
- Multi-metric evaluation (accuracy, consistency, confidence)
- Comparison with baseline models
"""

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    problem: str
    expected: str
    predicted: Optional[str]
    correct: bool
    latency_ms: float
    confidence: float = 0.0
    num_rounds: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark results."""
    total: int
    correct: int
    accuracy: float
    avg_latency_ms: float
    avg_confidence: float
    consistency_rate: float  # How often self-consistency agrees
    results: List[BenchmarkResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy * 100, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "avg_confidence": round(self.avg_confidence * 100, 2),
            "consistency_rate": round(self.consistency_rate * 100, 2),
        }


def normalize_answer(answer: Optional[str]) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    ans = str(answer).strip().lower()
    # Remove common prefixes
    for prefix in ["$", "£", "€", "the answer is", "answer:", "="]:
        ans = ans.lstrip(prefix).strip()
    # Remove trailing punctuation
    ans = ans.rstrip(".,;:!?")
    # Remove commas in numbers
    ans = ans.replace(",", "")
    # Normalize fractions
    ans = re.sub(r"(\d+)/(\d+)", lambda m: str(float(m.group(1))/float(m.group(2))), ans)
    return ans


def answers_match(predicted: Optional[str], expected: str, tolerance: float = 0.001) -> bool:
    """Check if predicted answer matches expected answer."""
    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)
    
    if pred_norm == exp_norm:
        return True
    
    # Try numeric comparison
    try:
        pred_num = float(pred_norm)
        exp_num = float(exp_norm)
        return abs(pred_num - exp_num) <= tolerance * max(1, abs(exp_num))
    except (ValueError, TypeError):
        pass
    
    # Check if one contains the other
    if pred_norm in exp_norm or exp_norm in pred_norm:
        return True
    
    return False


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load evaluation dataset from JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def create_math_benchmark_dataset() -> List[Dict[str, Any]]:
    """Create a diverse math benchmark dataset for evaluation."""
    return [
        # Basic arithmetic
        {"problem": "What is 23 * 17?", "answer": "391"},
        {"problem": "Calculate 144 / 12", "answer": "12"},
        {"problem": "What is 987 - 654?", "answer": "333"},
        {"problem": "Compute 256 + 789", "answer": "1045"},
        
        # Multi-step problems
        {"problem": "A store sells apples for $2 each. If you buy 15 apples and pay with a $50 bill, how much change do you get?", "answer": "20"},
        {"problem": "A train travels 60 miles per hour for 2.5 hours. How many miles does it travel?", "answer": "150"},
        {"problem": "If a rectangle has length 8 and width 5, what is its perimeter?", "answer": "26"},
        {"problem": "A pizza is cut into 8 slices. If 3 people each eat 2 slices, how many slices are left?", "answer": "2"},
        
        # Word problems
        {"problem": "John has 3 times as many marbles as Mary. If Mary has 12 marbles, how many do they have together?", "answer": "48"},
        {"problem": "A bookshelf has 5 shelves with 8 books each. If 7 books are removed, how many remain?", "answer": "33"},
        {"problem": "If a car uses 4 gallons of gas for every 120 miles, how many gallons are needed for 300 miles?", "answer": "10"},
        {"problem": "A recipe calls for 2/3 cup of sugar. If you want to make 3 batches, how many cups of sugar do you need?", "answer": "2"},
        
        # Percentage problems
        {"problem": "What is 25% of 80?", "answer": "20"},
        {"problem": "If a shirt costs $40 and is 30% off, what is the sale price?", "answer": "28"},
        {"problem": "A student scored 85 out of 100. What percentage is this?", "answer": "85"},
        
        # Algebra-style
        {"problem": "If 3x + 7 = 22, what is x?", "answer": "5"},
        {"problem": "Solve for y: 2y - 4 = 10", "answer": "7"},
        {"problem": "What number, when multiplied by 6 and then adding 9, gives 45?", "answer": "6"},
        
        # Sequences
        {"problem": "What is the next number in the sequence: 2, 4, 8, 16, ?", "answer": "32"},
        {"problem": "Find the pattern and continue: 1, 4, 9, 16, 25, ?", "answer": "36"},
    ]


class PaCoreBenchmark:
    """Benchmark runner for PaCoRe pipeline."""
    
    def __init__(self, pipeline, verbose: bool = True):
        """
        Initialize benchmark runner.
        
        Args:
            pipeline: PaCoRePipeline instance
            verbose: Whether to print progress
        """
        self.pipeline = pipeline
        self.verbose = verbose
    
    def run_single(
        self,
        problem: str,
        expected: str,
        num_rounds: int = 1,
        self_consistency_k: int = 1,
    ) -> BenchmarkResult:
        """Run benchmark on a single problem."""
        start_time = time.time()
        
        result = self.pipeline.run(
            problem,
            num_rounds=num_rounds,
            self_consistency_k=self_consistency_k,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        predicted = result.get("final_answer")
        correct = answers_match(predicted, expected)
        confidence = result.get("vote_confidence", 1.0 if correct else 0.0)
        
        return BenchmarkResult(
            problem=problem,
            expected=expected,
            predicted=str(predicted) if predicted else None,
            correct=correct,
            latency_ms=latency_ms,
            confidence=confidence,
            num_rounds=num_rounds,
            metadata={
                "self_consistency_k": self_consistency_k,
                "all_answers": result.get("all_answers"),
            }
        )
    
    def run_dataset(
        self,
        dataset: List[Dict[str, Any]],
        num_rounds: int = 1,
        self_consistency_k: int = 1,
        problem_key: str = "problem",
        answer_key: str = "answer",
        max_samples: Optional[int] = None,
        shuffle: bool = False,
    ) -> BenchmarkSummary:
        """
        Run benchmark on a dataset.
        
        Args:
            dataset: List of {problem, answer} dicts
            num_rounds: Number of refinement rounds per problem
            self_consistency_k: Number of attempts for self-consistency voting
            problem_key: Key for problem text in dataset
            answer_key: Key for expected answer in dataset
            max_samples: Maximum number of samples to evaluate (None = all)
            shuffle: Whether to shuffle dataset before evaluation
        
        Returns:
            BenchmarkSummary with all results
        """
        if shuffle:
            dataset = random.sample(dataset, len(dataset))
        
        if max_samples:
            dataset = dataset[:max_samples]
        
        results: List[BenchmarkResult] = []
        correct_count = 0
        total_latency = 0.0
        total_confidence = 0.0
        consistency_agreements = 0
        
        for i, item in enumerate(dataset):
            problem = item[problem_key]
            expected = str(item[answer_key])
            
            if self.verbose:
                print(f"\n[{i+1}/{len(dataset)}] {problem[:60]}...")
            
            result = self.run_single(
                problem, expected, num_rounds, self_consistency_k
            )
            results.append(result)
            
            if result.correct:
                correct_count += 1
            total_latency += result.latency_ms
            total_confidence += result.confidence
            
            # Check self-consistency agreement
            if self_consistency_k > 1 and result.metadata.get("all_answers"):
                all_ans = result.metadata["all_answers"]
                if all_ans and len(set(all_ans)) == 1:
                    consistency_agreements += 1
            else:
                consistency_agreements += 1  # Single attempt = 100% agreement
            
            if self.verbose:
                status = "✓" if result.correct else "✗"
                print(f"   {status} Predicted: {result.predicted} | Expected: {expected}")
        
        n = len(results)
        return BenchmarkSummary(
            total=n,
            correct=correct_count,
            accuracy=correct_count / n if n > 0 else 0.0,
            avg_latency_ms=total_latency / n if n > 0 else 0.0,
            avg_confidence=total_confidence / n if n > 0 else 0.0,
            consistency_rate=consistency_agreements / n if n > 0 else 0.0,
            results=results,
        )
    
    def compare_configurations(
        self,
        dataset: List[Dict[str, Any]],
        configurations: List[Dict[str, Any]],
        problem_key: str = "problem",
        answer_key: str = "answer",
        max_samples: Optional[int] = None,
    ) -> Dict[str, BenchmarkSummary]:
        """
        Compare different configurations on the same dataset.
        
        Args:
            dataset: Evaluation dataset
            configurations: List of {"name": str, "num_rounds": int, "self_consistency_k": int}
            
        Returns:
            Dict mapping configuration names to their results
        """
        results = {}
        
        for config in configurations:
            name = config.get("name", f"rounds={config.get('num_rounds', 1)}, k={config.get('self_consistency_k', 1)}")
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Running configuration: {name}")
                print(f"{'='*60}")
            
            summary = self.run_dataset(
                dataset,
                num_rounds=config.get("num_rounds", 1),
                self_consistency_k=config.get("self_consistency_k", 1),
                problem_key=problem_key,
                answer_key=answer_key,
                max_samples=max_samples,
            )
            results[name] = summary
            
            if self.verbose:
                print(f"\n{name} Results:")
                print(f"  Accuracy: {summary.accuracy*100:.1f}%")
                print(f"  Avg Latency: {summary.avg_latency_ms:.0f}ms")
                print(f"  Consistency: {summary.consistency_rate*100:.1f}%")
        
        return results


def print_comparison_table(results: Dict[str, BenchmarkSummary]):
    """Print a comparison table of benchmark results."""
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON")
    print("="*80)
    print(f"{'Configuration':<30} {'Accuracy':<12} {'Latency':<12} {'Consistency':<12}")
    print("-"*80)
    
    for name, summary in results.items():
        print(f"{name:<30} {summary.accuracy*100:>6.1f}%     {summary.avg_latency_ms:>8.0f}ms   {summary.consistency_rate*100:>6.1f}%")
    
    print("="*80)


if __name__ == "__main__":
    # Demo usage
    print("PaCoRe Benchmark Module")
    print("="*40)
    print("\nTo run benchmarks, use:")
    print("  from src.benchmark import PaCoreBenchmark, create_math_benchmark_dataset")
    print("  benchmark = PaCoreBenchmark(pipeline)")
    print("  results = benchmark.run_dataset(create_math_benchmark_dataset())")
    print("\nOr compare configurations:")
    print("  configs = [")
    print("    {'name': 'baseline', 'num_rounds': 1, 'self_consistency_k': 1},")
    print("    {'name': 'multi-round', 'num_rounds': 2, 'self_consistency_k': 1},")
    print("    {'name': 'self-consistency', 'num_rounds': 1, 'self_consistency_k': 3},")
    print("  ]")
    print("  results = benchmark.compare_configurations(dataset, configs)")
