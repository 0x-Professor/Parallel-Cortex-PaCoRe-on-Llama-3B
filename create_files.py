"""
Creates all source files for the PaCoRe project
Run this script to generate the complete project structure
"""

import os
from pathlib import Path

# Source files content
FILES = {
    'src/__init__.py': '''"""
PaCoRe: Parallel Collaborative Reasoning Framework
"""

__version__ = "0.1.0"
__author__ = "PDC Project"

from .consensus import ConsensusEngine

__all__ = ["ConsensusEngine"]
''',

    'src/consensus.py': '''"""
PaCoRe Consensus Engine
Implements various consensus algorithms for aggregating model responses.
"""

from typing import List, Dict, Any
from collections import Counter
import numpy as np
from loguru import logger

class ConsensusEngine:
    """
    Implements multiple consensus strategies for collaborative reasoning
    """
    
    def __init__(self, strategy: str = "weighted_voting", threshold: float = 0.7):
        self.strategy = strategy
        self.threshold = threshold
        self.strategies = {
            "majority_voting": self.majority_voting,
            "weighted_voting": self.weighted_voting,
            "ensemble": self.ensemble_average,
            "ranking": self.ranking_based,
            "borda_count": self.borda_count
        }
    
    def aggregate(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply the configured consensus strategy"""
        if not responses:
            return {"text": "", "confidence": 0.0, "method": self.strategy}
        
        strategy_func = self.strategies.get(self.strategy, self.weighted_voting)
        result = strategy_func(responses)
        result["method"] = self.strategy
        
        return result
    
    def majority_voting(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simple majority voting - most common response wins"""
        response_texts = [r.get('response', '') for r in responses]
        
        if not response_texts:
            return {"text": "", "confidence": 0.0}
        
        counter = Counter(response_texts)
        most_common = counter.most_common(1)[0]
        confidence = most_common[1] / len(responses)
        
        return {
            "text": most_common[0],
            "confidence": confidence,
            "votes": dict(counter)
        }
    
    def weighted_voting(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Weighted voting based on model confidence scores"""
        if not responses:
            return {"text": "", "confidence": 0.0}
        
        weighted_responses = {}
        total_weight = 0
        
        for r in responses:
            text = r.get('response', '')
            conf = r.get('confidence', 0.5)
            
            if text in weighted_responses:
                weighted_responses[text] += conf
            else:
                weighted_responses[text] = conf
            
            total_weight += conf
        
        if not weighted_responses:
            return {"text": "", "confidence": 0.0}
        
        best_response = max(weighted_responses.items(), key=lambda x: x[1])
        normalized_confidence = best_response[1] / total_weight if total_weight > 0 else 0
        
        return {
            "text": best_response[0],
            "confidence": normalized_confidence,
            "weights": weighted_responses
        }
    
    def ensemble_average(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensemble averaging - combine all responses with confidence weighting"""
        if not responses:
            return {"text": "", "confidence": 0.0}
        
        sorted_responses = sorted(
            responses,
            key=lambda r: r.get('confidence', 0),
            reverse=True
        )
        
        filtered = [
            r for r in sorted_responses 
            if r.get('confidence', 0) >= self.threshold
        ]
        
        if not filtered:
            filtered = sorted_responses[:1]
        
        combined_text = " ".join([r.get('response', '') for r in filtered])
        avg_confidence = np.mean([r.get('confidence', 0) for r in filtered])
        
        return {
            "text": combined_text,
            "confidence": float(avg_confidence),
            "num_combined": len(filtered)
        }
    
    def ranking_based(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ranking-based consensus - rank responses and select top-ranked"""
        if not responses:
            return {"text": "", "confidence": 0.0}
        
        scored_responses = []
        
        for r in responses:
            score = 0
            score += r.get('confidence', 0) * 0.5
            
            response_len = len(r.get('response', ''))
            normalized_len = min(response_len / 500, 1.0)
            score += normalized_len * 0.3
            
            unique_words = len(set(r.get('response', '').split()))
            score += min(unique_words / 100, 1.0) * 0.2
            
            scored_responses.append((r, score))
        
        best = max(scored_responses, key=lambda x: x[1])
        
        return {
            "text": best[0].get('response', ''),
            "confidence": best[1],
            "rank_score": best[1]
        }
    
    def borda_count(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Borda count voting - each response ranks others, points awarded"""
        if not responses:
            return {"text": "", "confidence": 0.0}
        
        n = len(responses)
        points = {i: 0 for i in range(n)}
        
        sorted_indices = sorted(
            range(n),
            key=lambda i: responses[i].get('confidence', 0),
            reverse=True
        )
        
        for rank, idx in enumerate(sorted_indices):
            points[idx] = n - rank - 1
        
        winner_idx = max(points.items(), key=lambda x: x[1])[0]
        winner = responses[winner_idx]
        
        confidence = points[winner_idx] / (n * (n - 1) / 2) if n > 1 else 1.0
        
        return {
            "text": winner.get('response', ''),
            "confidence": confidence,
            "borda_points": points[winner_idx]
        }
''',

    'examples/simple_usage.py': '''"""
Simple usage example of PaCoRe framework
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from consensus import ConsensusEngine

def example_consensus():
    """Example: Using consensus engine to aggregate multiple model responses"""
    
    responses = [
        {
            "response": "Parallel computing is a type of computation where many calculations "
                       "are carried out simultaneously on multiple processors.",
            "confidence": 0.92,
            "model": "phi-2"
        },
        {
            "response": "Parallel computing involves breaking down large problems into smaller "
                       "tasks that can be executed concurrently on multiple processors.",
            "confidence": 0.88,
            "model": "tinyllama"
        },
        {
            "response": "Parallel computing is a type of computation where many calculations "
                       "are carried out simultaneously on multiple processors.",
            "confidence": 0.90,
            "model": "opt-1.3b"
        }
    ]
    
    print("=" * 80)
    print("PaCoRe Consensus Example")
    print("=" * 80)
    
    strategies = ["majority_voting", "weighted_voting", "ensemble", "ranking", "borda_count"]
    
    for strategy in strategies:
        print(f"\\n--- Strategy: {strategy.upper()} ---")
        
        engine = ConsensusEngine(strategy=strategy, threshold=0.7)
        result = engine.aggregate(responses)
        
        print(f"Consensus Text: {result['text'][:150]}...")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Method: {result['method']}")

if __name__ == "__main__":
    example_consensus()
''',

    'tests/__init__.py': '"""Test package"""\\n',

    'tests/test_consensus.py': '''"""
Unit tests for consensus engine
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from consensus import ConsensusEngine

class TestConsensusEngine:
    
    @pytest.fixture
    def sample_responses(self):
        return [
            {"response": "Answer A", "confidence": 0.9, "model": "model1"},
            {"response": "Answer B", "confidence": 0.7, "model": "model2"},
            {"response": "Answer A", "confidence": 0.85, "model": "model3"}
        ]
    
    def test_majority_voting(self, sample_responses):
        engine = ConsensusEngine(strategy="majority_voting")
        result = engine.aggregate(sample_responses)
        
        assert result['text'] == "Answer A"
        assert result['confidence'] > 0.5
        assert 'votes' in result
    
    def test_weighted_voting(self, sample_responses):
        engine = ConsensusEngine(strategy="weighted_voting")
        result = engine.aggregate(sample_responses)
        
        assert result['text'] in ["Answer A", "Answer B"]
        assert 0 <= result['confidence'] <= 1
    
    def test_empty_responses(self):
        engine = ConsensusEngine()
        result = engine.aggregate([])
        
        assert result['text'] == ""
        assert result['confidence'] == 0.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
}

# Create directories
directories = ['src', 'tests', 'examples', 'logs', 'data']
for dir_path in directories:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {dir_path}")

# Create files
for file_path, content in FILES.items():
    full_path = Path(file_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content)
    print(f"✓ Created file: {file_path}")

print("\\n" + "=" * 80)
print("✅ PaCoRe project setup complete!")
print("=" * 80)
print("\\nNext steps:")
print("1. Install dependencies: pip install -r requirements.txt")
print("2. Run example: python examples/simple_usage.py")
print("3. Run tests: pytest tests/ -v")
print("4. Read GETTING_STARTED.md for detailed guide")
