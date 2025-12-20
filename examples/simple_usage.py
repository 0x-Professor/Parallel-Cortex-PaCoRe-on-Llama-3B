"""
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
        print(f"\n--- Strategy: {strategy.upper()} ---")
        
        engine = ConsensusEngine(strategy=strategy, threshold=0.7)
        result = engine.aggregate(responses)
        
        print(f"Consensus Text: {result['text'][:150]}...")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Method: {result['method']}")

if __name__ == "__main__":
    example_consensus()
