"""
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
