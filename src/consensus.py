"""
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
