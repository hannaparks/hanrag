"""
Evaluation & Monitoring Framework for RAG System

This module provides comprehensive evaluation metrics for retrieval and response quality,
including Precision@K, Recall@K, MRR, NDCG, and custom quality assessments.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import statistics
from enum import Enum

import numpy as np


class MetricType(str, Enum):
    """Types of evaluation metrics"""
    RETRIEVAL = "retrieval"
    QUALITY = "quality"
    PERFORMANCE = "performance"
    COST = "cost"


@dataclass
class RetrievalResult:
    """Single retrieval result with relevance information"""
    document_id: str
    content: str
    score: float
    relevant: bool = False  # Ground truth relevance
    position: int = 0  # Position in result list
    source: str = ""
    metadata: Dict[str, Any] = None


@dataclass
class QueryResult:
    """Complete query result with retrieved documents and generated response"""
    query: str
    retrieved_docs: List[RetrievalResult]
    generated_response: str
    query_time: float
    generation_time: float
    total_time: float
    relevant_doc_count: int = 0
    timestamp: datetime = None
    cost_info: Dict[str, float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.cost_info is None:
            self.cost_info = {}


@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics"""
    # Retrieval Metrics
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float] 
    mrr: float  # Mean Reciprocal Rank
    ndcg_at_k: Dict[int, float]  # Normalized Discounted Cumulative Gain
    
    # Quality Metrics
    response_quality_score: float
    factual_accuracy: float
    completeness: float
    coherence: float
    source_attribution_accuracy: float
    
    # Performance Metrics
    avg_query_latency: float
    avg_generation_latency: float
    avg_total_latency: float
    throughput_queries_per_second: float
    
    # Cost Metrics
    total_cost: float
    cost_per_query: float
    openai_cost: float
    anthropic_cost: float
    
    # Metadata
    total_queries: int
    evaluation_period: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class RetrievalEvaluator:
    """Evaluates retrieval performance using standard IR metrics"""
    
    def __init__(self):
        self.query_results: List[QueryResult] = []
    
    def add_query_result(self, query_result: QueryResult):
        """Add a query result for evaluation"""
        self.query_results.append(query_result)
    
    def precision_at_k(self, k: int) -> float:
        """Calculate Precision@K across all queries"""
        if not self.query_results:
            return 0.0
        
        precisions = []
        for result in self.query_results:
            retrieved_k = result.retrieved_docs[:k]
            if not retrieved_k:
                precisions.append(0.0)
                continue
            
            relevant_in_k = sum(1 for doc in retrieved_k if doc.relevant)
            precision = relevant_in_k / len(retrieved_k)
            precisions.append(precision)
        
        return statistics.mean(precisions)
    
    def recall_at_k(self, k: int) -> float:
        """Calculate Recall@K across all queries"""
        if not self.query_results:
            return 0.0
        
        recalls = []
        for result in self.query_results:
            if result.relevant_doc_count == 0:
                recalls.append(1.0)  # No relevant docs, perfect recall
                continue
            
            retrieved_k = result.retrieved_docs[:k]
            relevant_in_k = sum(1 for doc in retrieved_k if doc.relevant)
            recall = relevant_in_k / result.relevant_doc_count
            recalls.append(recall)
        
        return statistics.mean(recalls)
    
    def mean_reciprocal_rank(self) -> float:
        """Calculate Mean Reciprocal Rank (MRR)"""
        if not self.query_results:
            return 0.0
        
        reciprocal_ranks = []
        for result in self.query_results:
            # Find position of first relevant document
            first_relevant_pos = None
            for i, doc in enumerate(result.retrieved_docs, 1):
                if doc.relevant:
                    first_relevant_pos = i
                    break
            
            if first_relevant_pos:
                reciprocal_ranks.append(1.0 / first_relevant_pos)
            else:
                reciprocal_ranks.append(0.0)
        
        return statistics.mean(reciprocal_ranks)
    
    def ndcg_at_k(self, k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at K"""
        if not self.query_results:
            return 0.0
        
        ndcg_scores = []
        for result in self.query_results:
            retrieved_k = result.retrieved_docs[:k]
            
            # Calculate DCG
            dcg = 0.0
            for i, doc in enumerate(retrieved_k):
                relevance = 1.0 if doc.relevant else 0.0
                dcg += relevance / math.log2(i + 2)  # i+2 because log2(1) = 0
            
            # Calculate IDCG (perfect ranking)
            relevant_docs = [doc for doc in result.retrieved_docs if doc.relevant]
            idcg = 0.0
            for i in range(min(k, len(relevant_docs))):
                idcg += 1.0 / math.log2(i + 2)
            
            # Normalize
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
            else:
                ndcg_scores.append(0.0)
        
        return statistics.mean(ndcg_scores)


class QualityEvaluator:
    """Evaluates response quality using multiple dimensions"""
    
    def __init__(self):
        self.quality_scores: List[Dict[str, float]] = []
    
    def evaluate_response_quality(
        self, 
        query: str,
        response: str,
        retrieved_docs: List[RetrievalResult],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate response quality across multiple dimensions"""
        
        scores = {
            'factual_accuracy': self._assess_factual_accuracy(response, retrieved_docs),
            'completeness': self._assess_completeness(query, response),
            'coherence': self._assess_coherence(response),
            'source_attribution': self._assess_source_attribution(response, retrieved_docs),
            'relevance': self._assess_relevance(query, response),
            'conciseness': self._assess_conciseness(response)
        }
        
        # Overall quality score (weighted average)
        weights = {
            'factual_accuracy': 0.25,
            'completeness': 0.20,
            'coherence': 0.20,
            'source_attribution': 0.15,
            'relevance': 0.15,
            'conciseness': 0.05
        }
        
        overall_score = sum(scores[metric] * weights[metric] for metric in scores)
        scores['overall'] = overall_score
        
        self.quality_scores.append(scores)
        return scores
    
    def _assess_factual_accuracy(self, response: str, retrieved_docs: List[RetrievalResult]) -> float:
        """Assess factual accuracy based on source content alignment"""
        # Simple heuristic: check if response contains information from retrieved docs
        if not retrieved_docs or not response:
            return 0.0
        
        # Count overlapping terms (simplified approach)
        response_words = set(response.lower().split())
        doc_words = set()
        for doc in retrieved_docs:
            doc_words.update(doc.content.lower().split())
        
        if not doc_words:
            return 0.0
        
        overlap = len(response_words.intersection(doc_words))
        return min(1.0, overlap / max(20, len(response_words) * 0.3))
    
    def _assess_completeness(self, query: str, response: str) -> float:
        """Assess how completely the response addresses the query"""
        if not query or not response:
            return 0.0
        
        # Simple heuristic based on response length and query complexity
        query_words = len(query.split())
        response_words = len(response.split())
        
        # Expected response length based on query complexity
        expected_length = max(20, query_words * 3)
        
        if response_words >= expected_length:
            return 1.0
        else:
            return response_words / expected_length
    
    def _assess_coherence(self, response: str) -> float:
        """Assess response coherence and logical flow"""
        if not response:
            return 0.0
        
        sentences = response.split('.')
        if len(sentences) < 2:
            return 0.8  # Single sentence responses are generally coherent
        
        # Simple heuristic: longer responses with proper punctuation tend to be more coherent
        avg_sentence_length = statistics.mean([len(s.strip().split()) for s in sentences if s.strip()])
        
        # Optimal sentence length is around 15-20 words
        if 10 <= avg_sentence_length <= 25:
            return 1.0
        else:
            return max(0.6, 1.0 - abs(avg_sentence_length - 17.5) / 20)
    
    def _assess_source_attribution(self, response: str, retrieved_docs: List[RetrievalResult]) -> float:
        """Assess accuracy of source citations"""
        if not retrieved_docs:
            return 1.0  # No sources required
        
        # Look for source indicators in response
        source_indicators = ['source:', 'from:', 'according to', 'based on', 'reference:', 'doc:']
        has_attribution = any(indicator in response.lower() for indicator in source_indicators)
        
        if has_attribution:
            return 1.0
        elif len(retrieved_docs) > 0:
            return 0.5  # Sources available but not cited
        else:
            return 1.0  # No sources to cite
    
    def _assess_relevance(self, query: str, response: str) -> float:
        """Assess how relevant the response is to the query"""
        if not query or not response:
            return 0.0
        
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common stop words for better relevance assessment
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        query_content_words = query_words - stop_words
        response_content_words = response_words - stop_words
        
        if not query_content_words:
            return 1.0
        
        overlap = len(query_content_words.intersection(response_content_words))
        return overlap / len(query_content_words)
    
    def _assess_conciseness(self, response: str) -> float:
        """Assess response conciseness (not too verbose)"""
        if not response:
            return 0.0
        
        word_count = len(response.split())
        
        # Optimal response length is 50-200 words
        if 50 <= word_count <= 200:
            return 1.0
        elif word_count < 50:
            return word_count / 50
        else:
            # Penalize overly long responses
            return max(0.3, 1.0 - (word_count - 200) / 500)
    
    def get_average_scores(self) -> Dict[str, float]:
        """Get average quality scores across all evaluations"""
        if not self.quality_scores:
            return {}
        
        metrics = self.quality_scores[0].keys()
        return {
            metric: statistics.mean([score[metric] for score in self.quality_scores])
            for metric in metrics
        }


class PerformanceMonitor:
    """Monitors system performance metrics"""
    
    def __init__(self):
        self.query_times: List[float] = []
        self.generation_times: List[float] = []
        self.total_times: List[float] = []
        self.query_timestamps: List[datetime] = []
    
    def record_query_performance(
        self,
        query_time: float,
        generation_time: float,
        total_time: float
    ):
        """Record performance metrics for a query"""
        self.query_times.append(query_time)
        self.generation_times.append(generation_time)
        self.total_times.append(total_time)
        self.query_timestamps.append(datetime.now())
    
    def get_latency_stats(self) -> Dict[str, Dict[str, float]]:
        """Get latency statistics"""
        if not self.total_times:
            return {}
        
        return {
            'query_latency': {
                'mean': statistics.mean(self.query_times),
                'median': statistics.median(self.query_times),
                'p95': np.percentile(self.query_times, 95),
                'p99': np.percentile(self.query_times, 99),
                'min': min(self.query_times),
                'max': max(self.query_times)
            },
            'generation_latency': {
                'mean': statistics.mean(self.generation_times),
                'median': statistics.median(self.generation_times),
                'p95': np.percentile(self.generation_times, 95),
                'p99': np.percentile(self.generation_times, 99),
                'min': min(self.generation_times),
                'max': max(self.generation_times)
            },
            'total_latency': {
                'mean': statistics.mean(self.total_times),
                'median': statistics.median(self.total_times),
                'p95': np.percentile(self.total_times, 95),
                'p99': np.percentile(self.total_times, 99),
                'min': min(self.total_times),
                'max': max(self.total_times)
            }
        }
    
    def get_throughput(self, time_window_minutes: int = 60) -> float:
        """Calculate queries per second over time window"""
        if not self.query_timestamps:
            return 0.0
        
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_queries = [ts for ts in self.query_timestamps if ts >= cutoff_time]
        
        if len(recent_queries) < 2:
            return 0.0
        
        time_span = (recent_queries[-1] - recent_queries[0]).total_seconds()
        if time_span == 0:
            return 0.0
        
        return len(recent_queries) / time_span


class CostAnalyzer:
    """Analyzes API usage costs"""
    
    # Cost per token (approximate, as of 2024)
    OPENAI_COSTS = {
        'text-embedding-3-small': 0.00002 / 1000,  # $0.02 per 1M tokens
        'text-embedding-3-large': 0.00013 / 1000,  # $0.13 per 1M tokens
    }
    
    ANTHROPIC_COSTS = {
        'claude-3-5-sonnet-20241022': {
            'input': 0.003 / 1000,   # $3 per 1M input tokens
            'output': 0.015 / 1000,  # $15 per 1M output tokens
        }
    }
    
    def __init__(self):
        self.openai_usage: List[Dict[str, Any]] = []
        self.anthropic_usage: List[Dict[str, Any]] = []
    
    def record_openai_usage(
        self,
        model: str,
        tokens: int,
        operation: str = "embedding"
    ):
        """Record OpenAI API usage"""
        cost = self.OPENAI_COSTS.get(model, 0) * tokens
        
        self.openai_usage.append({
            'timestamp': datetime.now(),
            'model': model,
            'tokens': tokens,
            'operation': operation,
            'cost': cost
        })
    
    def record_anthropic_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ):
        """Record Anthropic API usage"""
        if model in self.ANTHROPIC_COSTS:
            input_cost = self.ANTHROPIC_COSTS[model]['input'] * input_tokens
            output_cost = self.ANTHROPIC_COSTS[model]['output'] * output_tokens
            total_cost = input_cost + output_cost
        else:
            total_cost = 0
        
        self.anthropic_usage.append({
            'timestamp': datetime.now(),
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost': total_cost
        })
    
    def get_cost_summary(self, days: int = 30) -> Dict[str, float]:
        """Get cost summary for specified period"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # OpenAI costs
        openai_recent = [usage for usage in self.openai_usage if usage['timestamp'] >= cutoff_date]
        openai_total = sum(usage['cost'] for usage in openai_recent)
        
        # Anthropic costs
        anthropic_recent = [usage for usage in self.anthropic_usage if usage['timestamp'] >= cutoff_date]
        anthropic_total = sum(usage['cost'] for usage in anthropic_recent)
        
        total_cost = openai_total + anthropic_total
        total_queries = len(anthropic_recent)  # Assume one Anthropic call per query
        
        return {
            'total_cost': total_cost,
            'openai_cost': openai_total,
            'anthropic_cost': anthropic_total,
            'cost_per_query': total_cost / max(1, total_queries),
            'total_queries': total_queries,
            'period_days': days
        }


class EvaluationFramework:
    """Main evaluation and monitoring framework"""
    
    def __init__(self, save_path: Optional[str] = None):
        self.retrieval_evaluator = RetrievalEvaluator()
        self.quality_evaluator = QualityEvaluator()
        self.performance_monitor = PerformanceMonitor()
        self.cost_analyzer = CostAnalyzer()
        
        self.save_path = save_path or "evaluation_results"
        self.evaluation_sessions: List[EvaluationMetrics] = []
    
    async def evaluate_query(
        self,
        query: str,
        retrieved_docs: List[RetrievalResult],
        generated_response: str,
        query_time: float,
        generation_time: float,
        total_time: float,
        relevant_doc_count: int = 0,
        openai_usage: Optional[Dict[str, Any]] = None,
        anthropic_usage: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate a complete query-response cycle"""
        
        # Create query result
        query_result = QueryResult(
            query=query,
            retrieved_docs=retrieved_docs,
            generated_response=generated_response,
            query_time=query_time,
            generation_time=generation_time,
            total_time=total_time,
            relevant_doc_count=relevant_doc_count
        )
        
        # Add to retrieval evaluator
        self.retrieval_evaluator.add_query_result(query_result)
        
        # Evaluate response quality
        quality_scores = self.quality_evaluator.evaluate_response_quality(
            query, generated_response, retrieved_docs
        )
        
        # Record performance
        self.performance_monitor.record_query_performance(
            query_time, generation_time, total_time
        )
        
        # Record costs
        if openai_usage:
            self.cost_analyzer.record_openai_usage(**openai_usage)
        
        if anthropic_usage:
            self.cost_analyzer.record_anthropic_usage(**anthropic_usage)
        
        return {
            'quality_scores': quality_scores,
            'performance': {
                'query_time': query_time,
                'generation_time': generation_time,
                'total_time': total_time
            }
        }
    
    def generate_evaluation_report(self, k_values: List[int] = None) -> EvaluationMetrics:
        """Generate comprehensive evaluation report"""
        
        if k_values is None:
            k_values = [1, 5, 10, 20]
        
        # Retrieval metrics
        precision_at_k = {k: self.retrieval_evaluator.precision_at_k(k) for k in k_values}
        recall_at_k = {k: self.retrieval_evaluator.recall_at_k(k) for k in k_values}
        ndcg_at_k = {k: self.retrieval_evaluator.ndcg_at_k(k) for k in k_values}
        mrr = self.retrieval_evaluator.mean_reciprocal_rank()
        
        # Quality metrics
        quality_scores = self.quality_evaluator.get_average_scores()
        
        # Performance metrics
        latency_stats = self.performance_monitor.get_latency_stats()
        throughput = self.performance_monitor.get_throughput()
        
        # Cost metrics
        cost_summary = self.cost_analyzer.get_cost_summary()
        
        metrics = EvaluationMetrics(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            mrr=mrr,
            ndcg_at_k=ndcg_at_k,
            response_quality_score=quality_scores.get('overall', 0.0),
            factual_accuracy=quality_scores.get('factual_accuracy', 0.0),
            completeness=quality_scores.get('completeness', 0.0),
            coherence=quality_scores.get('coherence', 0.0),
            source_attribution_accuracy=quality_scores.get('source_attribution', 0.0),
            avg_query_latency=latency_stats.get('query_latency', {}).get('mean', 0.0),
            avg_generation_latency=latency_stats.get('generation_latency', {}).get('mean', 0.0),
            avg_total_latency=latency_stats.get('total_latency', {}).get('mean', 0.0),
            throughput_queries_per_second=throughput,
            total_cost=cost_summary.get('total_cost', 0.0),
            cost_per_query=cost_summary.get('cost_per_query', 0.0),
            openai_cost=cost_summary.get('openai_cost', 0.0),
            anthropic_cost=cost_summary.get('anthropic_cost', 0.0),
            total_queries=len(self.retrieval_evaluator.query_results),
            evaluation_period=f"{cost_summary.get('period_days', 30)} days"
        )
        
        self.evaluation_sessions.append(metrics)
        return metrics
    
    def save_evaluation_report(self, metrics: EvaluationMetrics, filename: Optional[str] = None):
        """Save evaluation report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{timestamp}.json"
        
        save_dir = Path(self.save_path)
        save_dir.mkdir(exist_ok=True)
        
        filepath = save_dir / filename
        
        # Convert to dict for JSON serialization
        report_dict = asdict(metrics)
        
        # Handle datetime serialization
        report_dict['timestamp'] = metrics.timestamp.isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"Evaluation report saved to: {filepath}")
    
    def get_performance_trends(self, days: int = 7) -> Dict[str, List[float]]:
        """Get performance trends over time"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter recent sessions
        recent_sessions = [
            session for session in self.evaluation_sessions 
            if session.timestamp >= cutoff_date
        ]
        
        if not recent_sessions:
            return {}
        
        return {
            'response_quality': [s.response_quality_score for s in recent_sessions],
            'avg_latency': [s.avg_total_latency for s in recent_sessions],
            'throughput': [s.throughput_queries_per_second for s in recent_sessions],
            'cost_per_query': [s.cost_per_query for s in recent_sessions],
            'timestamps': [s.timestamp.isoformat() for s in recent_sessions]
        }
    
    def reset_metrics(self):
        """Reset all metrics for new evaluation period"""
        self.retrieval_evaluator = RetrievalEvaluator()
        self.quality_evaluator = QualityEvaluator()
        self.performance_monitor = PerformanceMonitor()
        self.cost_analyzer = CostAnalyzer()


# Example usage and testing
if __name__ == "__main__":
    # Demo evaluation framework
    framework = EvaluationFramework()
    
    # Simulate some query results
    sample_docs = [
        RetrievalResult("doc1", "Content about AI", 0.9, relevant=True, position=1),
        RetrievalResult("doc2", "Content about ML", 0.8, relevant=True, position=2),
        RetrievalResult("doc3", "Unrelated content", 0.7, relevant=False, position=3),
    ]
    
    # Simulate evaluation
    asyncio.run(
        framework.evaluate_query(
            query="What is artificial intelligence?",
            retrieved_docs=sample_docs,
            generated_response="AI is a field of computer science focused on creating intelligent machines.",
            query_time=0.5,
            generation_time=1.2,
            total_time=1.7,
            relevant_doc_count=2,
            openai_usage={"model": "text-embedding-3-large", "tokens": 100},
            anthropic_usage={"model": "claude-3-5-sonnet-20241022", "input_tokens": 200, "output_tokens": 50}
        )
    )
    
    # Generate report
    report = framework.generate_evaluation_report()
    framework.save_evaluation_report(report)
    
    print("Evaluation framework demo completed!")