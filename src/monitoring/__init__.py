"""
Monitoring Framework for RAG System

This package provides comprehensive monitoring capabilities including:
- Evaluation metrics (Precision@K, Recall@K, MRR, NDCG)
- Response quality assessment
- Performance monitoring with latency and throughput tracking
- Cost analytics for OpenAI and Anthropic API usage
- Integration with the existing RAG pipeline
"""

from .evaluation_metrics import (
    EvaluationFramework,
    RetrievalEvaluator,
    QualityEvaluator,
    PerformanceMonitor as MetricsPerformanceMonitor,
    CostAnalyzer,
    EvaluationMetrics,
    RetrievalResult,
    QueryResult,
    MetricType
)

from .performance_monitor import (
    PerformanceMonitor,
    MetricsCollector,
    SystemMonitor,
    AlertManager,
    RequestTracker,
    PerformanceSnapshot,
    PerformanceAlert
)

from .cost_analytics import (
    CostTracker,
    CostCalculator,
    CostOptimizer,
    APIUsage,
    CostProjection,
    CostAlert,
    APIProvider
)

from .monitoring_integration import (
    MonitoringManager,
    MonitoredRAGPipeline
)

__all__ = [
    # Evaluation metrics
    'EvaluationFramework',
    'RetrievalEvaluator', 
    'QualityEvaluator',
    'MetricsPerformanceMonitor',
    'CostAnalyzer',
    'EvaluationMetrics',
    'RetrievalResult',
    'QueryResult',
    'MetricType',
    
    # Performance monitoring
    'PerformanceMonitor',
    'MetricsCollector',
    'SystemMonitor',
    'AlertManager', 
    'RequestTracker',
    'PerformanceSnapshot',
    'PerformanceAlert',
    
    # Cost analytics
    'CostTracker',
    'CostCalculator',
    'CostOptimizer',
    'APIUsage',
    'CostProjection',
    'CostAlert',
    'APIProvider',
    
    # Integration
    'MonitoringManager',
    'MonitoredRAGPipeline'
]