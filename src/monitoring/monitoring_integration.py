"""
Monitoring Framework Integration with RAG Pipeline

This module integrates the monitoring capabilities with the existing RAG pipeline,
providing seamless monitoring without disrupting the core functionality.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import json
from pathlib import Path

if TYPE_CHECKING:
    from ..core.rag_pipeline import RAGPipeline
from .evaluation_metrics import EvaluationFramework, RetrievalResult
from .performance_monitor import PerformanceMonitor
from .cost_analytics import CostTracker


class MonitoringManager:
    """Central monitoring manager that coordinates all monitoring components"""
    
    def __init__(
        self,
        enable_evaluation: bool = True,
        enable_performance: bool = True,
        enable_cost_tracking: bool = True,
        save_path: str = "monitoring_data"
    ):
        self.save_path = Path(save_path)
        self.save_path.mkdir(exist_ok=True)
        
        # Initialize monitoring components
        self.evaluation_framework = EvaluationFramework(
            save_path=str(self.save_path / "evaluation")
        ) if enable_evaluation else None
        
        self.performance_monitor = PerformanceMonitor(
            save_path=str(self.save_path / "performance")
        ) if enable_performance else None
        
        self.cost_tracker = CostTracker(
            save_path=str(self.save_path / "costs")
        ) if enable_cost_tracking else None
        
        # Configuration
        self.enabled = {
            'evaluation': enable_evaluation,
            'performance': enable_performance,
            'cost_tracking': enable_cost_tracking
        }
        
        # Start performance monitoring if enabled
        if self.performance_monitor:
            self.performance_monitor.start_monitoring(interval=30)
    
    async def monitor_query(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        generated_response: str,
        query_time: float,
        generation_time: float,
        total_time: float,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        openai_usage: Optional[Dict[str, Any]] = None,
        anthropic_usage: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Monitor a complete query-response cycle"""
        
        monitoring_results = {}
        
        # Convert retrieved docs to RetrievalResult format
        retrieval_results = []
        for i, doc in enumerate(retrieved_docs):
            retrieval_result = RetrievalResult(
                document_id=doc.get('id', f'doc_{i}'),
                content=doc.get('content', ''),
                score=doc.get('score', 0.0),
                relevant=doc.get('relevant', False),  # Would need ground truth
                position=i + 1,
                source=doc.get('source', ''),
                metadata=doc.get('metadata', {})
            )
            retrieval_results.append(retrieval_result)
        
        # Evaluation monitoring
        if self.evaluation_framework:
            try:
                eval_results = await self.evaluation_framework.evaluate_query(
                    query=query,
                    retrieved_docs=retrieval_results,
                    generated_response=generated_response,
                    query_time=query_time,
                    generation_time=generation_time,
                    total_time=total_time,
                    relevant_doc_count=len([d for d in retrieval_results if d.relevant]),
                    openai_usage=openai_usage,
                    anthropic_usage=anthropic_usage
                )
                monitoring_results['evaluation'] = eval_results
            except Exception as e:
                print(f"Error in evaluation monitoring: {e}")
        
        # Cost tracking
        if self.cost_tracker:
            try:
                if openai_usage:
                    self.cost_tracker.record_openai_usage(
                        model=openai_usage.get('model', 'unknown'),
                        tokens=openai_usage.get('tokens', 0),
                        operation=openai_usage.get('operation', 'embedding'),
                        request_id=request_id,
                        user_id=user_id,
                        channel_id=channel_id,
                        query_type='rag_query'
                    )
                
                if anthropic_usage:
                    self.cost_tracker.record_anthropic_usage(
                        model=anthropic_usage.get('model', 'unknown'),
                        input_tokens=anthropic_usage.get('input_tokens', 0),
                        output_tokens=anthropic_usage.get('output_tokens', 0),
                        request_id=request_id,
                        user_id=user_id,
                        channel_id=channel_id,
                        query_type='rag_query'
                    )
                
                # Get cost summary
                cost_summary = self.cost_tracker.get_cost_summary()
                monitoring_results['cost'] = cost_summary
            except Exception as e:
                print(f"Error in cost tracking: {e}")
        
        return monitoring_results
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive monitoring report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_enabled': self.enabled
        }
        
        # Evaluation metrics
        if self.evaluation_framework:
            try:
                eval_metrics = self.evaluation_framework.generate_evaluation_report()
                report['evaluation'] = {
                    'retrieval_metrics': {
                        'precision_at_k': eval_metrics.precision_at_k,
                        'recall_at_k': eval_metrics.recall_at_k,
                        'mrr': eval_metrics.mrr,
                        'ndcg_at_k': eval_metrics.ndcg_at_k
                    },
                    'quality_metrics': {
                        'response_quality_score': eval_metrics.response_quality_score,
                        'factual_accuracy': eval_metrics.factual_accuracy,
                        'completeness': eval_metrics.completeness,
                        'coherence': eval_metrics.coherence,
                        'source_attribution_accuracy': eval_metrics.source_attribution_accuracy
                    },
                    'total_queries': eval_metrics.total_queries
                }
            except Exception as e:
                report['evaluation'] = {'error': str(e)}
        
        # Performance metrics
        if self.performance_monitor:
            try:
                perf_summary = self.performance_monitor.get_performance_summary(hours=24)
                report['performance'] = perf_summary
                
                # Add current snapshot
                current_snapshot = self.performance_monitor.capture_snapshot()
                report['current_performance'] = {
                    'cpu_percent': current_snapshot.cpu_percent,
                    'memory_percent': current_snapshot.memory_percent,
                    'active_requests': current_snapshot.active_requests,
                    'avg_response_time': current_snapshot.avg_response_time,
                    'requests_per_second': current_snapshot.requests_per_second
                }
            except Exception as e:
                report['performance'] = {'error': str(e)}
        
        # Cost analytics
        if self.cost_tracker:
            try:
                cost_summary = self.cost_tracker.get_cost_summary()
                cost_projection = self.cost_tracker.get_cost_projection(days_ahead=30)
                optimization_recs = self.cost_tracker.get_optimization_recommendations()
                
                report['cost'] = {
                    'current_summary': cost_summary,
                    'projection_30_days': {
                        'projected_cost': cost_projection.projected_cost,
                        'confidence_interval': cost_projection.confidence_interval
                    },
                    'optimization_recommendations': optimization_recs
                }
            except Exception as e:
                report['cost'] = {'error': str(e)}
        
        return report
    
    def save_monitoring_report(self, filename: Optional[str] = None):
        """Save comprehensive monitoring report"""
        report = self.get_comprehensive_report()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"monitoring_report_{timestamp}.json"
        
        filepath = self.save_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Monitoring report saved to: {filepath}")
        return str(filepath)
    
    def get_health_status(self) -> Dict[str, str]:
        """Get overall health status of the monitoring system"""
        status = {}
        
        # Check evaluation framework
        if self.evaluation_framework:
            try:
                # Simple health check - verify we can generate metrics
                metrics = self.evaluation_framework.generate_evaluation_report()
                status['evaluation'] = 'healthy'
            except Exception as e:
                status['evaluation'] = f'error: {str(e)}'
        else:
            status['evaluation'] = 'disabled'
        
        # Check performance monitor
        if self.performance_monitor:
            try:
                # Verify we can capture snapshots
                snapshot = self.performance_monitor.capture_snapshot()
                status['performance'] = 'healthy'
            except Exception as e:
                status['performance'] = f'error: {str(e)}'
        else:
            status['performance'] = 'disabled'
        
        # Check cost tracker
        if self.cost_tracker:
            try:
                # Verify we can generate summaries
                summary = self.cost_tracker.get_cost_summary()
                status['cost_tracking'] = 'healthy'
            except Exception as e:
                status['cost_tracking'] = f'error: {str(e)}'
        else:
            status['cost_tracking'] = 'disabled'
        
        return status
    
    def cleanup(self):
        """Cleanup monitoring resources"""
        if self.performance_monitor:
            self.performance_monitor.stop_monitoring()


class MonitoredRAGPipeline:
    """RAG Pipeline wrapper with integrated monitoring"""
    
    def __init__(
        self,
        rag_pipeline: 'RAGPipeline',
        monitoring_manager: Optional[MonitoringManager] = None,
        enable_monitoring: bool = True
    ):
        self.rag_pipeline = rag_pipeline
        
        if enable_monitoring and monitoring_manager is None:
            self.monitoring_manager = MonitoringManager()
        else:
            self.monitoring_manager = monitoring_manager
        
        self.enable_monitoring = enable_monitoring and (monitoring_manager is not None)
    
    async def query(
        self,
        query: str,
        context_filters: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        user_id: Optional[str] = None,
        channel_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute query with monitoring"""
        
        # Generate request ID for tracking
        request_id = f"req_{int(time.time() * 1000000)}"
        
        # Track performance if enabled
        performance_tracker = None
        if self.enable_monitoring and self.monitoring_manager.performance_monitor:
            performance_tracker = self.monitoring_manager.performance_monitor.track_request()
        
        try:
            if performance_tracker:
                with performance_tracker:
                    start_time = time.time()
                    
                    # Execute the original query
                    result = await self.rag_pipeline.query(
                        query=query,
                        context_filters=context_filters,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    
                    end_time = time.time()
                    total_time = (end_time - start_time) * 1000  # Convert to ms
                    
                    # Record component times if available in result
                    if 'embedding_time' in result:
                        performance_tracker.record_embedding_time(result['embedding_time'])
                    if 'retrieval_time' in result:
                        performance_tracker.record_retrieval_time(result['retrieval_time'])
                    if 'generation_time' in result:
                        performance_tracker.record_generation_time(result['generation_time'])
            else:
                # Execute without performance tracking
                start_time = time.time()
                result = await self.rag_pipeline.query(
                    query=query,
                    context_filters=context_filters,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                end_time = time.time()
                total_time = (end_time - start_time) * 1000
            
            # Add monitoring data if enabled
            if self.enable_monitoring and self.monitoring_manager:
                
                # Extract timing information
                query_time = result.get('retrieval_time', 0)
                generation_time = result.get('generation_time', 0)
                
                # Extract usage information
                openai_usage = result.get('openai_usage')
                anthropic_usage = result.get('anthropic_usage')
                
                # Monitor the query
                monitoring_results = await self.monitoring_manager.monitor_query(
                    query=query,
                    retrieved_docs=result.get('sources', []),
                    generated_response=result.get('response', ''),
                    query_time=query_time,
                    generation_time=generation_time,
                    total_time=total_time,
                    request_id=request_id,
                    user_id=user_id,
                    channel_id=channel_id,
                    openai_usage=openai_usage,
                    anthropic_usage=anthropic_usage
                )
                
                # Add monitoring results to response
                result['monitoring'] = monitoring_results
                result['request_id'] = request_id
            
            return result
            
        except Exception as e:
            # Record failed request in performance monitoring
            if performance_tracker:
                # The context manager will handle the failure
                pass
            raise e
    
    async def ingest_documents(
        self,
        sources: List[str],
        source_type: str = "file",
        **kwargs
    ) -> Dict[str, Any]:
        """Ingest documents with monitoring"""
        
        # Track ingestion performance
        start_time = time.time()
        
        try:
            result = await self.rag_pipeline.ingest_documents(
                sources=sources,
                source_type=source_type,
                **kwargs
            )
            
            end_time = time.time()
            ingestion_time = (end_time - start_time) * 1000
            
            # Add ingestion monitoring
            if self.enable_monitoring and self.monitoring_manager:
                # Track ingestion costs if usage info is available
                if 'openai_usage' in result and self.monitoring_manager.cost_tracker:
                    openai_usage = result['openai_usage']
                    self.monitoring_manager.cost_tracker.record_openai_usage(
                        model=openai_usage.get('model', 'unknown'),
                        tokens=openai_usage.get('tokens', 0),
                        operation='embedding'
                    )
                
                result['monitoring'] = {
                    'ingestion_time': ingestion_time,
                    'ingestion_timestamp': datetime.now().isoformat()
                }
            
            return result
            
        except Exception as e:
            # Log ingestion failure
            if self.enable_monitoring:
                print(f"Ingestion failed after {(time.time() - start_time)*1000:.2f}ms: {e}")
            raise e
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        if not self.enable_monitoring or not self.monitoring_manager:
            return {'monitoring_enabled': False}
        
        return {
            'monitoring_enabled': True,
            'health_status': self.monitoring_manager.get_health_status(),
            'components_enabled': self.monitoring_manager.enabled
        }
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        if not self.enable_monitoring or not self.monitoring_manager:
            return {'error': 'Monitoring not enabled'}
        
        return self.monitoring_manager.get_comprehensive_report()
    
    def save_monitoring_report(self, filename: Optional[str] = None) -> str:
        """Save monitoring report to file"""
        if not self.enable_monitoring or not self.monitoring_manager:
            raise ValueError("Monitoring not enabled")
        
        return self.monitoring_manager.save_monitoring_report(filename)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.monitoring_manager:
            self.monitoring_manager.cleanup()


# Example usage
if __name__ == "__main__":
    # This would normally import the actual RAG pipeline
    # For demo purposes, we'll create a mock
    
    class MockRAGPipeline:
        async def query(self, query: str, **kwargs):
            await asyncio.sleep(0.1)  # Simulate processing
            return {
                'response': f"Mock response to: {query}",
                'sources': [
                    {'id': 'doc1', 'content': 'Sample content', 'score': 0.9},
                    {'id': 'doc2', 'content': 'More content', 'score': 0.8}
                ],
                'retrieval_time': 50,
                'generation_time': 100,
                'openai_usage': {'model': 'text-embedding-3-large', 'tokens': 200},
                'anthropic_usage': {'model': 'claude-3-5-sonnet-20241022', 'input_tokens': 150, 'output_tokens': 50}
            }
        
        async def ingest_documents(self, sources, source_type="file", **kwargs):
            await asyncio.sleep(0.2)  # Simulate ingestion
            return {
                'ingested_count': len(sources),
                'openai_usage': {'model': 'text-embedding-3-large', 'tokens': 1000}
            }
    
    async def demo():
        # Create mock pipeline and monitored wrapper
        mock_pipeline = MockRAGPipeline()
        monitored_pipeline = MonitoredRAGPipeline(mock_pipeline)
        
        # Execute some queries
        for i in range(3):
            result = await monitored_pipeline.query(
                f"What is question {i}?",
                user_id="user123",
                channel_id="channel456"
            )
            print(f"Query {i} completed with monitoring")
        
        # Generate and save report
        report = monitored_pipeline.generate_monitoring_report()
        print("Monitoring Report:", json.dumps(report, indent=2, default=str))
        
        # Cleanup
        monitored_pipeline.cleanup()
    
    # Run demo
    asyncio.run(demo())
    print("Monitoring integration demo completed!")