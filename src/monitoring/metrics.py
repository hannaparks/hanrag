"""Prometheus metrics for RAG system monitoring"""

from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
import time
from functools import wraps
from loguru import logger
import psutil
import asyncio

# Create a custom registry to avoid conflicts
REGISTRY = CollectorRegistry()

# System Info
system_info = Info(
    'rag_system_info',
    'RAG system information',
    registry=REGISTRY
)

# Request Metrics
request_counter = Counter(
    'rag_requests_total',
    'Total number of requests',
    ['endpoint', 'method', 'status'],
    registry=REGISTRY
)

request_duration = Histogram(
    'rag_request_duration_seconds',
    'Request duration in seconds',
    ['endpoint', 'method'],
    registry=REGISTRY
)

# Query Metrics
query_counter = Counter(
    'rag_queries_total',
    'Total number of RAG queries',
    ['search_method', 'query_type'],
    registry=REGISTRY
)

query_duration = Histogram(
    'rag_query_duration_seconds',
    'Query processing duration in seconds',
    ['search_method'],
    registry=REGISTRY
)

query_results_count = Histogram(
    'rag_query_results_count',
    'Number of results returned per query',
    ['search_method'],
    buckets=(0, 10, 25, 50, 100, 200, 500),
    registry=REGISTRY
)

# Ingestion Metrics
ingestion_counter = Counter(
    'rag_ingestion_total',
    'Total number of ingestion operations',
    ['source_type', 'status'],
    registry=REGISTRY
)

ingestion_chunks_counter = Counter(
    'rag_ingestion_chunks_total',
    'Total number of chunks ingested',
    ['source_type'],
    registry=REGISTRY
)

ingestion_duration = Histogram(
    'rag_ingestion_duration_seconds',
    'Ingestion operation duration in seconds',
    ['source_type'],
    registry=REGISTRY
)

# Vector Database Metrics
vector_db_operations = Counter(
    'rag_vector_db_operations_total',
    'Total vector database operations',
    ['operation', 'status'],
    registry=REGISTRY
)

vector_db_operation_duration = Histogram(
    'rag_vector_db_operation_duration_seconds',
    'Vector database operation duration',
    ['operation'],
    registry=REGISTRY
)

vector_db_documents = Gauge(
    'rag_vector_db_documents_total',
    'Total documents in vector database',
    registry=REGISTRY
)

# Cache Metrics
cache_operations = Counter(
    'rag_cache_operations_total',
    'Total cache operations',
    ['operation', 'status'],
    registry=REGISTRY
)

cache_hit_rate = Gauge(
    'rag_cache_hit_rate',
    'Cache hit rate (percentage)',
    registry=REGISTRY
)

# LLM Metrics
llm_requests = Counter(
    'rag_llm_requests_total',
    'Total LLM API requests',
    ['model', 'status'],
    registry=REGISTRY
)

llm_tokens = Counter(
    'rag_llm_tokens_total',
    'Total LLM tokens processed',
    ['model', 'token_type'],
    registry=REGISTRY
)

llm_duration = Histogram(
    'rag_llm_duration_seconds',
    'LLM API call duration',
    ['model'],
    registry=REGISTRY
)

# System Resource Metrics
cpu_usage = Gauge(
    'rag_system_cpu_usage_percent',
    'System CPU usage percentage',
    registry=REGISTRY
)

memory_usage = Gauge(
    'rag_system_memory_usage_bytes',
    'System memory usage in bytes',
    registry=REGISTRY
)

# Connection Pool Metrics
connection_pool_size = Gauge(
    'rag_connection_pool_size',
    'Number of connections in pool',
    ['pool_name'],
    registry=REGISTRY
)

connection_pool_active = Gauge(
    'rag_connection_pool_active',
    'Number of active connections',
    ['pool_name'],
    registry=REGISTRY
)

# Error Metrics
error_counter = Counter(
    'rag_errors_total',
    'Total number of errors',
    ['error_type', 'component'],
    registry=REGISTRY
)


def track_request(endpoint: str, method: str = "POST"):
    """Decorator to track API request metrics"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                request_counter.labels(endpoint=endpoint, method=method, status=status).inc()
                request_duration.labels(endpoint=endpoint, method=method).observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                request_counter.labels(endpoint=endpoint, method=method, status=status).inc()
                request_duration.labels(endpoint=endpoint, method=method).observe(duration)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def track_query(search_method: str, query_type: str = "general"):
    """Decorator to track query metrics"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Track metrics
                duration = time.time() - start_time
                query_counter.labels(search_method=search_method, query_type=query_type).inc()
                query_duration.labels(search_method=search_method).observe(duration)
                
                # Track result count if available
                if isinstance(result, dict) and 'context_count' in result:
                    query_results_count.labels(search_method=search_method).observe(result['context_count'])
                
                return result
            except Exception as e:
                error_counter.labels(error_type=type(e).__name__, component="query").inc()
                raise
        
        return wrapper
    return decorator


def track_ingestion(source_type: str):
    """Decorator to track ingestion metrics"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                
                # Extract chunk count from result if available
                if isinstance(result, dict) and 'chunks_processed' in result:
                    ingestion_chunks_counter.labels(source_type=source_type).inc(result['chunks_processed'])
                
                return result
            except Exception as e:
                status = "error"
                error_counter.labels(error_type=type(e).__name__, component="ingestion").inc()
                raise
            finally:
                duration = time.time() - start_time
                ingestion_counter.labels(source_type=source_type, status=status).inc()
                ingestion_duration.labels(source_type=source_type).observe(duration)
        
        return wrapper
    return decorator


def track_vector_operation(operation: str):
    """Decorator to track vector database operations"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                error_counter.labels(error_type=type(e).__name__, component="vector_db").inc()
                raise
            finally:
                duration = time.time() - start_time
                vector_db_operations.labels(operation=operation, status=status).inc()
                vector_db_operation_duration.labels(operation=operation).observe(duration)
        
        return wrapper
    return decorator


def track_cache_operation(operation: str):
    """Track cache operations"""
    def track(status: str):
        cache_operations.labels(operation=operation, status=status).inc()
    return track


def track_llm_request(model: str, prompt_tokens: int, completion_tokens: int, duration: float, status: str = "success"):
    """Track LLM API request metrics"""
    llm_requests.labels(model=model, status=status).inc()
    llm_tokens.labels(model=model, token_type="prompt").inc(prompt_tokens)
    llm_tokens.labels(model=model, token_type="completion").inc(completion_tokens)
    llm_duration.labels(model=model).observe(duration)


def update_system_metrics():
    """Update system resource metrics"""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_usage.set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage.set(memory.used)
        
    except Exception as e:
        logger.error(f"Failed to update system metrics: {e}")


def update_vector_db_metrics(document_count: int):
    """Update vector database metrics"""
    vector_db_documents.set(document_count)


def update_cache_metrics(hit_rate: float):
    """Update cache metrics"""
    cache_hit_rate.set(hit_rate * 100)  # Convert to percentage


def update_connection_pool_metrics(pool_name: str, pool_size: int, active_connections: int):
    """Update connection pool metrics"""
    connection_pool_size.labels(pool_name=pool_name).set(pool_size)
    connection_pool_active.labels(pool_name=pool_name).set(active_connections)


def set_system_info(version: str, environment: str):
    """Set system information"""
    system_info.info({
        'version': version,
        'environment': environment,
        'embedding_model': 'text-embedding-3-large',
        'llm_model': 'claude-3-5-sonnet-20241022'
    })


def get_metrics() -> bytes:
    """Generate Prometheus metrics in text format"""
    # Update system metrics before generating
    update_system_metrics()
    
    return generate_latest(REGISTRY)


class MetricsCollector:
    """Background metrics collector"""
    
    def __init__(self, rag_pipeline=None):
        self.rag_pipeline = rag_pipeline
        self.running = False
        self._task = None
    
    async def start(self):
        """Start background metrics collection"""
        if self.running:
            return
        
        self.running = True
        self._task = asyncio.create_task(self._collect_metrics())
        logger.info("Metrics collector started")
    
    async def stop(self):
        """Stop background metrics collection"""
        self.running = False
        if self._task:
            await self._task
        logger.info("Metrics collector stopped")
    
    async def _collect_metrics(self):
        """Periodically collect metrics"""
        while self.running:
            try:
                # Update system metrics
                update_system_metrics()
                
                # Update vector DB metrics if pipeline available
                if self.rag_pipeline:
                    try:
                        stats = await self.rag_pipeline.get_stats()
                        if 'total_documents' in stats:
                            update_vector_db_metrics(stats['total_documents'])
                    except Exception as e:
                        logger.error(f"Failed to collect RAG pipeline stats: {e}")
                
                # Sleep for 30 seconds
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(60)  # Sleep longer on error