"""
Performance Monitoring System for RAG Pipeline

This module provides real-time performance monitoring, latency tracking,
throughput analysis, and system health monitoring.
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import deque
import json
from pathlib import Path
import statistics

import numpy as np


@dataclass
class PerformanceSnapshot:
    """Single point-in-time performance snapshot"""
    timestamp: datetime
    
    # System metrics
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    
    # Application metrics
    active_requests: int
    queue_size: int
    
    # Response times (ms)
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    
    # Throughput
    requests_per_second: float
    successful_requests: int
    failed_requests: int
    
    # Component-specific metrics
    avg_embedding_time: float
    avg_retrieval_time: float
    avg_generation_time: float
    avg_total_time: float


@dataclass
class PerformanceAlert:
    """Performance alert definition"""
    alert_id: str
    timestamp: datetime
    severity: str  # INFO, WARNING, CRITICAL
    component: str
    metric: str
    current_value: float
    threshold: float
    message: str


class MetricsCollector:
    """Collects and aggregates performance metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        
        # Request timing data
        self.request_times: deque = deque(maxlen=max_history)
        self.embedding_times: deque = deque(maxlen=max_history)
        self.retrieval_times: deque = deque(maxlen=max_history)
        self.generation_times: deque = deque(maxlen=max_history)
        self.total_times: deque = deque(maxlen=max_history)
        
        # Request counting
        self.request_timestamps: deque = deque(maxlen=max_history)
        self.success_count: int = 0
        self.failure_count: int = 0
        
        # Current state
        self.active_requests: int = 0
        self.queue_size: int = 0
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def record_request_start(self) -> str:
        """Record the start of a request, return request ID"""
        with self._lock:
            request_id = f"req_{int(time.time() * 1000000)}"
            self.active_requests += 1
            return request_id
    
    def record_request_end(
        self,
        request_id: str,
        success: bool = True,
        embedding_time: Optional[float] = None,
        retrieval_time: Optional[float] = None,
        generation_time: Optional[float] = None,
        total_time: Optional[float] = None
    ):
        """Record the completion of a request"""
        with self._lock:
            self.active_requests = max(0, self.active_requests - 1)
            self.request_timestamps.append(datetime.now())
            
            if success:
                self.success_count += 1
            else:
                self.failure_count += 1
            
            # Record timing data if provided
            if embedding_time is not None:
                self.embedding_times.append(embedding_time)
            
            if retrieval_time is not None:
                self.retrieval_times.append(retrieval_time)
            
            if generation_time is not None:
                self.generation_times.append(generation_time)
            
            if total_time is not None:
                self.total_times.append(total_time)
                self.request_times.append(total_time)
    
    def update_queue_size(self, size: int):
        """Update current queue size"""
        with self._lock:
            self.queue_size = size
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        with self._lock:
            now = datetime.now()
            
            # Calculate throughput over last minute
            one_minute_ago = now - timedelta(minutes=1)
            recent_requests = [ts for ts in self.request_timestamps if ts >= one_minute_ago]
            requests_per_second = len(recent_requests) / 60.0
            
            # Calculate response time percentiles
            response_times = list(self.request_times) if self.request_times else [0]
            
            metrics = {
                'active_requests': self.active_requests,
                'queue_size': self.queue_size,
                'requests_per_second': requests_per_second,
                'successful_requests': self.success_count,
                'failed_requests': self.failure_count,
                'total_requests': self.success_count + self.failure_count,
                'success_rate': self.success_count / max(1, self.success_count + self.failure_count),
                
                # Response time stats
                'avg_response_time': statistics.mean(response_times),
                'p50_response_time': np.percentile(response_times, 50),
                'p95_response_time': np.percentile(response_times, 95),
                'p99_response_time': np.percentile(response_times, 99),
                
                # Component timing
                'avg_embedding_time': statistics.mean(self.embedding_times) if self.embedding_times else 0,
                'avg_retrieval_time': statistics.mean(self.retrieval_times) if self.retrieval_times else 0,
                'avg_generation_time': statistics.mean(self.generation_times) if self.generation_times else 0,
                'avg_total_time': statistics.mean(self.total_times) if self.total_times else 0,
            }
            
            return metrics


class SystemMonitor:
    """Monitors system resource usage"""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system resource metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        process_memory = self.process.memory_info()
        
        # Disk usage for current directory
        disk = psutil.disk_usage('.')
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_mb': process_memory.rss / 1024 / 1024,
            'memory_available_mb': memory.available / 1024 / 1024,
            'disk_percent': (disk.used / disk.total) * 100,
            'disk_free_gb': disk.free / 1024 / 1024 / 1024,
        }


class AlertManager:
    """Manages performance alerts and thresholds"""
    
    def __init__(self):
        self.alerts: List[PerformanceAlert] = []
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
        # Default thresholds
        self.thresholds = {
            'cpu_percent': {'warning': 70, 'critical': 90},
            'memory_percent': {'warning': 75, 'critical': 90},
            'avg_response_time': {'warning': 5000, 'critical': 10000},  # ms
            'p95_response_time': {'warning': 8000, 'critical': 15000},
            'failure_rate': {'warning': 0.05, 'critical': 0.1},  # 5% and 10%
            'queue_size': {'warning': 50, 'critical': 100},
        }
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback function to be called when alerts are triggered"""
        self.alert_callbacks.append(callback)
    
    def update_thresholds(self, thresholds: Dict[str, Dict[str, float]]):
        """Update alert thresholds"""
        self.thresholds.update(thresholds)
    
    def check_alerts(self, metrics: Dict[str, Any], system_metrics: Dict[str, Any]):
        """Check metrics against thresholds and generate alerts"""
        all_metrics = {**metrics, **system_metrics}
        
        for metric_name, value in all_metrics.items():
            if metric_name in self.thresholds:
                thresholds = self.thresholds[metric_name]
                
                # Check critical threshold
                if value >= thresholds.get('critical', float('inf')):
                    alert = PerformanceAlert(
                        alert_id=f"{metric_name}_critical_{int(time.time())}",
                        timestamp=datetime.now(),
                        severity='CRITICAL',
                        component='system',
                        metric=metric_name,
                        current_value=value,
                        threshold=thresholds['critical'],
                        message=f"Critical: {metric_name} is {value:.2f}, exceeding critical threshold of {thresholds['critical']}"
                    )
                    self._trigger_alert(alert)
                
                # Check warning threshold
                elif value >= thresholds.get('warning', float('inf')):
                    alert = PerformanceAlert(
                        alert_id=f"{metric_name}_warning_{int(time.time())}",
                        timestamp=datetime.now(),
                        severity='WARNING',
                        component='system',
                        metric=metric_name,
                        current_value=value,
                        threshold=thresholds['warning'],
                        message=f"Warning: {metric_name} is {value:.2f}, exceeding warning threshold of {thresholds['warning']}"
                    )
                    self._trigger_alert(alert)
    
    def _trigger_alert(self, alert: PerformanceAlert):
        """Trigger an alert"""
        self.alerts.append(alert)
        
        # Call all registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Error in alert callback: {e}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[PerformanceAlert]:
        """Get alerts from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff_time]


class PerformanceMonitor:
    """Main performance monitoring system"""
    
    def __init__(self, save_path: str = "performance_data"):
        self.metrics_collector = MetricsCollector()
        self.system_monitor = SystemMonitor()
        self.alert_manager = AlertManager()
        
        self.save_path = Path(save_path)
        self.save_path.mkdir(exist_ok=True)
        
        # Historical data
        self.snapshots: List[PerformanceSnapshot] = []
        
        # Monitoring control
        self.monitoring_active = False
        self.monitoring_interval = 30  # seconds
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Setup default alert callback
        self.alert_manager.add_alert_callback(self._log_alert)
    
    def start_monitoring(self, interval: int = 30):
        """Start continuous performance monitoring"""
        self.monitoring_interval = interval
        self.monitoring_active = True
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        print(f"Performance monitoring started with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                snapshot = self.capture_snapshot()
                self.snapshots.append(snapshot)
                
                # Check for alerts
                metrics = self.metrics_collector.get_current_metrics()
                system_metrics = self.system_monitor.get_system_metrics()
                self.alert_manager.check_alerts(metrics, system_metrics)
                
                # Keep only recent snapshots (last 24 hours)
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Short sleep before retry
    
    def capture_snapshot(self) -> PerformanceSnapshot:
        """Capture a performance snapshot"""
        metrics = self.metrics_collector.get_current_metrics()
        system_metrics = self.system_monitor.get_system_metrics()
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=system_metrics['cpu_percent'],
            memory_percent=system_metrics['memory_percent'],
            memory_used_mb=system_metrics['memory_used_mb'],
            memory_available_mb=system_metrics['memory_available_mb'],
            active_requests=metrics['active_requests'],
            queue_size=metrics['queue_size'],
            avg_response_time=metrics['avg_response_time'],
            p50_response_time=metrics['p50_response_time'],
            p95_response_time=metrics['p95_response_time'],
            p99_response_time=metrics['p99_response_time'],
            requests_per_second=metrics['requests_per_second'],
            successful_requests=metrics['successful_requests'],
            failed_requests=metrics['failed_requests'],
            avg_embedding_time=metrics['avg_embedding_time'],
            avg_retrieval_time=metrics['avg_retrieval_time'],
            avg_generation_time=metrics['avg_generation_time'],
            avg_total_time=metrics['avg_total_time']
        )
        
        return snapshot
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]
        
        if not recent_snapshots:
            return {}
        
        # Calculate aggregated metrics
        summary = {
            'period_hours': hours,
            'snapshot_count': len(recent_snapshots),
            'avg_cpu_percent': statistics.mean([s.cpu_percent for s in recent_snapshots]),
            'max_cpu_percent': max([s.cpu_percent for s in recent_snapshots]),
            'avg_memory_percent': statistics.mean([s.memory_percent for s in recent_snapshots]),
            'max_memory_percent': max([s.memory_percent for s in recent_snapshots]),
            'avg_response_time': statistics.mean([s.avg_response_time for s in recent_snapshots]),
            'max_response_time': max([s.avg_response_time for s in recent_snapshots]),
            'avg_requests_per_second': statistics.mean([s.requests_per_second for s in recent_snapshots]),
            'max_requests_per_second': max([s.requests_per_second for s in recent_snapshots]),
            'total_successful_requests': sum([s.successful_requests for s in recent_snapshots]),
            'total_failed_requests': sum([s.failed_requests for s in recent_snapshots]),
        }
        
        # Calculate success rate
        total_requests = summary['total_successful_requests'] + summary['total_failed_requests']
        summary['success_rate'] = summary['total_successful_requests'] / max(1, total_requests)
        
        return summary
    
    def save_performance_data(self, filename: Optional[str] = None):
        """Save performance data to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_data_{timestamp}.json"
        
        filepath = self.save_path / filename
        
        # Prepare data for JSON serialization
        data = {
            'snapshots': [
                {**asdict(snapshot), 'timestamp': snapshot.timestamp.isoformat()}
                for snapshot in self.snapshots
            ],
            'alerts': [
                {**asdict(alert), 'timestamp': alert.timestamp.isoformat()}
                for alert in self.alert_manager.alerts
            ],
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'monitoring_interval': self.monitoring_interval,
                'snapshot_count': len(self.snapshots),
                'alert_count': len(self.alert_manager.alerts)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Performance data saved to: {filepath}")
    
    def _log_alert(self, alert: PerformanceAlert):
        """Default alert logging callback"""
        print(f"[{alert.severity}] {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {alert.message}")
    
    # Context manager for request timing
    def track_request(self):
        """Context manager for tracking request performance"""
        return RequestTracker(self.metrics_collector)


class RequestTracker:
    """Context manager for tracking individual request performance"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.request_id: Optional[str] = None
        self.start_time: Optional[float] = None
        self.embedding_time: Optional[float] = None
        self.retrieval_time: Optional[float] = None
        self.generation_time: Optional[float] = None
    
    def __enter__(self):
        self.request_id = self.metrics_collector.record_request_start()
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.request_id and self.start_time:
            total_time = (time.time() - self.start_time) * 1000  # Convert to ms
            success = exc_type is None
            
            self.metrics_collector.record_request_end(
                self.request_id,
                success=success,
                embedding_time=self.embedding_time,
                retrieval_time=self.retrieval_time,
                generation_time=self.generation_time,
                total_time=total_time
            )
    
    def record_embedding_time(self, time_ms: float):
        """Record embedding processing time"""
        self.embedding_time = time_ms
    
    def record_retrieval_time(self, time_ms: float):
        """Record retrieval processing time"""
        self.retrieval_time = time_ms
    
    def record_generation_time(self, time_ms: float):
        """Record generation processing time"""
        self.generation_time = time_ms


# Example usage and testing
if __name__ == "__main__":
    # Demo performance monitoring
    monitor = PerformanceMonitor()
    
    # Start monitoring
    monitor.start_monitoring(interval=5)
    
    # Simulate some requests
    for i in range(5):
        with monitor.track_request() as tracker:
            time.sleep(0.1)  # Simulate embedding
            tracker.record_embedding_time(100)
            
            time.sleep(0.2)  # Simulate retrieval
            tracker.record_retrieval_time(200)
            
            time.sleep(0.3)  # Simulate generation
            tracker.record_generation_time(300)
        
        time.sleep(1)
    
    # Get summary
    summary = monitor.get_performance_summary(hours=1)
    print("Performance Summary:", json.dumps(summary, indent=2))
    
    # Save data
    monitor.save_performance_data()
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    print("Performance monitoring demo completed!")