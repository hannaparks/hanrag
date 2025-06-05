"""Circuit breaker pattern for resilient external service calls"""
import asyncio
from typing import Optional, Callable, Any, Dict
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import functools
from loguru import logger


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: int = 60  # Seconds before half-open
    success_threshold: int = 3  # Successes to close from half-open
    time_window: int = 60      # Time window for failure counting
    excluded_exceptions: tuple = ()  # Exceptions that don't trip breaker


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_changes: deque = field(default_factory=lambda: deque(maxlen=100))
    failure_reasons: deque = field(default_factory=lambda: deque(maxlen=50))


class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.circuit_opened_at: Optional[datetime] = None
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
        
        # Time-windowed failure tracking
        self.recent_failures: deque = deque()
    
    async def _count_recent_failures(self) -> int:
        """Count failures within the time window"""
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.config.time_window)
        
        # Remove old failures
        while self.recent_failures and self.recent_failures[0] < cutoff:
            self.recent_failures.popleft()
        
        return len(self.recent_failures)
    
    async def _change_state(self, new_state: CircuitState, reason: str):
        """Change circuit state and log"""
        old_state = self.state
        self.state = new_state
        
        logger.info(f"Circuit breaker '{self.name}' state change: {old_state.value} -> {new_state.value} ({reason})")
        
        self.stats.state_changes.append({
            "timestamp": datetime.utcnow(),
            "from": old_state.value,
            "to": new_state.value,
            "reason": reason
        })
        
        if new_state == CircuitState.OPEN:
            self.circuit_opened_at = datetime.utcnow()
        elif new_state == CircuitState.CLOSED:
            self.failure_count = 0
            self.success_count = 0
            self.recent_failures.clear()
    
    async def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset"""
        if self.state != CircuitState.OPEN:
            return False
        
        if not self.circuit_opened_at:
            return True
        
        elapsed = (datetime.utcnow() - self.circuit_opened_at).total_seconds()
        return elapsed >= self.config.recovery_timeout
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        async with self._lock:
            self.stats.total_calls += 1
            
            # Check if we should attempt reset
            if await self._should_attempt_reset():
                await self._change_state(CircuitState.HALF_OPEN, "Recovery timeout reached")
            
            # Reject calls if circuit is open
            if self.state == CircuitState.OPEN:
                self.stats.rejected_calls += 1
                raise CircuitBreakerError(f"Circuit breaker '{self.name}' is OPEN")
        
        # Execute the function
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            async with self._lock:
                self.stats.successful_calls += 1
                self.stats.last_success_time = datetime.utcnow()
                
                if self.state == CircuitState.HALF_OPEN:
                    self.success_count += 1
                    if self.success_count >= self.config.success_threshold:
                        await self._change_state(CircuitState.CLOSED, f"Recovery successful after {self.success_count} successes")
                
                return result
                
        except Exception as e:
            # Check if this exception should trip the breaker
            if isinstance(e, self.config.excluded_exceptions):
                raise
            
            async with self._lock:
                self.stats.failed_calls += 1
                self.stats.last_failure_time = datetime.utcnow()
                self.last_failure_time = datetime.utcnow()
                self.recent_failures.append(datetime.utcnow())
                
                # Record failure reason
                self.stats.failure_reasons.append({
                    "timestamp": datetime.utcnow(),
                    "error": type(e).__name__,
                    "message": str(e)[:200]
                })
                
                if self.state == CircuitState.CLOSED:
                    failure_count = await self._count_recent_failures()
                    if failure_count >= self.config.failure_threshold:
                        await self._change_state(CircuitState.OPEN, f"Failure threshold reached ({failure_count} failures)")
                
                elif self.state == CircuitState.HALF_OPEN:
                    await self._change_state(CircuitState.OPEN, "Failure during recovery")
                    self.success_count = 0
            
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.stats.total_calls,
            "successful_calls": self.stats.successful_calls,
            "failed_calls": self.stats.failed_calls,
            "rejected_calls": self.stats.rejected_calls,
            "success_rate": f"{(self.stats.successful_calls / max(1, self.stats.total_calls)) * 100:.1f}%",
            "last_failure": self.stats.last_failure_time.isoformat() if self.stats.last_failure_time else None,
            "last_success": self.stats.last_success_time.isoformat() if self.stats.last_success_time else None,
            "recent_state_changes": list(self.stats.state_changes)[-5:],
            "recent_failures": list(self.stats.failure_reasons)[-5:]
        }


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create a circuit breaker"""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    success_threshold: int = 3,
    excluded_exceptions: tuple = ()
):
    """Decorator for applying circuit breaker to functions"""
    
    def decorator(func):
        breaker_name = name or f"{func.__module__}.{func.__name__}"
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            excluded_exceptions=excluded_exceptions
        )
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            breaker = get_circuit_breaker(breaker_name, config)
            return await breaker.call(func, *args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            breaker = get_circuit_breaker(breaker_name, config)
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(breaker.call(func, *args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def get_all_circuit_breakers() -> Dict[str, Dict[str, Any]]:
    """Get stats for all circuit breakers"""
    return {name: breaker.get_stats() for name, breaker in _circuit_breakers.items()}