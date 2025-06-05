import asyncio
import traceback
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger

from ..utils.retry_utils import RetryConfig as BaseRetryConfig, retry_async
from ..utils.circuit_breaker import get_circuit_breaker, CircuitBreakerConfig


class ErrorType(Enum):
    """Classification of error types"""
    NETWORK_ERROR = "network_error"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    PERMISSION_DENIED = "permission_denied"
    FILE_NOT_FOUND = "file_not_found"
    PARSING_ERROR = "parsing_error"
    STORAGE_ERROR = "storage_error"
    TIMEOUT = "timeout"
    MEMORY_ERROR = "memory_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"          # Minor issues, can continue
    MEDIUM = "medium"    # Significant issues, retry recommended
    HIGH = "high"        # Major issues, may need manual intervention
    CRITICAL = "critical"  # System-level issues, stop processing


@dataclass
class ErrorInfo:
    """Detailed error information"""
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    exception: Optional[Exception] = None
    traceback: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "error_type": self.error_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "exception_type": type(self.exception).__name__ if self.exception else None,
            "traceback": self.traceback,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }


class ErrorRetryConfig:
    """Extended retry configuration for error-specific behavior"""
    
    def __init__(self):
        # Error type to retry config mapping
        self.error_configs = {
            ErrorType.RATE_LIMIT: BaseRetryConfig(
                max_attempts=5,
                initial_delay=5.0,
                max_delay=300.0,
                exponential_base=2.0
            ),
            ErrorType.NETWORK_ERROR: BaseRetryConfig(
                max_attempts=4,
                initial_delay=2.0,
                max_delay=60.0,
                exponential_base=1.5
            ),
            ErrorType.TIMEOUT: BaseRetryConfig(
                max_attempts=3,
                initial_delay=3.0,
                max_delay=30.0,
                exponential_base=2.0
            ),
            ErrorType.PARSING_ERROR: BaseRetryConfig(
                max_attempts=2,
                initial_delay=1.0,
                max_delay=5.0,
                exponential_base=1.0
            ),
            ErrorType.STORAGE_ERROR: BaseRetryConfig(
                max_attempts=4,
                initial_delay=2.0,
                max_delay=60.0,
                exponential_base=2.0
            )
        }
        
        # Default config for unknown errors
        self.default_config = BaseRetryConfig(
            max_attempts=3,
            initial_delay=1.0,
            max_delay=60.0
        )
        
        # Non-retryable error types
        self.non_retryable = {
            ErrorType.AUTHENTICATION,
            ErrorType.PERMISSION_DENIED,
            ErrorType.VALIDATION_ERROR
        }
    
    def get_config(self, error_type: ErrorType) -> BaseRetryConfig:
        """Get retry config for error type"""
        return self.error_configs.get(error_type, self.default_config)
    
    def should_retry(self, error_type: ErrorType) -> bool:
        """Check if error type is retryable"""
        return error_type not in self.non_retryable


class ErrorClassifier:
    """Classifies errors into types and severities"""
    
    def __init__(self):
        # Exception type mappings
        self.exception_mappings = {
            # Network errors
            (ConnectionError, TimeoutError, OSError): (ErrorType.NETWORK_ERROR, ErrorSeverity.MEDIUM),
            
            # Rate limiting
            (Exception,): self._check_rate_limit,
            
            # File system errors
            (FileNotFoundError,): (ErrorType.FILE_NOT_FOUND, ErrorSeverity.MEDIUM),
            (PermissionError,): (ErrorType.PERMISSION_DENIED, ErrorSeverity.HIGH),
            
            # Memory errors
            (MemoryError,): (ErrorType.MEMORY_ERROR, ErrorSeverity.CRITICAL),
            
            # Parsing errors
            (ValueError, TypeError): self._check_parsing_error,
            
            # Async errors
            (asyncio.TimeoutError,): (ErrorType.TIMEOUT, ErrorSeverity.MEDIUM),
        }
    
    def classify_error(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """Classify an exception into error type and severity"""
        
        context = context or {}
        
        # Check exception type mappings
        for exception_types, result in self.exception_mappings.items():
            if isinstance(exception, exception_types):
                if callable(result):
                    error_type, severity = result(exception, context)
                else:
                    error_type, severity = result
                
                return ErrorInfo(
                    error_type=error_type,
                    severity=severity,
                    message=str(exception),
                    exception=exception,
                    traceback=traceback.format_exc(),
                    context=context
                )
        
        # Default classification
        return ErrorInfo(
            error_type=ErrorType.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            message=str(exception),
            exception=exception,
            traceback=traceback.format_exc(),
            context=context
        )
    
    def _check_rate_limit(self, exception: Exception, context: Dict[str, Any]) -> tuple:
        """Check if exception is rate limiting"""
        
        error_msg = str(exception).lower()
        if any(phrase in error_msg for phrase in ['rate limit', 'too many requests', '429']):
            return ErrorType.RATE_LIMIT, ErrorSeverity.MEDIUM
        
        # Check for HTTP status codes in context
        if context.get('status_code') == 429:
            return ErrorType.RATE_LIMIT, ErrorSeverity.MEDIUM
        
        return ErrorType.UNKNOWN, ErrorSeverity.MEDIUM
    
    def _check_parsing_error(self, exception: Exception, context: Dict[str, Any]) -> tuple:
        """Check if exception is parsing related"""
        
        error_msg = str(exception).lower()
        parsing_keywords = ['parse', 'decode', 'format', 'invalid', 'malformed']
        
        if any(keyword in error_msg for keyword in parsing_keywords):
            return ErrorType.PARSING_ERROR, ErrorSeverity.MEDIUM
        
        return ErrorType.VALIDATION_ERROR, ErrorSeverity.MEDIUM


class ErrorRecoveryManager:
    """Manages error recovery and retry logic"""
    
    def __init__(self):
        self.error_retry_config = ErrorRetryConfig()
        self.classifier = ErrorClassifier()
        self.error_history: List[ErrorInfo] = []
    
    async def execute_with_retry(
        self,
        operation: Callable,
        operation_id: str,
        context: Dict[str, Any] = None,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with retry logic"""
        
        context = context or {}
        attempt = 1
        last_error = None
        
        # Get max attempts from all possible retry configs
        max_possible_attempts = max(
            config.max_attempts 
            for config in self.error_retry_config.error_configs.values()
        )
        
        while attempt <= max_possible_attempts:
            try:
                # Get circuit breaker for this operation
                circuit_breaker = get_circuit_breaker(
                    operation_id,
                    CircuitBreakerConfig(
                        failure_threshold=5,
                        recovery_timeout=300,  # 5 minutes
                        expected_exception=Exception
                    )
                )
                
                # Execute operation with circuit breaker
                async def _operation():
                    if asyncio.iscoroutinefunction(operation):
                        return await operation(*args, **kwargs)
                    else:
                        return operation(*args, **kwargs)
                
                if asyncio.iscoroutinefunction(operation):
                    result = await circuit_breaker.call_async(_operation)
                else:
                    result = await circuit_breaker.call_async(_operation)
                
                if attempt > 1:
                    logger.info(f"Operation {operation_id} succeeded on attempt {attempt}")
                
                return result
                
            except Exception as e:
                # Classify error
                error_info = self.classifier.classify_error(e, context)
                
                # Get retry config for this error type
                retry_config = self.error_retry_config.get_config(error_info.error_type)
                
                error_info.context.update({
                    "operation_id": operation_id,
                    "attempt": attempt,
                    "max_attempts": retry_config.max_attempts
                })
                
                # Record error
                self.error_history.append(error_info)
                last_error = error_info
                
                # Check if should retry
                if not self.error_retry_config.should_retry(error_info.error_type):
                    logger.error(f"Operation {operation_id} failed permanently: {error_info.message}")
                    break
                
                if attempt >= retry_config.max_attempts:
                    logger.error(f"Operation {operation_id} exhausted all retry attempts")
                    break
                
                # Calculate delay
                delay = retry_config.get_delay(attempt)
                
                logger.warning(
                    f"Operation {operation_id} failed (attempt {attempt}/{retry_config.max_attempts}), "
                    f"retrying in {delay:.1f}s: {error_info.message}"
                )
                
                if delay > 0:
                    await asyncio.sleep(delay)
                
                attempt += 1
        
        # All retries exhausted
        if last_error:
            raise Exception(f"Operation failed after {attempt-1} attempts: {last_error.message}") from last_error.exception
        else:
            raise Exception(f"Operation failed after {attempt-1} attempts")
    
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        
        if not self.error_history:
            return {"total_errors": 0}
        
        # Error type distribution
        error_types = {}
        severity_counts = {}
        
        for error in self.error_history:
            error_type = error.error_type.value
            severity = error.severity.value
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Recent errors (last hour)
        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_errors = [e for e in self.error_history if e.timestamp > recent_cutoff]
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "error_type_distribution": error_types,
            "severity_distribution": severity_counts,
            "circuit_breakers": self._get_circuit_breaker_states()
        }
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent errors"""
        
        recent_errors = sorted(self.error_history, key=lambda x: x.timestamp, reverse=True)
        return [error.to_dict() for error in recent_errors[:limit]]
    
    def clear_error_history(self):
        """Clear error history"""
        self.error_history.clear()
        logger.info("Error history cleared")
    
    def create_recovery_strategy(self, error_info: ErrorInfo) -> Dict[str, Any]:
        """Create a recovery strategy for an error"""
        
        strategies = {
            ErrorType.RATE_LIMIT: {
                "action": "backoff_and_retry",
                "recommended_delay": 30,
                "max_retries": 5,
                "description": "Rate limit hit, implement exponential backoff"
            },
            ErrorType.NETWORK_ERROR: {
                "action": "retry_with_backoff",
                "recommended_delay": 10,
                "max_retries": 3,
                "description": "Network issue, retry with increasing delays"
            },
            ErrorType.AUTHENTICATION: {
                "action": "refresh_credentials",
                "recommended_delay": 0,
                "max_retries": 1,
                "description": "Authentication failed, refresh tokens/credentials"
            },
            ErrorType.FILE_NOT_FOUND: {
                "action": "verify_path",
                "recommended_delay": 0,
                "max_retries": 0,
                "description": "File not found, verify path and permissions"
            },
            ErrorType.PARSING_ERROR: {
                "action": "alternative_parser",
                "recommended_delay": 1,
                "max_retries": 2,
                "description": "Parsing failed, try alternative parsing method"
            },
            ErrorType.MEMORY_ERROR: {
                "action": "reduce_batch_size",
                "recommended_delay": 5,
                "max_retries": 2,
                "description": "Memory exhausted, reduce processing batch size"
            },
            ErrorType.STORAGE_ERROR: {
                "action": "retry_storage",
                "recommended_delay": 5,
                "max_retries": 3,
                "description": "Storage operation failed, retry with backoff"
            }
        }
        
        strategy = strategies.get(error_info.error_type, {
            "action": "manual_intervention",
            "recommended_delay": 0,
            "max_retries": 0,
            "description": "Unknown error type, manual investigation required"
        })
        
        strategy.update({
            "error_type": error_info.error_type.value,
            "severity": error_info.severity.value,
            "timestamp": error_info.timestamp.isoformat()
        })
        
        return strategy


    def _get_circuit_breaker_states(self) -> Dict[str, Any]:
        """Get current circuit breaker states"""
        from ..utils.circuit_breaker import get_all_circuit_breakers
        
        states = {}
        for name, info in get_all_circuit_breakers().items():
            if info["state"] != "closed" or info["failure_count"] > 0:
                states[name] = info
        
        return states


# Decorator for automatic retry with error recovery
def with_error_recovery(
    operation_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
):
    """Decorator to add error recovery logic to functions"""
    
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            recovery_manager = ErrorRecoveryManager()
            op_id = operation_id or f"{func.__module__}.{func.__name__}"
            
            return await recovery_manager.execute_with_retry(
                func, op_id, context, *args, **kwargs
            )
        
        def sync_wrapper(*args, **kwargs):
            recovery_manager = ErrorRecoveryManager()
            op_id = operation_id or f"{func.__module__}.{func.__name__}"
            
            return asyncio.run(recovery_manager.execute_with_retry(
                func, op_id, context, *args, **kwargs
            ))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator