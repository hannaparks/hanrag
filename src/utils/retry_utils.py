"""Retry utilities with exponential backoff and jitter"""
import asyncio
import random
from typing import Optional, Callable, Any, Type, Tuple
import functools
from loguru import logger


class RetryConfig:
    """Configuration for retry behavior"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_on: Optional[Tuple[Type[Exception], ...]] = None,
        retry_condition: Optional[Callable[[Exception], bool]] = None
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_on = retry_on or (Exception,)
        self.retry_condition = retry_condition
    
    def should_retry(self, exception: Exception) -> bool:
        """Check if exception should trigger retry"""
        if not isinstance(exception, self.retry_on):
            return False
        
        if self.retry_condition:
            return self.retry_condition(exception)
        
        return True
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff"""
        delay = min(
            self.initial_delay * (self.exponential_base ** (attempt - 1)),
            self.max_delay
        )
        
        if self.jitter:
            # Add jitter to prevent thundering herd
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay


async def retry_async(
    func: Callable,
    config: Optional[RetryConfig] = None,
    context: Optional[str] = None
) -> Any:
    """Retry an async function with exponential backoff"""
    config = config or RetryConfig()
    context = context or func.__name__
    
    last_exception = None
    
    for attempt in range(1, config.max_attempts + 1):
        try:
            return await func()
            
        except Exception as e:
            last_exception = e
            
            if not config.should_retry(e):
                logger.error(f"[{context}] Non-retryable error: {type(e).__name__}: {str(e)}")
                raise
            
            if attempt == config.max_attempts:
                logger.error(f"[{context}] All {config.max_attempts} attempts failed")
                raise
            
            delay = config.get_delay(attempt)
            logger.warning(
                f"[{context}] Attempt {attempt}/{config.max_attempts} failed: "
                f"{type(e).__name__}: {str(e)}. Retrying in {delay:.1f}s..."
            )
            
            await asyncio.sleep(delay)
    
    # Should never reach here
    if last_exception:
        raise last_exception


def retry_sync(
    func: Callable,
    config: Optional[RetryConfig] = None,
    context: Optional[str] = None
) -> Any:
    """Retry a sync function with exponential backoff"""
    config = config or RetryConfig()
    context = context or func.__name__
    
    last_exception = None
    
    for attempt in range(1, config.max_attempts + 1):
        try:
            return func()
            
        except Exception as e:
            last_exception = e
            
            if not config.should_retry(e):
                logger.error(f"[{context}] Non-retryable error: {type(e).__name__}: {str(e)}")
                raise
            
            if attempt == config.max_attempts:
                logger.error(f"[{context}] All {config.max_attempts} attempts failed")
                raise
            
            delay = config.get_delay(attempt)
            logger.warning(
                f"[{context}] Attempt {attempt}/{config.max_attempts} failed: "
                f"{type(e).__name__}: {str(e)}. Retrying in {delay:.1f}s..."
            )
            
            import time
            time.sleep(delay)
    
    # Should never reach here
    if last_exception:
        raise last_exception


def with_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_on: Optional[Tuple[Type[Exception], ...]] = None,
    retry_condition: Optional[Callable[[Exception], bool]] = None
):
    """Decorator for adding retry logic to functions"""
    
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retry_on=retry_on,
        retry_condition=retry_condition
    )
    
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            async def _func():
                return await func(*args, **kwargs)
            
            return await retry_async(_func, config, func.__name__)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            def _func():
                return func(*args, **kwargs)
            
            return retry_sync(_func, config, func.__name__)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Common retry configurations
RETRY_NETWORK = RetryConfig(
    max_attempts=3,
    initial_delay=1.0,
    retry_on=(ConnectionError, TimeoutError, asyncio.TimeoutError)
)

RETRY_API = RetryConfig(
    max_attempts=5,
    initial_delay=2.0,
    max_delay=30.0,
    retry_condition=lambda e: (
        hasattr(e, 'status_code') and 
        e.status_code in [429, 500, 502, 503, 504]
    )
)

RETRY_DATABASE = RetryConfig(
    max_attempts=3,
    initial_delay=0.5,
    max_delay=5.0,
    jitter=True
)