"""Rate limiting for API endpoints"""
from fastapi import HTTPException, Request, status
from typing import Dict, Optional, Tuple
import time
import asyncio
from collections import defaultdict, deque
import hashlib

from src.config.settings import settings


class RateLimiter:
    """Token bucket rate limiter implementation"""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size
        
        # Storage for rate limit data
        self._minute_buckets: Dict[str, deque] = defaultdict(lambda: deque(maxlen=requests_per_minute))
        self._hour_buckets: Dict[str, deque] = defaultdict(lambda: deque(maxlen=requests_per_hour))
        self._burst_tokens: Dict[str, int] = defaultdict(lambda: burst_size)
        self._last_refill: Dict[str, float] = {}
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    def _get_client_id(self, request: Request, api_key: Optional[str] = None) -> str:
        """Get unique client identifier from request"""
        if api_key:
            # Use hashed API key if available
            return hashlib.sha256(api_key.encode()).hexdigest()[:16]
        
        # Fall back to IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        return client_ip
    
    async def _refill_burst_tokens(self, client_id: str) -> None:
        """Refill burst tokens based on time elapsed"""
        current_time = time.time()
        last_refill = self._last_refill.get(client_id, current_time)
        
        # Refill 1 token per 6 seconds (10 per minute)
        time_elapsed = current_time - last_refill
        tokens_to_add = int(time_elapsed / 6)
        
        if tokens_to_add > 0:
            self._burst_tokens[client_id] = min(
                self._burst_tokens[client_id] + tokens_to_add,
                self.burst_size
            )
            self._last_refill[client_id] = current_time
    
    async def _clean_old_requests(self, client_id: str) -> None:
        """Remove requests older than the time window"""
        current_time = time.time()
        
        # Clean minute bucket
        minute_bucket = self._minute_buckets[client_id]
        while minute_bucket and minute_bucket[0] < current_time - 60:
            minute_bucket.popleft()
        
        # Clean hour bucket
        hour_bucket = self._hour_buckets[client_id]
        while hour_bucket and hour_bucket[0] < current_time - 3600:
            hour_bucket.popleft()
    
    async def check_rate_limit(
        self,
        request: Request,
        api_key: Optional[str] = None,
        weight: int = 1
    ) -> Tuple[bool, Optional[Dict[str, any]]]:
        """
        Check if request should be rate limited
        
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        async with self._lock:
            client_id = self._get_client_id(request, api_key)
            current_time = time.time()
            
            # Clean old requests
            await self._clean_old_requests(client_id)
            
            # Refill burst tokens
            await self._refill_burst_tokens(client_id)
            
            # Get current buckets
            minute_bucket = self._minute_buckets[client_id]
            hour_bucket = self._hour_buckets[client_id]
            burst_tokens = self._burst_tokens[client_id]
            
            # Check limits
            minute_count = len(minute_bucket)
            hour_count = len(hour_bucket)
            
            # Calculate remaining limits
            minute_remaining = self.requests_per_minute - minute_count
            hour_remaining = self.requests_per_hour - hour_count
            
            # Rate limit info for headers
            rate_limit_info = {
                "X-RateLimit-Limit-Minute": str(self.requests_per_minute),
                "X-RateLimit-Remaining-Minute": str(max(0, minute_remaining)),
                "X-RateLimit-Limit-Hour": str(self.requests_per_hour),
                "X-RateLimit-Remaining-Hour": str(max(0, hour_remaining)),
                "X-RateLimit-Burst-Remaining": str(burst_tokens),
                "X-RateLimit-Reset": str(int(current_time + 60))
            }
            
            # Check if any limit is exceeded
            if minute_count >= self.requests_per_minute:
                # Check burst tokens
                if burst_tokens >= weight:
                    self._burst_tokens[client_id] -= weight
                    rate_limit_info["X-RateLimit-Burst-Remaining"] = str(self._burst_tokens[client_id])
                else:
                    return False, rate_limit_info
            
            if hour_count >= self.requests_per_hour:
                return False, rate_limit_info
            
            # Record the request
            for _ in range(weight):
                minute_bucket.append(current_time)
                hour_bucket.append(current_time)
            
            return True, rate_limit_info


# Global rate limiter instances
default_limiter = RateLimiter(
    requests_per_minute=getattr(settings, 'RATE_LIMIT_PER_MINUTE', 60),
    requests_per_hour=getattr(settings, 'RATE_LIMIT_PER_HOUR', 1000),
    burst_size=getattr(settings, 'RATE_LIMIT_BURST_SIZE', 10)
)

# Stricter limiter for expensive operations
expensive_limiter = RateLimiter(
    requests_per_minute=getattr(settings, 'EXPENSIVE_RATE_LIMIT_PER_MINUTE', 10),
    requests_per_hour=getattr(settings, 'EXPENSIVE_RATE_LIMIT_PER_HOUR', 100),
    burst_size=getattr(settings, 'EXPENSIVE_RATE_LIMIT_BURST_SIZE', 3)
)


async def rate_limit(
    request: Request,
    api_key: Optional[str] = None,
    limiter: RateLimiter = default_limiter,
    weight: int = 1
) -> None:
    """
    Rate limit decorator/dependency for FastAPI endpoints
    
    Args:
        request: FastAPI request object
        api_key: Optional API key for per-key rate limiting
        limiter: RateLimiter instance to use
        weight: Request weight (for expensive operations)
    
    Raises:
        HTTPException: If rate limit is exceeded
    """
    is_allowed, rate_limit_info = await limiter.check_rate_limit(request, api_key, weight)
    
    # Add rate limit headers to response
    if rate_limit_info:
        for header, value in rate_limit_info.items():
            request.state.rate_limit_headers = rate_limit_info
    
    if not is_allowed:
        # Calculate retry after
        retry_after = 60  # Default to 1 minute
        
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers={
                **rate_limit_info,
                "Retry-After": str(retry_after)
            }
        )


# Dependency functions for different rate limit tiers
async def require_default_rate_limit(
    request: Request,
    api_key: Optional[str] = None
) -> None:
    """Default rate limit for most endpoints"""
    await rate_limit(request, api_key, default_limiter)


async def require_expensive_rate_limit(
    request: Request,
    api_key: Optional[str] = None
) -> None:
    """Stricter rate limit for expensive operations"""
    await rate_limit(request, api_key, expensive_limiter, weight=5)