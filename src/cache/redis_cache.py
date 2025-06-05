"""Redis caching for query results"""
import json
import hashlib
from typing import Optional, Dict, Any
from datetime import datetime
import redis.asyncio as redis
from redis.exceptions import RedisError
from loguru import logger

from src.config.settings import settings


class RedisCache:
    """Redis cache manager for query results"""
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = 0,
        password: str = None,
        default_ttl: int = 3600,  # 1 hour default
        key_prefix: str = "rag:"
    ):
        self.host = host or getattr(settings, 'REDIS_HOST', 'localhost')
        self.port = port or getattr(settings, 'REDIS_PORT', 6379)
        self.db = db or getattr(settings, 'REDIS_DB', 0)
        self.password = password or getattr(settings, 'REDIS_PASSWORD', None)
        self.default_ttl = default_ttl or getattr(settings, 'CACHE_TTL', 3600)
        self.key_prefix = key_prefix
        
        self._client: Optional[redis.Redis] = None
        self._connected = False
        
    async def connect(self) -> bool:
        """Connect to Redis"""
        try:
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self._client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            return True
            
        except (RedisError, ConnectionError) as e:
            logger.warning(f"Failed to connect to Redis: {e}. Cache disabled.")
            self._connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self._client:
            await self._client.close()
            self._connected = False
            logger.info("Disconnected from Redis")
    
    def _generate_key(self, query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key from query and parameters"""
        # Create a deterministic key from query and params
        key_data = {
            "query": query.strip().lower(),
            "params": params or {}
        }
        
        # Sort params for consistent hashing
        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]
        
        return f"{self.key_prefix}query:{key_hash}"
    
    async def get(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached result for query"""
        if not self._connected or not self._client:
            return None
        
        key = self._generate_key(query, params)
        
        try:
            # Get cached value
            cached = await self._client.get(key)
            
            if cached:
                # Update access time for analytics
                await self._client.hincrby(f"{key}:meta", "hits", 1)
                await self._client.hset(f"{key}:meta", "last_accessed", datetime.utcnow().isoformat())
                
                result = json.loads(cached)
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return result
            
            logger.debug(f"Cache miss for query: {query[:50]}...")
            return None
            
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(
        self,
        query: str,
        result: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache query result"""
        if not self._connected or not self._client:
            return False
        
        key = self._generate_key(query, params)
        ttl = ttl or self.default_ttl
        
        try:
            # Store result
            await self._client.setex(
                key,
                ttl,
                json.dumps(result)
            )
            
            # Store metadata
            meta_key = f"{key}:meta"
            await self._client.hset(meta_key, mapping={
                "query": query[:200],  # Store truncated query
                "created": datetime.utcnow().isoformat(),
                "last_accessed": datetime.utcnow().isoformat(),
                "hits": "0",
                "ttl": str(ttl)
            })
            await self._client.expire(meta_key, ttl + 86400)  # Keep meta 1 day longer
            
            logger.debug(f"Cached result for query: {query[:50]}... (TTL: {ttl}s)")
            return True
            
        except (RedisError, json.JSONEncodeError) as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def invalidate(
        self,
        query: Optional[str] = None,
        pattern: Optional[str] = None
    ) -> int:
        """Invalidate cached results"""
        if not self._connected or not self._client:
            return 0
        
        try:
            if query:
                # Invalidate specific query
                key = self._generate_key(query)
                deleted = await self._client.delete(key, f"{key}:meta")
                return deleted // 2  # Count pairs as single invalidation
            
            elif pattern:
                # Invalidate by pattern
                pattern_key = f"{self.key_prefix}query:*{pattern}*"
                keys = []
                async for key in self._client.scan_iter(match=pattern_key):
                    keys.extend([key, f"{key}:meta"])
                
                if keys:
                    deleted = await self._client.delete(*keys)
                    return deleted // 2
                return 0
            
            else:
                # Invalidate all query cache
                pattern_key = f"{self.key_prefix}query:*"
                keys = []
                async for key in self._client.scan_iter(match=pattern_key):
                    keys.append(key)
                
                if keys:
                    deleted = await self._client.delete(*keys)
                    return deleted
                return 0
                
        except RedisError as e:
            logger.error(f"Cache invalidation error: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self._connected or not self._client:
            return {"status": "disconnected"}
        
        try:
            # Get all cache keys
            pattern = f"{self.key_prefix}query:*"
            total_keys = 0
            total_hits = 0
            
            meta_keys = []
            async for key in self._client.scan_iter(match=f"{pattern}:meta"):
                meta_keys.append(key)
                total_keys += 1
            
            # Get hit counts
            if meta_keys:
                pipeline = self._client.pipeline()
                for key in meta_keys:
                    pipeline.hget(key, "hits")
                
                hits = await pipeline.execute()
                total_hits = sum(int(h or 0) for h in hits)
            
            # Get Redis info
            info = await self._client.info()
            
            return {
                "status": "connected",
                "total_cached_queries": total_keys,
                "total_cache_hits": total_hits,
                "hit_rate": f"{(total_hits / (total_keys + 1)) * 100:.1f}%" if total_keys > 0 else "0%",
                "memory_used": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "uptime_days": info.get("uptime_in_days", 0)
            }
            
        except RedisError as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"status": "error", "error": str(e)}
    
    async def clear_all(self) -> bool:
        """Clear all cached data with this key prefix"""
        if not self._connected or not self._client:
            logger.warning("Redis client not connected")
            return False
        
        try:
            # Use SCAN to find all keys with our prefix
            cursor = 0
            deleted_count = 0
            
            while True:
                cursor, keys = await self._client.scan(
                    cursor=cursor,
                    match=f"{self.key_prefix}*",
                    count=100
                )
                
                if keys:
                    # Delete keys in batches
                    deleted = await self._client.delete(*keys)
                    deleted_count += deleted
                    logger.debug(f"Deleted {deleted} keys from cache")
                
                if cursor == 0:
                    break
            
            logger.info(f"Cleared {deleted_count} keys from Redis cache")
            return True
            
        except RedisError as e:
            logger.error(f"Failed to clear cache: {e}")
            return False


# Global cache instance
cache_manager: Optional[RedisCache] = None


async def initialize_cache() -> Optional[RedisCache]:
    """Initialize global cache instance"""
    global cache_manager
    
    if not getattr(settings, 'ENABLE_REDIS_CACHE', True):
        logger.info("Redis cache disabled by configuration")
        return None
    
    cache_manager = RedisCache()
    
    if await cache_manager.connect():
        return cache_manager
    else:
        logger.warning("Running without Redis cache")
        cache_manager = None
        return None


async def get_cache() -> Optional[RedisCache]:
    """Get cache instance (for dependency injection)"""
    return cache_manager