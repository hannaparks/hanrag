import asyncio
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    VectorParams, 
    Distance, 
    CollectionInfo,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    ScoredPoint
)
import uuid
from loguru import logger
from contextlib import asynccontextmanager
import threading
import queue
import time
from ..config.settings import settings
from ..utils.circuit_breaker import circuit_breaker
from ..utils.retry_utils import with_retry
from ..monitoring.metrics import track_vector_operation, update_connection_pool_metrics


class QdrantConnectionPool:
    """Connection pool for Qdrant clients"""
    
    def __init__(self, host: str, port: int, pool_size: int = 10, max_idle_time: int = 300):
        self.host = host
        self.port = port
        self.pool_size = pool_size
        self.max_idle_time = max_idle_time
        
        # Connection pool
        self._pool = queue.Queue(maxsize=pool_size)
        self._all_connections = []
        self._lock = threading.Lock()
        self._last_use_time = {}
        
        # Initialize pool
        self._initialize_pool()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_idle_connections, daemon=True)
        self._cleanup_thread.start()
    
    def _initialize_pool(self):
        """Initialize the connection pool with clients"""
        for _ in range(self.pool_size):
            client = self._create_client()
            self._pool.put(client)
            self._all_connections.append(client)
            self._last_use_time[id(client)] = time.time()
    
    def _create_client(self) -> QdrantClient:
        """Create a new Qdrant client"""
        return QdrantClient(
            host=self.host,
            port=self.port,
            timeout=60,
            prefer_grpc=False  # Use HTTP instead of gRPC
        )
    
    @asynccontextmanager
    async def get_client(self):
        """Get a client from the pool"""
        client = None
        try:
            # Get client from pool (with timeout to prevent deadlock)
            client = self._pool.get(timeout=5.0)
            self._last_use_time[id(client)] = time.time()
            
            # Update metrics
            active_connections = len(self._all_connections) - self._pool.qsize()
            update_connection_pool_metrics("qdrant", len(self._all_connections), active_connections)
            
            yield client
        except queue.Empty:
            # Pool exhausted, create temporary client
            logger.warning("Connection pool exhausted, creating temporary client")
            temp_client = self._create_client()
            yield temp_client
        finally:
            # Return client to pool
            if client and client in self._all_connections:
                self._pool.put(client)
                self._last_use_time[id(client)] = time.time()
    
    def _cleanup_idle_connections(self):
        """Cleanup idle connections periodically"""
        while True:
            time.sleep(60)  # Check every minute
            
            with self._lock:
                current_time = time.time()
                
                # Find idle connections
                for client in list(self._all_connections):
                    client_id = id(client)
                    if client_id in self._last_use_time:
                        idle_time = current_time - self._last_use_time[client_id]
                        
                        if idle_time > self.max_idle_time:
                            try:
                                # Remove from pool and close
                                self._pool.get_nowait()  # Remove if in pool
                                self._all_connections.remove(client)
                                del self._last_use_time[client_id]
                                
                                # Create new client to maintain pool size
                                new_client = self._create_client()
                                self._pool.put(new_client)
                                self._all_connections.append(new_client)
                                self._last_use_time[id(new_client)] = current_time
                                
                                logger.debug(f"Replaced idle connection (idle for {idle_time:.1f}s)")
                            except queue.Empty:
                                pass  # Client not in pool, skip


class QdrantManager:
    """Manages Qdrant vector database operations with connection pooling"""
    
    def __init__(self):
        # Initialize connection pool
        self.pool = QdrantConnectionPool(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            pool_size=getattr(settings, 'QDRANT_POOL_SIZE', 10),
            max_idle_time=getattr(settings, 'QDRANT_MAX_IDLE_TIME', 300)
        )
        
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.vector_size = settings.EMBEDDING_DIMENSION
        
    @track_vector_operation("create_collection")
    @circuit_breaker(
        name="qdrant_db",
        failure_threshold=3,
        recovery_timeout=60,
        success_threshold=2
    )
    @with_retry(
        max_attempts=3,
        initial_delay=1.0,
        retry_on=(ConnectionError, TimeoutError, asyncio.TimeoutError)
    )
    async def create_collection(self, collection_name: Optional[str] = None) -> bool:
        """Create a new collection if it doesn't exist"""
        collection_name = collection_name or self.collection_name
        
        try:
            # Check if collection exists
            async with self.pool.get_client() as client:
                collections = client.get_collections()
                existing_names = [col.name for col in collections.collections]
                
                if collection_name in existing_names:
                    logger.info(f"Collection '{collection_name}' already exists")
                    return True
                    
                # Create collection
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
            
            logger.info(f"Created collection '{collection_name}' with {self.vector_size} dimensions")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            return False
    
    @track_vector_operation("upsert")
    async def upsert_points(
        self, 
        points: List[Dict[str, Any]], 
        collection_name: Optional[str] = None
    ) -> bool:
        """Upsert points (vectors with metadata) to collection"""
        collection_name = collection_name or self.collection_name
        
        try:
            qdrant_points = []
            
            for point in points:
                # Generate UUID if not provided
                point_id = point.get("id", str(uuid.uuid4()))
                
                qdrant_point = PointStruct(
                    id=point_id,
                    vector=point["vector"],
                    payload=point.get("payload", {})
                )
                qdrant_points.append(qdrant_point)
            
            # Upsert points in batches
            batch_size = 100
            async with self.pool.get_client() as client:
                for i in range(0, len(qdrant_points), batch_size):
                    batch = qdrant_points[i:i + batch_size]
                    client.upsert(
                        collection_name=collection_name,
                        points=batch
                    )
            
            logger.info(f"Upserted {len(qdrant_points)} points to '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert points to '{collection_name}': {e}")
            return False
    
    @track_vector_operation("search")
    @circuit_breaker(
        name="qdrant_db",
        failure_threshold=3,
        recovery_timeout=60,
        success_threshold=2
    )
    @with_retry(
        max_attempts=3,
        initial_delay=0.5,
        retry_on=(ConnectionError, TimeoutError, asyncio.TimeoutError)
    )
    async def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 50,
        collection_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ScoredPoint]:
        """Search for similar vectors with optional filtering"""
        collection_name = collection_name or self.collection_name
        
        try:
            # Build filter if provided
            query_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    # Handle nested metadata fields
                    if key in ["channel_id", "team_id", "ingested_by", "source_type", "date_range"]:
                        field_key = f"metadata.{key}"
                    else:
                        field_key = key
                    
                    # Special handling for date ranges
                    if key == "date_range" and isinstance(value, dict):
                        if "start" in value:
                            conditions.append(
                                FieldCondition(
                                    key=f"metadata.timestamp",
                                    range=models.Range(gte=value["start"])
                                )
                            )
                        if "end" in value:
                            conditions.append(
                                FieldCondition(
                                    key=f"metadata.timestamp",
                                    range=models.Range(lte=value["end"])
                                )
                            )
                    else:
                        conditions.append(
                            FieldCondition(
                                key=field_key,
                                match=MatchValue(value=value)
                            )
                        )
                
                if conditions:
                    query_filter = Filter(must=conditions)
            
            # Perform search with optional filtering
            async with self.pool.get_client() as client:
                search_result = client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False
                )
            
            logger.debug(f"Found {len(search_result)} similar vectors" + 
                        (f" with filters: {filters}" if filters else ""))
            return search_result
            
        except Exception as e:
            logger.error(f"Failed to search vectors in '{collection_name}': {e}")
            return []
    
    async def hybrid_search(
        self, 
        query_vector: List[float],
        query_text: str,
        top_k: int = 50,
        collection_name: Optional[str] = None
    ) -> List[ScoredPoint]:
        """Hybrid search combining vector similarity and BM25 lexical search"""
        collection_name = collection_name or self.collection_name
        
        try:
            # For now, delegate to the dedicated hybrid retriever
            # The actual fusion logic is implemented in the HybridRetriever class
            # This method exists for compatibility but should use HybridRetriever directly
            
            logger.debug("QdrantManager.hybrid_search delegating to vector search - "
                        "use HybridRetriever for full hybrid functionality")
            
            # Vector search as fallback
            vector_results = await self.search_vectors(
                query_vector=query_vector,
                top_k=top_k,
                collection_name=collection_name
            )
            
            return vector_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed in QdrantManager: {e}")
            return []
    
    async def search_with_parent_context(
        self,
        query_vector: List[float],
        top_k: int = 50,
        include_parents: bool = True,
        collection_name: Optional[str] = None
    ) -> Dict[str, List[Any]]:
        """Search with hierarchical context - returns both child matches and their parents"""
        collection_name = collection_name or self.collection_name
        
        try:
            # First, search for child chunks (most relevant)
            child_results = await self.search_vectors(
                query_vector=query_vector,
                top_k=top_k,
                collection_name=collection_name
            )
            
            parent_results = []
            parent_ids = set()
            
            if include_parents:
                # Collect parent IDs from child results
                for result in child_results:
                    parent_id = result.payload.get("parent_id")
                    if parent_id and parent_id not in parent_ids:
                        parent_ids.add(parent_id)
                
                # Fetch parent chunks
                if parent_ids:
                    parent_results = await self._get_chunks_by_ids(
                        list(parent_ids), 
                        collection_name
                    )
            
            return {
                "children": child_results,
                "parents": parent_results
            }
            
        except Exception as e:
            logger.error(f"Failed hierarchical search in '{collection_name}': {e}")
            return {"children": [], "parents": []}
    
    async def _get_chunks_by_ids(
        self, 
        chunk_ids: List[str], 
        collection_name: Optional[str] = None
    ) -> List[Any]:
        """Retrieve specific chunks by their IDs"""
        collection_name = collection_name or self.collection_name
        
        try:
            # Use scroll with ID filter to get specific chunks
            results = []
            
            for chunk_id in chunk_ids:
                try:
                    # Retrieve individual point
                    async with self.pool.get_client() as client:
                        point = client.retrieve(
                            collection_name=collection_name,
                            ids=[chunk_id],
                            with_payload=True,
                            with_vectors=False
                        )
                    
                    if point:
                        # Create a simple object with the required attributes
                        # This avoids ScoredPoint version compatibility issues
                        mock_scored_point = type('MockScoredPoint', (), {
                            'id': point[0].id,
                            'score': 1.0,  # No score for direct retrieval
                            'payload': point[0].payload
                        })()
                        
                        results.append(mock_scored_point)
                        
                except Exception as e:
                    logger.warning(f"Failed to retrieve chunk {chunk_id}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get chunks by IDs: {e}")
            return []
    
    async def purge_collection(
        self, 
        collection_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        confirm: bool = False
    ) -> Dict[str, Any]:
        """Purge (delete) points from collection with optional filtering"""
        collection_name = collection_name or self.collection_name
        
        if not confirm:
            return {
                "success": False,
                "error": "Purge operation requires explicit confirmation",
                "message": "Use confirm=True to proceed with purge"
            }
        
        try:
            # Get collection info before purging
            collection_info = await self.get_collection_info(collection_name)
            if not collection_info:
                return {
                    "success": False,
                    "error": f"Collection '{collection_name}' not found"
                }
            
            original_count = collection_info.points_count
            
            if filters:
                # Selective purge based on filters
                return await self._purge_with_filters(collection_name, filters)
            else:
                # Full collection purge - recreate collection
                return await self._purge_full_collection(collection_name, original_count)
                
        except Exception as e:
            logger.error(f"Failed to purge collection '{collection_name}': {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _purge_with_filters(
        self, 
        collection_name: str, 
        filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Purge points matching specific filters"""
        
        try:
            # Build filter conditions
            conditions = []
            for key, value in filters.items():
                # Handle nested metadata fields
                if key in ["channel_id", "team_id", "ingested_by", "source_type"]:
                    field_key = f"metadata.{key}"
                else:
                    field_key = key
                
                conditions.append(
                    FieldCondition(
                        key=field_key,
                        match=MatchValue(value=value)
                    )
                )
            
            query_filter = Filter(must=conditions)
            
            # First, get points to be deleted (for counting)
            async with self.pool.get_client() as client:
                points_to_delete = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=query_filter,
                    limit=10000,  # Large batch
                    with_payload=False,
                    with_vectors=False
                )
                
                points_count = len(points_to_delete[0])
                
                if points_count == 0:
                    return {
                        "success": True,
                        "message": "No points matched the filter criteria",
                        "deleted_count": 0
                    }
                
                # Delete the points
                point_ids = [point.id for point in points_to_delete[0]]
                
                client.delete(
                    collection_name=collection_name,
                    points_selector=point_ids
                )
            
            logger.info(f"Purged {points_count} points from '{collection_name}' with filters: {filters}")
            
            return {
                "success": True,
                "message": f"Successfully purged {points_count} points matching filters",
                "deleted_count": points_count,
                "filters_applied": filters
            }
            
        except Exception as e:
            logger.error(f"Failed to purge with filters: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _purge_full_collection(
        self, 
        collection_name: str, 
        original_count: int
    ) -> Dict[str, Any]:
        """Completely purge a collection by recreating it"""
        
        try:
            # Delete the entire collection
            delete_success = await self.delete_collection(collection_name)
            if not delete_success:
                return {
                    "success": False,
                    "error": f"Failed to delete collection '{collection_name}'"
                }
            
            # Recreate the collection
            create_success = await self.create_collection(collection_name)
            if not create_success:
                return {
                    "success": False,
                    "error": f"Failed to recreate collection '{collection_name}'"
                }
            
            logger.info(f"Fully purged collection '{collection_name}' - deleted {original_count} points")
            
            return {
                "success": True,
                "message": f"Successfully purged entire collection '{collection_name}'",
                "deleted_count": original_count,
                "operation": "full_collection_recreate"
            }
            
        except Exception as e:
            logger.error(f"Failed to fully purge collection: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_purge_preview(
        self, 
        collection_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Preview what would be deleted in a purge operation"""
        collection_name = collection_name or self.collection_name
        
        try:
            collection_info = await self.get_collection_info(collection_name)
            if not collection_info:
                return {
                    "success": False,
                    "error": f"Collection '{collection_name}' not found"
                }
            
            total_points = collection_info.points_count
            
            if not filters:
                # Full collection purge preview
                return {
                    "success": True,
                    "collection_name": collection_name,
                    "total_points": total_points,
                    "points_to_delete": total_points,
                    "operation": "full_collection_purge",
                    "warning": "This will delete ALL data in the collection!"
                }
            else:
                # Filtered purge preview
                conditions = []
                for key, value in filters.items():
                    if key in ["channel_id", "team_id", "ingested_by", "source_type"]:
                        field_key = f"metadata.{key}"
                    else:
                        field_key = key
                    
                    conditions.append(
                        FieldCondition(
                            key=field_key,
                            match=MatchValue(value=value)
                        )
                    )
                
                query_filter = Filter(must=conditions)
                
                # Count matching points
                async with self.pool.get_client() as client:
                    matching_points = client.scroll(
                        collection_name=collection_name,
                        scroll_filter=query_filter,
                        limit=10000,  # Large batch for counting
                        with_payload=False,
                        with_vectors=False
                    )
                
                points_to_delete = len(matching_points[0])
                
                return {
                    "success": True,
                    "collection_name": collection_name,
                    "total_points": total_points,
                    "points_to_delete": points_to_delete,
                    "filters": filters,
                    "operation": "filtered_purge",
                    "warning": f"This will delete {points_to_delete} points matching the filter criteria"
                }
                
        except Exception as e:
            logger.error(f"Failed to get purge preview: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_collection_info(self, collection_name: Optional[str] = None) -> Optional[CollectionInfo]:
        """Get information about a collection"""
        collection_name = collection_name or self.collection_name
        
        try:
            async with self.pool.get_client() as client:
                info = client.get_collection(collection_name)
                logger.debug(f"Collection '{collection_name}' info: {info}")
                return info
        except Exception as e:
            logger.error(f"Failed to get collection info for '{collection_name}': {e}")
            return None
    
    async def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """Delete a collection"""
        collection_name = collection_name or self.collection_name
        
        try:
            async with self.pool.get_client() as client:
                client.delete_collection(collection_name)
                logger.info(f"Deleted collection '{collection_name}'")
                return True
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            return False
    
    async def count_points(self, collection_name: Optional[str] = None) -> int:
        """Count points in collection"""
        collection_name = collection_name or self.collection_name
        
        try:
            info = await self.get_collection_info(collection_name)
            if info:
                return info.points_count
            return 0
        except Exception as e:
            logger.error(f"Failed to count points in '{collection_name}': {e}")
            return 0
    
    async def scroll_points(
        self,
        limit: int = 100,
        offset: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Scroll through points in collection"""
        collection_name = collection_name or self.collection_name
        
        try:
            # Build query filter
            query_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    # Handle nested metadata fields
                    if key in ["channel_id", "team_id", "ingested_by", "source_type"]:
                        field_key = f"metadata.{key}"
                    else:
                        field_key = key
                    
                    conditions.append(
                        FieldCondition(
                            key=field_key,
                            match=MatchValue(value=value)
                        )
                    )
                query_filter = Filter(must=conditions)
            
            async with self.pool.get_client() as client:
                result = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=query_filter,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
            
            points = []
            for point in result[0]:  # result is tuple (points, next_page_offset)
                points.append({
                    "id": point.id,
                    "payload": point.payload
                })
            
            return points
            
        except Exception as e:
            logger.error(f"Failed to scroll points in '{collection_name}': {e}")
            return []
    
    async def scroll_all_points(self, collection_name: Optional[str] = None) -> List[Any]:
        """Scroll through all points in collection (for BM25 initialization)"""
        collection_name = collection_name or self.collection_name
        
        try:
            all_points = []
            offset = None
            batch_size = 1000
            
            async with self.pool.get_client() as client:
                while True:
                    result = client.scroll(
                        collection_name=collection_name,
                        limit=batch_size,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    points, next_offset = result
                    
                    if not points:
                        break
                    
                    all_points.extend(points)
                    
                    if not next_offset:
                        break
                        
                    offset = next_offset
            
            logger.info(f"Retrieved {len(all_points)} total points from '{collection_name}'")
            return all_points
            
        except Exception as e:
            logger.error(f"Failed to scroll all points in '{collection_name}': {e}")
            return []