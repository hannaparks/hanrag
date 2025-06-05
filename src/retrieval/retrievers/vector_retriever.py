import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger

from ...storage.qdrant_client import QdrantManager
from ...config.settings import settings


@dataclass
class RetrievalResult:
    """Represents a retrieval result with content and metadata"""
    content: str
    score: float
    metadata: Dict[str, Any]
    chunk_id: str
    source: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'score': self.score,
            'metadata': self.metadata,
            'chunk_id': self.chunk_id,
            'source': self.source
        }


class VectorRetriever:
    """Advanced vector retrieval with embeddings and semantic search"""
    
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.qdrant_manager = QdrantManager()
        self.embedding_model = settings.EMBEDDING_MODEL
        self.embedding_dimension = settings.EMBEDDING_DIMENSION
        
        # Ensure collection exists
        asyncio.create_task(self._ensure_collection())
    
    async def _ensure_collection(self):
        """Ensure the vector collection exists"""
        await self.qdrant_manager.create_collection()
    
    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from OpenAI with batch processing"""
        
        if not texts:
            return []
        
        # Estimate tokens and determine batch size
        total_estimated_tokens = sum(len(text.split()) * 1.3 for text in texts)  # Rough estimate
        max_tokens_per_batch = 250000  # Leave some headroom under 300k limit
        
        # If small enough, process in single batch
        if total_estimated_tokens <= max_tokens_per_batch:
            return await self._get_embeddings_batch(texts)
        
        # Otherwise, split into smaller batches
        logger.info(f"Large embedding request: {len(texts)} texts, ~{total_estimated_tokens:.0f} tokens. Splitting into batches.")
        
        # Estimate how many batches we need
        estimated_batches = max(1, int(total_estimated_tokens / max_tokens_per_batch) + 1)
        batch_size = max(1, len(texts) // estimated_batches)
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await self._get_embeddings_batch(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Log progress for large operations
            if len(texts) > 100:
                logger.info(f"Embedded batch {len(all_embeddings)}/{len(texts)} texts")
        
        logger.info(f"Completed embedding generation: {len(all_embeddings)} embeddings")
        return all_embeddings

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIError))
    )
    async def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a single batch"""
        
        try:
            # Estimate tokens for this batch
            estimated_tokens = sum(len(text.split()) * 1.3 for text in texts)
            logger.debug(f"Requesting embeddings for {len(texts)} texts (~{estimated_tokens:.0f} tokens)")
            
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            
            embeddings = [data.embedding for data in response.data]
            logger.debug(f"Generated {len(embeddings)} embeddings with {len(embeddings[0])} dimensions")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to get embeddings batch: {e}")
            raise
    
    async def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a single query"""
        embeddings = await self._get_embeddings([query])
        return embeddings[0]
    
    async def similarity_search(
        self,
        query: str,
        top_k: int = 50,
        collection_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Perform similarity search with optional filtering"""
        
        try:
            # Get query embedding
            query_embedding = await self.get_query_embedding(query)
            
            # Search vectors with optional filtering
            search_results = await self.qdrant_manager.search_vectors(
                query_vector=query_embedding,
                top_k=top_k,
                collection_name=collection_name,
                filters=filters
            )
            
            # Convert to RetrievalResult objects
            results = []
            for result in search_results:
                retrieval_result = RetrievalResult(
                    content=result.payload.get('content', ''),
                    score=result.score,
                    metadata=result.payload.get('metadata', {}),
                    chunk_id=str(result.id),
                    source=result.payload.get('source', 'unknown')
                )
                results.append(retrieval_result)
            
            logger.info(f"Retrieved {len(results)} similar documents for query" +
                       (f" with filters: {filters}" if filters else ""))
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 50,
        collection_name: Optional[str] = None
    ) -> List[RetrievalResult]:
        """Hybrid search combining vector similarity across all channels"""
        
        try:
            # Get query embedding
            query_embedding = await self.get_query_embedding(query)
            
            # Perform hybrid search (no channel filtering)
            search_results = await self.qdrant_manager.hybrid_search(
                query_vector=query_embedding,
                query_text=query,
                top_k=top_k,
                collection_name=collection_name
            )
            
            # Convert to RetrievalResult objects
            results = []
            for result in search_results:
                retrieval_result = RetrievalResult(
                    content=result.payload.get('content', ''),
                    score=result.score,
                    metadata=result.payload.get('metadata', {}),
                    chunk_id=str(result.id),
                    source=result.payload.get('source', 'unknown')
                )
                results.append(retrieval_result)
            
            logger.info(f"Hybrid search retrieved {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    async def mmr_search(
        self,
        query: str,
        top_k: int = 50,
        mmr_lambda: float = 0.7
    ) -> List[RetrievalResult]:
        """Maximal Marginal Relevance search for diverse results"""
        
        try:
            # Get more candidates than needed
            candidate_k = min(top_k * 3, 150)
            
            # Initial similarity search (no filtering)
            candidates = await self.similarity_search(
                query=query,
                top_k=candidate_k
            )
            
            if not candidates:
                return []
            
            # MMR selection
            selected = []
            remaining = candidates.copy()
            
            # Always select the most relevant first
            if remaining:
                best = max(remaining, key=lambda x: x.score)
                selected.append(best)
                remaining.remove(best)
            
            # Get embeddings for selected and remaining documents
            query_embedding = await self.get_query_embedding(query)
            
            # Select remaining documents using MMR
            while len(selected) < top_k and remaining:
                mmr_scores = []
                
                for candidate in remaining:
                    # Relevance score (already computed)
                    relevance = candidate.score
                    
                    # Diversity score (similarity to already selected)
                    if selected:
                        # Simple diversity based on content similarity
                        max_similarity = 0
                        for selected_doc in selected:
                            # Use simple text overlap as diversity metric
                            overlap = self._calculate_text_overlap(candidate.content, selected_doc.content)
                            max_similarity = max(max_similarity, overlap)
                        
                        diversity = 1 - max_similarity
                    else:
                        diversity = 1.0
                    
                    # MMR score
                    mmr_score = mmr_lambda * relevance + (1 - mmr_lambda) * diversity
                    mmr_scores.append((candidate, mmr_score))
                
                # Select candidate with highest MMR score
                if mmr_scores:
                    best_candidate, _ = max(mmr_scores, key=lambda x: x[1])
                    selected.append(best_candidate)
                    remaining.remove(best_candidate)
                else:
                    break
            
            logger.info(f"MMR search selected {len(selected)} diverse documents")
            return selected
            
        except Exception as e:
            logger.error(f"MMR search failed: {e}")
            # Fallback to regular similarity search
            return await self.similarity_search(query, top_k)
    
    def _calculate_text_overlap(self, text1: str, text2: str) -> float:
        """Calculate simple text overlap between two texts"""
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def get_similar_documents(
        self,
        document_id: str,
        top_k: int = 10,
        collection_name: Optional[str] = None
    ) -> List[RetrievalResult]:
        """Find documents similar to a given document"""
        
        try:
            # First, get the document content
            points = await self.qdrant_manager.scroll_points(
                filters={'document_id': document_id},
                limit=1,
                collection_name=collection_name
            )
            
            if not points:
                logger.warning(f"Document {document_id} not found")
                return []
            
            document_content = points[0]['payload'].get('content', '')
            
            if not document_content:
                logger.warning(f"No content found for document {document_id}")
                return []
            
            # Search for similar documents
            results = await self.similarity_search(
                query=document_content,
                top_k=top_k + 1,  # +1 to exclude the original document
                collection_name=collection_name
            )
            
            # Filter out the original document
            filtered_results = [r for r in results if r.metadata.get('document_id') != document_id]
            
            return filtered_results[:top_k]
            
        except Exception as e:
            logger.error(f"Similar document search failed: {e}")
            return []
    
    async def search_by_metadata(
        self,
        metadata_filters: Dict[str, Any],
        limit: int = 100,
        collection_name: Optional[str] = None
    ) -> List[RetrievalResult]:
        """Search documents by metadata only"""
        
        try:
            points = await self.qdrant_manager.scroll_points(
                filters=metadata_filters,
                limit=limit,
                collection_name=collection_name
            )
            
            results = []
            for point in points:
                retrieval_result = RetrievalResult(
                    content=point['payload'].get('content', ''),
                    score=1.0,  # No similarity score for metadata-only search
                    metadata=point['payload'].get('metadata', {}),
                    chunk_id=str(point['id']),
                    source=point['payload'].get('source', 'unknown')
                )
                results.append(retrieval_result)
            
            logger.info(f"Found {len(results)} documents matching metadata filters")
            return results
            
        except Exception as e:
            logger.error(f"Metadata search failed: {e}")
            return []
    
    async def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about the collection"""
        
        try:
            info = await self.qdrant_manager.get_collection_info(collection_name)
            point_count = await self.qdrant_manager.count_points(collection_name)
            
            return {
                'total_points': point_count,
                'collection_info': info.__dict__ if info else {},
                'embedding_dimension': self.embedding_dimension,
                'embedding_model': self.embedding_model
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}