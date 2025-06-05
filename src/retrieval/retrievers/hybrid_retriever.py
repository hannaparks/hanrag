import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from .vector_retriever import VectorRetriever, RetrievalResult
from .bm25_retriever import BM25Retriever


class FusionMethod(Enum):
    """Methods for combining vector and BM25 search results"""
    RRF = "reciprocal_rank_fusion"  # Reciprocal Rank Fusion
    LINEAR = "linear_combination"    # Weighted linear combination
    ADAPTIVE = "adaptive"           # Query-type adaptive fusion


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search parameters"""
    vector_weight: float = 0.7      # Weight for vector search (0.0 to 1.0)
    bm25_weight: float = 0.3        # Weight for BM25 search (0.0 to 1.0)
    fusion_method: FusionMethod = FusionMethod.RRF
    rrf_k: int = 60                 # RRF parameter (typically 60)
    min_score_threshold: float = 0.0  # Minimum score to include result
    normalize_scores: bool = True    # Whether to normalize scores before fusion
    
    def __post_init__(self):
        """Validate configuration"""
        if abs(self.vector_weight + self.bm25_weight - 1.0) > 0.001:
            logger.warning(f"Vector and BM25 weights don't sum to 1.0: "
                          f"{self.vector_weight} + {self.bm25_weight} = "
                          f"{self.vector_weight + self.bm25_weight}")


class HybridRetriever:
    """Hybrid retriever combining vector similarity and BM25 lexical search"""
    
    def __init__(self, config: Optional[HybridSearchConfig] = None):
        """
        Initialize hybrid retriever
        
        Args:
            config: Hybrid search configuration
        """
        self.config = config or HybridSearchConfig()
        self.vector_retriever = VectorRetriever()
        self.bm25_retriever = BM25Retriever()
        
    async def search(
        self,
        query: str,
        top_k: int = 50,
        vector_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
        collection_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Perform hybrid search combining vector and BM25 results with optional filtering
        
        Args:
            query: Search query
            top_k: Final number of results to return
            vector_top_k: Number of vector results to retrieve (default: top_k * 2)
            bm25_top_k: Number of BM25 results to retrieve (default: top_k * 2)
            collection_name: Optional collection name override
            filters: Optional filters to apply (e.g., {"channel_id": "123", "date_range": {"start": "2024-01-01"}})
        """
        
        try:
            # Default to retrieving more candidates for better fusion
            vector_k = vector_top_k or min(top_k * 2, 100)
            bm25_k = bm25_top_k or min(top_k * 2, 100)
            
            logger.info(f"Hybrid search: vector_k={vector_k}, bm25_k={bm25_k}, "
                       f"fusion={self.config.fusion_method.value}" +
                       (f", filters={filters}" if filters else ""))
            
            # Perform searches in parallel
            vector_task = self.vector_retriever.similarity_search(
                query=query,
                top_k=vector_k,
                collection_name=collection_name,
                filters=filters
            )
            
            # Note: BM25 doesn't support filtering directly, but we'll filter results after
            bm25_task = self.bm25_retriever.search(
                query=query,
                top_k=bm25_k,
                collection_name=collection_name
            )
            
            vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)
            
            logger.debug(f"Retrieved {len(vector_results)} vector results, "
                        f"{len(bm25_results)} BM25 results")
            
            # Filter BM25 results if filters are provided
            if filters and bm25_results:
                bm25_results = self._filter_results(bm25_results, filters)
                logger.debug(f"Filtered BM25 results to {len(bm25_results)} matching filter criteria")
            
            # Fuse results
            fused_results = await self._fuse_results(
                vector_results=vector_results,
                bm25_results=bm25_results,
                query=query
            )
            
            # Apply final filtering and ranking
            final_results = self._post_process_results(fused_results, top_k)
            
            logger.info(f"Hybrid search returned {len(final_results)} final results")
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to vector search only
            logger.info("Falling back to vector search only")
            return await self.vector_retriever.similarity_search(query, top_k, collection_name, filters)
    
    async def _fuse_results(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        query: str
    ) -> List[RetrievalResult]:
        """Fuse vector and BM25 results using the configured method"""
        
        if self.config.fusion_method == FusionMethod.RRF:
            return await self._reciprocal_rank_fusion(vector_results, bm25_results)
        elif self.config.fusion_method == FusionMethod.LINEAR:
            return await self._linear_combination_fusion(vector_results, bm25_results)
        elif self.config.fusion_method == FusionMethod.ADAPTIVE:
            return await self._adaptive_fusion(vector_results, bm25_results, query)
        else:
            logger.warning(f"Unknown fusion method: {self.config.fusion_method}")
            return await self._reciprocal_rank_fusion(vector_results, bm25_results)
    
    async def _reciprocal_rank_fusion(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion (RRF) - combines results based on their ranks
        RRF score = Σ(1 / (k + rank)) for each result list
        """
        
        k = self.config.rrf_k
        fused_scores: Dict[str, float] = {}
        all_results: Dict[str, RetrievalResult] = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results, 1):
            rrf_score = 1.0 / (k + rank)
            fused_scores[result.chunk_id] = fused_scores.get(result.chunk_id, 0) + rrf_score
            all_results[result.chunk_id] = result
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results, 1):
            rrf_score = 1.0 / (k + rank)
            fused_scores[result.chunk_id] = fused_scores.get(result.chunk_id, 0) + rrf_score
            all_results[result.chunk_id] = result
        
        # Create fused results
        fused_results = []
        for chunk_id, fused_score in fused_scores.items():
            result = all_results[chunk_id]
            # Create new result with fused score
            fused_result = RetrievalResult(
                content=result.content,
                score=fused_score,
                metadata={**result.metadata, 'fusion_method': 'RRF'},
                chunk_id=result.chunk_id,
                source=result.source
            )
            fused_results.append(fused_result)
        
        # Sort by fused score
        fused_results.sort(key=lambda x: x.score, reverse=True)
        
        logger.debug(f"RRF fusion: {len(fused_results)} unique results")
        return fused_results
    
    async def _linear_combination_fusion(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Linear combination fusion - weighted average of normalized scores
        """
        
        # Normalize scores if enabled
        if self.config.normalize_scores:
            vector_results = self._normalize_scores(vector_results)
            bm25_results = self._normalize_scores(bm25_results)
        
        fused_scores: Dict[str, float] = {}
        all_results: Dict[str, RetrievalResult] = {}
        
        # Process vector results
        for result in vector_results:
            weighted_score = result.score * self.config.vector_weight
            fused_scores[result.chunk_id] = fused_scores.get(result.chunk_id, 0) + weighted_score
            all_results[result.chunk_id] = result
        
        # Process BM25 results
        for result in bm25_results:
            weighted_score = result.score * self.config.bm25_weight
            fused_scores[result.chunk_id] = fused_scores.get(result.chunk_id, 0) + weighted_score
            all_results[result.chunk_id] = result
        
        # Create fused results
        fused_results = []
        for chunk_id, fused_score in fused_scores.items():
            result = all_results[chunk_id]
            fused_result = RetrievalResult(
                content=result.content,
                score=fused_score,
                metadata={**result.metadata, 'fusion_method': 'LINEAR'},
                chunk_id=result.chunk_id,
                source=result.source
            )
            fused_results.append(fused_result)
        
        # Sort by fused score
        fused_results.sort(key=lambda x: x.score, reverse=True)
        
        logger.debug(f"Linear fusion: {len(fused_results)} unique results")
        return fused_results
    
    async def _adaptive_fusion(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        query: str
    ) -> List[RetrievalResult]:
        """
        Adaptive fusion - adjust weights based on query characteristics
        """
        
        # Analyze query to determine optimal weights
        adaptive_weights = self._analyze_query_for_weights(query)
        
        # Temporarily override config weights
        original_vector_weight = self.config.vector_weight
        original_bm25_weight = self.config.bm25_weight
        
        self.config.vector_weight = adaptive_weights['vector']
        self.config.bm25_weight = adaptive_weights['bm25']
        
        logger.debug(f"Adaptive weights: vector={adaptive_weights['vector']:.2f}, "
                    f"bm25={adaptive_weights['bm25']:.2f}")
        
        # Use linear combination with adaptive weights
        results = await self._linear_combination_fusion(vector_results, bm25_results)
        
        # Restore original weights
        self.config.vector_weight = original_vector_weight
        self.config.bm25_weight = original_bm25_weight
        
        # Add adaptive fusion metadata
        for result in results:
            result.metadata['fusion_method'] = 'ADAPTIVE'
            result.metadata['adaptive_weights'] = adaptive_weights
        
        return results
    
    def _analyze_query_for_weights(self, query: str) -> Dict[str, float]:
        """
        Analyze query characteristics to determine optimal fusion weights
        """
        
        # Default weights
        vector_weight = 0.7
        bm25_weight = 0.3
        
        query_lower = query.lower()
        
        # Keyword-heavy queries favor BM25
        if any(keyword in query_lower for keyword in ['name', 'title', 'called', 'specific']):
            vector_weight = 0.5
            bm25_weight = 0.5
        
        # Conceptual/semantic queries favor vector search
        elif any(keyword in query_lower for keyword in ['similar', 'related', 'concept', 'meaning', 'like']):
            vector_weight = 0.8
            bm25_weight = 0.2
        
        # Technical/exact match queries favor BM25
        elif any(keyword in query_lower for keyword in ['error', 'code', 'function', 'method', 'api']):
            vector_weight = 0.4
            bm25_weight = 0.6
        
        # Long, complex queries favor vector search
        elif len(query.split()) > 10:
            vector_weight = 0.8
            bm25_weight = 0.2
        
        # Short, specific queries favor BM25
        elif len(query.split()) <= 3:
            vector_weight = 0.4
            bm25_weight = 0.6
        
        return {'vector': vector_weight, 'bm25': bm25_weight}
    
    def _normalize_scores(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Normalize scores to 0-1 range using min-max normalization"""
        
        if not results:
            return results
        
        scores = [result.score for result in results]
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        if max_score == min_score:
            for result in results:
                result.score = 1.0
            return results
        
        # Normalize scores
        for result in results:
            result.score = (result.score - min_score) / (max_score - min_score)
        
        return results
    
    def _post_process_results(
        self,
        results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """Apply post-processing filters and limits"""
        
        # Filter by minimum score threshold
        if self.config.min_score_threshold > 0:
            results = [r for r in results if r.score >= self.config.min_score_threshold]
        
        # Remove duplicates (by chunk_id, keeping highest score)
        seen_chunks = set()
        deduplicated = []
        
        for result in results:
            if result.chunk_id not in seen_chunks:
                seen_chunks.add(result.chunk_id)
                deduplicated.append(result)
        
        # Return top_k results
        return deduplicated[:top_k]
    
    async def search_with_mmr(
        self,
        query: str,
        top_k: int = 50,
        mmr_lambda: float = 0.7,
        collection_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Hybrid search with MMR (Maximal Marginal Relevance) diversification
        """
        
        try:
            # Get more candidates for MMR
            candidate_k = min(top_k * 3, 150)
            
            # Initial hybrid search
            candidates = await self.search(
                query=query,
                top_k=candidate_k,
                collection_name=collection_name,
                filters=filters
            )
            
            if not candidates:
                return []
            
            # Apply MMR using vector retriever's implementation
            # Note: This will re-embed the query, but provides better diversity
            selected = []
            remaining = candidates.copy()
            
            # Always select the most relevant first
            if remaining:
                best = max(remaining, key=lambda x: x.score)
                selected.append(best)
                remaining.remove(best)
            
            # Get query embedding for diversity calculation
            query_embedding = await self.vector_retriever.get_query_embedding(query)
            
            # MMR selection process
            while len(selected) < top_k and remaining:
                mmr_scores = []
                
                for candidate in remaining:
                    # Relevance score (from hybrid search)
                    relevance = candidate.score
                    
                    # Calculate diversity (1 - max similarity to selected docs)
                    if selected:
                        max_similarity = 0
                        for selected_doc in selected:
                            # Use text overlap as simple similarity metric
                            overlap = self._calculate_text_overlap(
                                candidate.content, 
                                selected_doc.content
                            )
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
            
            logger.info(f"Hybrid MMR search selected {len(selected)} diverse results")
            return selected
            
        except Exception as e:
            logger.error(f"Hybrid MMR search failed: {e}")
            # Fallback to regular hybrid search
            return await self.search(query, top_k, collection_name, filters)
    
    def _calculate_text_overlap(self, text1: str, text2: str) -> float:
        """Calculate simple text overlap between two texts"""
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _filter_results(self, results: List[RetrievalResult], filters: Dict[str, Any]) -> List[RetrievalResult]:
        """Filter results based on metadata criteria"""
        filtered = []
        
        for result in results:
            matches = True
            metadata = result.metadata
            
            for key, value in filters.items():
                # Handle date range filtering
                if key == "date_range" and isinstance(value, dict):
                    timestamp = metadata.get("timestamp")
                    if timestamp:
                        if "start" in value and timestamp < value["start"]:
                            matches = False
                            break
                        if "end" in value and timestamp > value["end"]:
                            matches = False
                            break
                # Handle regular field matching
                elif metadata.get(key) != value:
                    matches = False
                    break
            
            if matches:
                filtered.append(result)
        
        return filtered
    
    async def explain_fusion(
        self,
        query: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Explain how fusion works for a specific query (for debugging)
        """
        
        try:
            # Get separate results
            vector_results = await self.vector_retriever.similarity_search(query, top_k * 2)
            bm25_results = await self.bm25_retriever.search(query, top_k * 2)
            
            # Get fused results
            fused_results = await self._fuse_results(vector_results, bm25_results, query)
            
            explanation = {
                'query': query,
                'fusion_method': self.config.fusion_method.value,
                'config': {
                    'vector_weight': self.config.vector_weight,
                    'bm25_weight': self.config.bm25_weight,
                    'rrf_k': self.config.rrf_k
                },
                'vector_results_count': len(vector_results),
                'bm25_results_count': len(bm25_results),
                'fused_results_count': len(fused_results),
                'top_fused_results': []
            }
            
            # Add details for top results
            for i, result in enumerate(fused_results[:top_k]):
                result_detail = {
                    'rank': i + 1,
                    'chunk_id': result.chunk_id,
                    'fused_score': result.score,
                    'content_preview': result.content[:200] + "..." if len(result.content) > 200 else result.content,
                    'source': result.source,
                    'in_vector_results': any(r.chunk_id == result.chunk_id for r in vector_results),
                    'in_bm25_results': any(r.chunk_id == result.chunk_id for r in bm25_results)
                }
                
                # Add original scores if available
                vector_match = next((r for r in vector_results if r.chunk_id == result.chunk_id), None)
                bm25_match = next((r for r in bm25_results if r.chunk_id == result.chunk_id), None)
                
                if vector_match:
                    result_detail['original_vector_score'] = vector_match.score
                if bm25_match:
                    result_detail['original_bm25_score'] = bm25_match.score
                
                explanation['top_fused_results'].append(result_detail)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to explain fusion: {e}")
            return {'error': str(e)}
    
    async def get_retriever_stats(self) -> Dict[str, Any]:
        """Get statistics about both retrievers"""
        
        try:
            vector_stats = await self.vector_retriever.get_collection_stats()
            bm25_stats = await self.bm25_retriever.get_index_stats()
            
            return {
                'hybrid_config': {
                    'vector_weight': self.config.vector_weight,
                    'bm25_weight': self.config.bm25_weight,
                    'fusion_method': self.config.fusion_method.value,
                    'rrf_k': self.config.rrf_k
                },
                'vector_retriever': vector_stats,
                'bm25_retriever': bm25_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get retriever stats: {e}")
            return {'error': str(e)}