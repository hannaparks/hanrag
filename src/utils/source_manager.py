"""
Enhanced source management for RAG system.
Handles source deduplication, ranking, validation, and citation mapping.
"""

import hashlib
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

@dataclass
class SourceInfo:
    """Rich source information structure"""
    source_id: str
    title: str
    url: Optional[str]
    source_type: str  # 'mattermost', 'web', 'document', 'file'
    timestamp: Optional[datetime]
    authority_score: float = 0.0
    relevance_score: float = 0.0
    chunk_count: int = 0
    best_snippet: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ProcessedSource:
    """Processed and deduplicated source with aggregated information"""
    source_info: SourceInfo
    chunks: List[Dict[str, Any]]
    combined_relevance: float
    snippet_preview: str
    citation_key: str

class SourceManager:
    """Advanced source management for RAG results"""
    
    def __init__(self):
        self.source_cache: Dict[str, SourceInfo] = {}
        
    def process_retrieval_results(
        self, 
        results: List[Any], 
        max_sources: int = 5,
        min_relevance_threshold: float = 0.3
    ) -> Tuple[List[ProcessedSource], Dict[str, str]]:
        """
        Process retrieval results into deduplicated, ranked sources.
        
        Returns:
            - List of processed sources
            - Citation mapping (citation_key -> source_id)
        """
        try:
            # Group results by source
            source_groups = self._group_by_source(results)
            
            # Adaptive threshold: if the threshold is too high for the actual data, lower it
            if source_groups:
                all_scores = []
                for chunks in source_groups.values():
                    for chunk in chunks:
                        all_scores.append(chunk["score"])
                
                if all_scores:
                    max_actual_score = max(all_scores)
                    # If the threshold is higher than the maximum score, use 20% of max score
                    if min_relevance_threshold > max_actual_score:
                        adaptive_threshold = max(0.01, max_actual_score * 0.2)
                        logger.info(f"Adapting relevance threshold from {min_relevance_threshold} to {adaptive_threshold:.4f} (max score: {max_actual_score:.4f})")
                        min_relevance_threshold = adaptive_threshold
            
            # Process each source group
            processed_sources = []
            citation_mapping = {}
            
            for source_id, chunks in source_groups.items():
                source_info = self._extract_source_info(source_id, chunks)
                
                # Debug logging for relevance scores
                logger.debug(f"Source {source_id}: relevance_score={source_info.relevance_score:.4f}, threshold={min_relevance_threshold}")
                
                # Filter by relevance threshold
                if source_info.relevance_score < min_relevance_threshold:
                    logger.debug(f"Filtered out source {source_id} (score {source_info.relevance_score:.4f} < threshold {min_relevance_threshold})")
                    continue
                    
                # Create processed source
                processed_source = self._create_processed_source(source_info, chunks)
                processed_sources.append(processed_source)
                
                # Create citation mapping
                citation_mapping[processed_source.citation_key] = source_id
            
            # If no sources pass the threshold, lower it and try again with top sources
            if not processed_sources and source_groups:
                logger.warning(f"No sources passed relevance threshold {min_relevance_threshold}, using top sources instead")
                for source_id, chunks in list(source_groups.items())[:max_sources]:
                    source_info = self._extract_source_info(source_id, chunks)
                    processed_source = self._create_processed_source(source_info, chunks)
                    processed_sources.append(processed_source)
                    citation_mapping[processed_source.citation_key] = source_id
            
            # Rank and limit sources
            ranked_sources = self._rank_sources(processed_sources)[:max_sources]
            
            # Update citation mapping for final sources only
            final_citation_mapping = {
                src.citation_key: citation_mapping[src.citation_key] 
                for src in ranked_sources
            }
            
            logger.info(f"Processed {len(source_groups)} source groups into {len(ranked_sources)} final sources")
            
            return ranked_sources, final_citation_mapping
            
        except Exception as e:
            logger.error(f"Source processing failed: {e}")
            return [], {}
    
    def _group_by_source(self, results: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Group retrieval results by source identifier"""
        source_groups = defaultdict(list)
        
        for result in results:
            try:
                # Extract source identifier
                source_id = self._normalize_source_id(result.source if hasattr(result, 'source') else str(result))
                
                # Create chunk data
                chunk_data = {
                    "content": result.content if hasattr(result, 'content') else str(result),
                    "score": getattr(result, 'score', 0.0),
                    "metadata": getattr(result, 'metadata', {})
                }
                
                source_groups[source_id].append(chunk_data)
                
            except Exception as e:
                logger.warning(f"Failed to process result: {e}")
                continue
                
        return dict(source_groups)
    
    def _normalize_source_id(self, raw_source: str) -> str:
        """Normalize source identifiers for consistent grouping"""
        if not raw_source:
            return "unknown_source"
            
        # Handle Mattermost sources
        if raw_source.startswith("mattermost://"):
            # Extract team and channel, normalize
            parts = raw_source.replace("mattermost://", "").split("/")
            if len(parts) >= 2:
                team, channel = parts[0], parts[1]
                return f"mattermost://{team}/{channel}"
            return raw_source
        
        # Handle web URLs
        if raw_source.startswith(("http://", "https://")):
            parsed = urlparse(raw_source)
            # Group by domain and path (without query params)
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        
        # Handle file sources
        if raw_source.startswith("file://"):
            return raw_source
            
        # Default normalization
        return raw_source.strip().lower()
    
    def _extract_source_info(self, source_id: str, chunks: List[Dict[str, Any]]) -> SourceInfo:
        """Extract comprehensive source information from chunks"""
        try:
            # Calculate aggregated scores
            total_score = sum(chunk["score"] for chunk in chunks)
            avg_score = total_score / len(chunks) if chunks else 0.0
            max_score = max(chunk["score"] for chunk in chunks) if chunks else 0.0
            
            # Find best snippet (highest scoring chunk)
            best_chunk = max(chunks, key=lambda x: x["score"]) if chunks else {}
            best_snippet = best_chunk.get("content", "")[:300]
            
            # Extract metadata from first chunk (should be consistent)
            first_metadata = chunks[0].get("metadata", {}) if chunks else {}
            
            # Determine source type and extract details
            source_type, title, url = self._parse_source_details(source_id, first_metadata)
            
            # Calculate authority score based on source type and metadata
            authority_score = self._calculate_authority_score(source_type, first_metadata)
            
            # Extract timestamp
            timestamp = self._extract_timestamp(first_metadata)
            
            return SourceInfo(
                source_id=source_id,
                title=title,
                url=url,
                source_type=source_type,
                timestamp=timestamp,
                authority_score=authority_score,
                relevance_score=max_score,  # Use max score as relevance
                chunk_count=len(chunks),
                best_snippet=best_snippet,
                metadata=first_metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to extract source info for {source_id}: {e}")
            return SourceInfo(
                source_id=source_id,
                title="Unknown Source",
                url=None,
                source_type="unknown",
                timestamp=None,
                relevance_score=0.0
            )
    
    def _parse_source_details(self, source_id: str, metadata: Dict[str, Any]) -> Tuple[str, str, Optional[str]]:
        """Parse source type, title, and URL from source ID and metadata"""
        
        # Mattermost sources
        if source_id.startswith("mattermost://"):
            parts = source_id.replace("mattermost://", "").split("/")
            team = parts[0] if len(parts) > 0 else "unknown"
            channel = parts[1] if len(parts) > 1 else "unknown"
            
            title = f"#{channel} channel ({team} team)"
            url = f"https://hanna-test.test.mattermost.cloud/{team}/channels/{channel}"
            return "mattermost", title, url
        
        # Web sources
        if source_id.startswith(("http://", "https://")):
            parsed = urlparse(source_id)
            title = metadata.get("title", parsed.netloc + parsed.path)
            return "web", title, source_id
        
        # File sources
        if source_id.startswith("file://"):
            file_path = source_id.replace("file://", "")
            title = metadata.get("title", file_path.split("/")[-1])
            return "document", title, None
        
        # Unknown sources
        title = metadata.get("title", source_id)
        return "unknown", title, None
    
    def _calculate_authority_score(self, source_type: str, metadata: Dict[str, Any]) -> float:
        """Calculate authority score based on source type and metadata"""
        base_scores = {
            "web": 0.7,
            "document": 0.8,
            "mattermost": 0.6,
            "unknown": 0.3
        }
        
        base_score = base_scores.get(source_type, 0.3)
        
        # Boost for official documentation
        if any(term in str(metadata).lower() for term in ["docs", "documentation", "api", "guide"]):
            base_score += 0.2
        
        # Boost for recent content (if timestamp available)
        timestamp = self._extract_timestamp(metadata)
        if timestamp:
            days_old = (datetime.now() - timestamp).days
            if days_old < 30:  # Content less than 30 days old
                base_score += 0.1
        
        return min(base_score, 1.0)
    
    def _extract_timestamp(self, metadata: Dict[str, Any]) -> Optional[datetime]:
        """Extract timestamp from metadata"""
        try:
            # Try different timestamp fields
            for field in ["timestamp", "create_at", "created_at", "date", "last_modified"]:
                if field in metadata:
                    value = metadata[field]
                    if isinstance(value, (int, float)):
                        # Assume Unix timestamp (milliseconds if > 1e10)
                        if value > 1e10:
                            value = value / 1000
                        return datetime.fromtimestamp(value)
                    elif isinstance(value, str):
                        # Try to parse ISO format
                        try:
                            return datetime.fromisoformat(value.replace('Z', '+00:00'))
                        except:
                            continue
            return None
        except Exception as e:
            logger.debug(f"Failed to extract timestamp: {e}")
            return None
    
    def _create_processed_source(self, source_info: SourceInfo, chunks: List[Dict[str, Any]]) -> ProcessedSource:
        """Create a processed source with combined information"""
        
        # Calculate combined relevance (weighted average favoring top chunks)
        if chunks:
            sorted_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)
            top_3_scores = [chunk["score"] for chunk in sorted_chunks[:3]]
            combined_relevance = sum(score * weight for score, weight in zip(top_3_scores, [0.5, 0.3, 0.2]))
        else:
            combined_relevance = 0.0
        
        # Create snippet preview from best content
        snippet_preview = source_info.best_snippet
        if len(snippet_preview) > 200:
            snippet_preview = snippet_preview[:200] + "..."
        
        # Generate citation key (short, readable identifier)
        citation_key = self._generate_citation_key(source_info)
        
        return ProcessedSource(
            source_info=source_info,
            chunks=chunks,
            combined_relevance=combined_relevance,
            snippet_preview=snippet_preview,
            citation_key=citation_key
        )
    
    def _generate_citation_key(self, source_info: SourceInfo) -> str:
        """Generate a short, readable citation key"""
        
        if source_info.source_type == "mattermost":
            # Extract channel name from source_id
            if "/" in source_info.source_id:
                channel = source_info.source_id.split("/")[-1]
                return f"#{channel}"
            return "mattermost"
        
        elif source_info.source_type == "web":
            # Use domain name
            if source_info.url:
                domain = urlparse(source_info.url).netloc
                return domain.replace("www.", "")
            return "web"
        
        elif source_info.source_type == "document":
            # Use filename
            if source_info.title:
                # Extract filename without extension
                name = source_info.title.split("/")[-1]
                name = name.split(".")[0] if "." in name else name
                return name[:20]  # Limit length
            return "doc"
        
        else:
            # Generate hash-based key for unknown sources
            hash_key = hashlib.md5(source_info.source_id.encode()).hexdigest()[:8]
            return f"src_{hash_key}"
    
    def _rank_sources(self, processed_sources: List[ProcessedSource]) -> List[ProcessedSource]:
        """Rank sources by combined score"""
        
        def source_score(source: ProcessedSource) -> float:
            # Combine multiple factors
            relevance_weight = 0.4
            authority_weight = 0.3
            freshness_weight = 0.2
            chunk_count_weight = 0.1
            
            relevance_score = source.combined_relevance
            authority_score = source.source_info.authority_score
            
            # Freshness score (higher for recent content)
            freshness_score = 0.5  # Default
            if source.source_info.timestamp:
                days_old = (datetime.now() - source.source_info.timestamp).days
                freshness_score = max(0.1, 1.0 - (days_old / 365))  # Decay over year
            
            # Chunk count score (more chunks = more comprehensive, but cap it)
            chunk_count_score = min(1.0, source.source_info.chunk_count / 5)
            
            total_score = (
                relevance_score * relevance_weight +
                authority_score * authority_weight +
                freshness_score * freshness_weight +
                chunk_count_score * chunk_count_weight
            )
            
            return total_score
        
        return sorted(processed_sources, key=source_score, reverse=True)
    
    def format_sources_for_display(self, processed_sources: List[ProcessedSource]) -> List[Dict[str, Any]]:
        """Format sources for API response"""
        
        formatted_sources = []
        
        for i, source in enumerate(processed_sources, 1):
            source_data = {
                "id": source.citation_key,
                "title": source.source_info.title,
                "type": source.source_info.source_type,
                "url": source.source_info.url,
                "snippet": source.snippet_preview,
                "relevance_score": round(source.combined_relevance, 3),
                "authority_score": round(source.source_info.authority_score, 3),
                "chunk_count": source.source_info.chunk_count,
                "timestamp": source.source_info.timestamp.isoformat() if source.source_info.timestamp else None,
                "citation_key": source.citation_key,
                "rank": i
            }
            
            # Add metadata for debugging (optional)
            if logger.level <= logging.DEBUG:
                source_data["debug_metadata"] = source.source_info.metadata
            
            formatted_sources.append(source_data)
        
        return formatted_sources
    
    def generate_source_context_for_llm(
        self, 
        processed_sources: List[ProcessedSource],
        citation_mapping: Dict[str, str]
    ) -> Tuple[str, str]:
        """
        Generate context and citation mapping for LLM.
        
        Returns:
            - Context text with proper source attribution
            - Citation instructions for the LLM
        """
        
        context_parts = []
        citation_instructions = []
        
        for source in processed_sources:
            # Add source header
            context_parts.append(f"\n--- Source: {source.citation_key} ({source.source_info.title}) ---")
            
            # Add chunks from this source
            for chunk in source.chunks:
                context_parts.append(chunk["content"])
            
            # Build citation instruction
            citation_instructions.append(
                f"- Use '{source.citation_key}' to cite: {source.source_info.title}"
            )
        
        context_text = "\n".join(context_parts)
        
        citation_guide = (
            "IMPORTANT: When citing sources in your response, use these exact citation keys:\n" +
            "\n".join(citation_instructions) +
            "\n\nFormat citations like: 'According to [citation_key], ...' or 'As mentioned in [citation_key], ...'"
        )
        
        return context_text, citation_guide

    def validate_citations_in_response(
        self, 
        response_text: str, 
        processed_sources: List[ProcessedSource]
    ) -> Dict[str, Any]:
        """Validate that citations in response are accurate"""
        
        valid_citation_keys = {source.citation_key for source in processed_sources}
        
        # Extract citations from response
        citation_pattern = r'\[([^\]]+)\]'
        found_citations = re.findall(citation_pattern, response_text)
        
        valid_citations = []
        invalid_citations = []
        
        for citation in found_citations:
            if citation in valid_citation_keys:
                valid_citations.append(citation)
            else:
                invalid_citations.append(citation)
        
        return {
            "valid_citations": valid_citations,
            "invalid_citations": invalid_citations,
            "citation_accuracy": len(valid_citations) / len(found_citations) if found_citations else 1.0,
            "total_citations": len(found_citations)
        }