"""
Metadata Integration for RAG Pipeline

This module provides seamless integration between the RAG pipeline and the
metadata management system, ensuring comprehensive tracking and attribution
throughout the entire document processing lifecycle.

Key Integration Points:
- Automatic metadata creation during ingestion
- Processing stage tracking throughout pipeline
- Quality assessment integration
- Chunk metadata linkage with vector storage
- Search result enrichment with metadata
- Performance analytics and monitoring
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import logging
logger = logging.getLogger(__name__)

# Import metadata store components (handle relative imports)
try:
    from ..storage.metadata_store import (
        metadata_store,
        SourceType, 
        ProcessingStage,
        DocumentMetadata,
        ChunkMetadata
    )
except ImportError:
    # For standalone testing
    from storage.metadata_store import (
        metadata_store,
        SourceType, 
        ProcessingStage,
        DocumentMetadata,
        ChunkMetadata
    )


class MetadataIntegration:
    """Handles metadata integration throughout the RAG pipeline"""
    
    def __init__(self):
        self.metadata_store = metadata_store
        self._processing_context = {}  # Track current processing context
    
    async def start_document_processing(
        self,
        source_path: str,
        source_type: SourceType,
        content: str,
        **kwargs
    ) -> DocumentMetadata:
        """Start document processing and create initial metadata"""
        
        logger.info(f"Starting metadata tracking for {source_path}")
        
        # Create comprehensive document metadata
        document_metadata = await self.metadata_store.create_document_metadata(
            source_path=source_path,
            source_type=source_type,
            content=content,
            **kwargs
        )
        
        # Store processing context
        self._processing_context[document_metadata.document_id] = {
            "start_time": datetime.now(timezone.utc),
            "current_stage": ProcessingStage.RAW_INGESTION,
            "chunk_count": 0,
            "errors": [],
            "warnings": []
        }
        
        logger.debug(f"Created metadata for document {document_metadata.document_id}")
        return document_metadata
    
    async def track_processing_stage(
        self,
        document_id: str,
        stage: ProcessingStage,
        status: str = "success",
        details: Optional[Dict[str, Any]] = None,
        processing_time: Optional[float] = None
    ) -> None:
        """Track a processing stage in the audit trail"""
        
        if document_id not in self._processing_context:
            logger.warning(f"No processing context found for document {document_id}")
            return
        
        context = self._processing_context[document_id]
        
        # Calculate processing time if not provided
        if processing_time is None:
            stage_start = context.get("stage_start_time", datetime.now(timezone.utc))
            processing_time = (datetime.now(timezone.utc) - stage_start).total_seconds()
        
        # Add processing stage to metadata
        await self.metadata_store.add_processing_stage(
            document_id=document_id,
            stage=stage,
            status=status,
            details=details or {},
            processing_time=processing_time
        )
        
        # Update processing context
        context["current_stage"] = stage
        context["stage_start_time"] = datetime.now(timezone.utc)
        
        if status == "error":
            context["errors"].append(f"{stage.value}: {details.get('error', 'Unknown error')}")
        elif status == "warning":
            context["warnings"].append(f"{stage.value}: {details.get('warning', 'Unknown warning')}")
        
        logger.debug(f"Tracked {stage.value} for document {document_id} with status {status}")
    
    async def add_chunk_tracking(
        self,
        document_id: str,
        chunk_id: str,
        content: str,
        chunk_index: int,
        total_chunks: int,
        vector_id: Optional[str] = None,
        **kwargs
    ) -> ChunkMetadata:
        """Add metadata tracking for a content chunk"""
        
        if document_id not in self._processing_context:
            logger.warning(f"No processing context found for document {document_id}")
        
        # Create chunk metadata
        chunk_metadata = await self.metadata_store.add_chunk_metadata(
            document_id=document_id,
            chunk_id=chunk_id,
            content=content,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            embedding_vector_id=vector_id,
            **kwargs
        )
        
        # Update processing context
        if document_id in self._processing_context:
            self._processing_context[document_id]["chunk_count"] += 1
        
        logger.debug(f"Added chunk metadata {chunk_id} for document {document_id}")
        return chunk_metadata
    
    async def complete_document_processing(
        self,
        document_id: str,
        final_status: str = "success",
        final_details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Complete document processing and finalize metadata"""
        
        if document_id not in self._processing_context:
            logger.warning(f"No processing context found for document {document_id}")
            return {"success": False, "error": "No processing context"}
        
        context = self._processing_context[document_id]
        
        # Calculate total processing time
        total_time = (datetime.now(timezone.utc) - context["start_time"]).total_seconds()
        
        # Add final processing stage
        await self.track_processing_stage(
            document_id=document_id,
            stage=ProcessingStage.INDEXING_COMPLETE,
            status=final_status,
            details={
                "total_processing_time": total_time,
                "total_chunks": context["chunk_count"],
                "error_count": len(context["errors"]),
                "warning_count": len(context["warnings"]),
                **(final_details or {})
            },
            processing_time=total_time
        )
        
        # Clean up processing context
        processing_summary = {
            "document_id": document_id,
            "total_time": total_time,
            "chunk_count": context["chunk_count"],
            "errors": context["errors"],
            "warnings": context["warnings"],
            "final_status": final_status
        }
        
        del self._processing_context[document_id]
        
        logger.info(f"Completed processing for document {document_id} in {total_time:.2f}s")
        return processing_summary
    
    async def enrich_search_results(
        self,
        search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enrich search results with comprehensive metadata"""
        
        enriched_results = []
        
        for result in search_results:
            # Extract document/chunk information from result
            chunk_id = result.get("chunk_id") or result.get("id")
            source_path = result.get("source") or result.get("metadata", {}).get("source")
            
            # Find associated document metadata
            document_metadata = None
            chunk_metadata = None
            
            # Search for document by source path
            if source_path:
                documents = await self.metadata_store.search_by_source(source_path)
                if documents:
                    document_metadata = documents[0]
                    
                    # Find specific chunk metadata
                    if chunk_id:
                        for chunk in document_metadata.chunk_metadata:
                            if chunk.chunk_id == chunk_id:
                                chunk_metadata = chunk
                                break
            
            # Create enriched result
            enriched_result = result.copy()
            
            if document_metadata:
                enriched_result["metadata_enhanced"] = {
                    "document_id": document_metadata.document_id,
                    "source_attribution": {
                        "source_type": document_metadata.source_attribution.source_type.value,
                        "extraction_timestamp": document_metadata.source_attribution.extraction_timestamp.isoformat(),
                        "channel_id": document_metadata.source_attribution.channel_id,
                        "team_id": document_metadata.source_attribution.team_id
                    },
                    "content_classification": {
                        "category": document_metadata.content_classification.category.value,
                        "confidence": document_metadata.content_classification.confidence,
                        "keywords": document_metadata.content_classification.keywords[:5],
                        "technical_level": document_metadata.content_classification.technical_level
                    },
                    "quality_metrics": {
                        "overall_score": document_metadata.quality_metrics.overall_score,
                        "relevance_score": document_metadata.quality_metrics.relevance_score,
                        "authority_score": document_metadata.quality_metrics.authority_score,
                        "freshness_score": document_metadata.quality_metrics.freshness_score,
                        "uniqueness_score": document_metadata.quality_metrics.uniqueness_score
                    },
                    "document_stats": {
                        "total_chunks": len(document_metadata.chunk_metadata),
                        "created_at": document_metadata.created_at.isoformat(),
                        "last_updated": document_metadata.last_updated.isoformat(),
                        "access_count": document_metadata.access_count,
                        "version": document_metadata.version
                    }
                }
                
                if chunk_metadata:
                    enriched_result["metadata_enhanced"]["chunk_info"] = {
                        "chunk_index": chunk_metadata.chunk_index,
                        "total_chunks": chunk_metadata.total_chunks,
                        "chunk_length": chunk_metadata.content_length,
                        "token_count": chunk_metadata.token_count,
                        "chunking_strategy": chunk_metadata.chunking_strategy
                    }
                
                # Update access tracking
                document_metadata.access_count += 1
                document_metadata.last_accessed = datetime.now(timezone.utc)
            
            enriched_results.append(enriched_result)
        
        return enriched_results
    
    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        
        # Get basic metadata statistics
        base_stats = await self.metadata_store.get_statistics()
        
        # Add processing-specific statistics
        active_processing = len(self._processing_context)
        
        # Calculate average processing times by stage
        stage_times = {}
        error_rates = {}
        
        for document_metadata in self.metadata_store.documents.values():
            for history in document_metadata.processing_history:
                stage = history.stage.value
                
                if stage not in stage_times:
                    stage_times[stage] = []
                    error_rates[stage] = {"total": 0, "errors": 0}
                
                stage_times[stage].append(history.processing_time)
                error_rates[stage]["total"] += 1
                
                if history.status == "error":
                    error_rates[stage]["errors"] += 1
        
        # Calculate averages
        avg_stage_times = {}
        stage_error_rates = {}
        
        for stage, times in stage_times.items():
            avg_stage_times[stage] = sum(times) / len(times) if times else 0
            
            rates = error_rates[stage]
            stage_error_rates[stage] = rates["errors"] / rates["total"] if rates["total"] > 0 else 0
        
        return {
            **base_stats,
            "processing_stats": {
                "active_documents": active_processing,
                "average_stage_times": avg_stage_times,
                "stage_error_rates": stage_error_rates,
                "total_processing_stages": sum(len(doc.processing_history) for doc in self.metadata_store.documents.values())
            }
        }
    
    async def analyze_content_patterns(self) -> Dict[str, Any]:
        """Analyze content patterns and trends"""
        
        # Analyze content by time periods
        time_analysis = {}
        quality_trends = {}
        category_growth = {}
        
        current_time = datetime.now(timezone.utc)
        
        for document_metadata in self.metadata_store.documents.values():
            # Time-based analysis
            doc_age_days = (current_time - document_metadata.created_at).days
            
            if doc_age_days <= 1:
                period = "last_24h"
            elif doc_age_days <= 7:
                period = "last_week"
            elif doc_age_days <= 30:
                period = "last_month"
            else:
                period = "older"
            
            time_analysis.setdefault(period, {
                "count": 0,
                "total_quality": 0,
                "categories": {}
            })
            
            time_analysis[period]["count"] += 1
            time_analysis[period]["total_quality"] += document_metadata.quality_metrics.overall_score
            
            category = document_metadata.content_classification.category.value
            time_analysis[period]["categories"].setdefault(category, 0)
            time_analysis[period]["categories"][category] += 1
        
        # Calculate averages
        for period, data in time_analysis.items():
            if data["count"] > 0:
                data["average_quality"] = data["total_quality"] / data["count"]
            else:
                data["average_quality"] = 0
        
        return {
            "content_analysis": {
                "time_periods": time_analysis,
                "total_analyzed": len(self.metadata_store.documents)
            }
        }


# Global metadata integration instance
metadata_integration = MetadataIntegration()