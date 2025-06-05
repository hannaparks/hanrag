"""
Metadata Migration Helper

This module provides utilities to migrate existing vector database content
to the new metadata management system without requiring full re-ingestion.

Features:
- Backward compatibility with existing vectors
- Progressive metadata creation
- Metadata backfilling for existing content
- Hybrid operation mode
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import logging

from .metadata_store import (
    metadata_store,
    SourceType,
    ProcessingStage,
    DocumentMetadata
)
from .qdrant_client import QdrantManager

logger = logging.getLogger(__name__)


class MetadataMigrator:
    """Handles migration of existing content to metadata system"""
    
    def __init__(self):
        self.qdrant_manager = QdrantManager()
        self.metadata_store = metadata_store
    
    async def analyze_existing_content(self) -> Dict[str, Any]:
        """Analyze existing content in vector database"""
        
        try:
            # Get collection info
            collection_info = await self.qdrant_manager.get_collection_info()
            if not collection_info:
                return {"error": "No existing collection found"}
            
            total_points = collection_info.points_count
            print(f"📊 Found {total_points} existing vectors")
            
            # Sample existing points to understand structure
            sample_points = await self.qdrant_manager.scroll_points(limit=100)
            
            # Analyze metadata structure
            metadata_patterns = {}
            source_types = set()
            missing_metadata_count = 0
            
            for point in sample_points:
                payload = point.get("payload", {})
                
                # Check for existing metadata fields
                if "metadata" in payload:
                    for key in payload["metadata"].keys():
                        metadata_patterns[key] = metadata_patterns.get(key, 0) + 1
                
                # Check source information
                source = payload.get("source") or payload.get("metadata", {}).get("source")
                if source:
                    if "mattermost" in source.lower():
                        source_types.add("mattermost_channel")
                    elif source.startswith("http"):
                        source_types.add("url_webpage")
                    elif source.endswith(".md"):
                        source_types.add("markdown_file")
                    elif source.endswith(".pdf"):
                        source_types.add("pdf_document")
                    else:
                        source_types.add("unknown")
                else:
                    missing_metadata_count += 1
            
            analysis = {
                "total_vectors": total_points,
                "analyzed_sample": len(sample_points),
                "source_types_found": list(source_types),
                "common_metadata_fields": metadata_patterns,
                "missing_source_info": missing_metadata_count,
                "migration_recommended": total_points > 0
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze existing content: {e}")
            return {"error": str(e)}
    
    async def create_metadata_for_existing_point(
        self, 
        point_data: Dict[str, Any]
    ) -> Optional[DocumentMetadata]:
        """Create metadata for an existing vector point"""
        
        try:
            payload = point_data.get("payload", {})
            point_id = point_data.get("id")
            
            # Extract source information
            source_path = (
                payload.get("source") or 
                payload.get("metadata", {}).get("source") or
                f"unknown_source_{point_id}"
            )
            
            # Determine source type
            source_type = self._detect_source_type(source_path, payload)
            
            # Extract content if available
            content = (
                payload.get("content") or 
                payload.get("text") or
                payload.get("chunk_content") or
                ""
            )
            
            # Create metadata with migration flag
            document_metadata = await self.metadata_store.create_document_metadata(
                source_path=source_path,
                source_type=source_type,
                content=content,
                ingested_by="metadata_migration",
                tags=["migrated", "backfilled"],
                # Migration-specific metadata
                migration_timestamp=datetime.now(timezone.utc),
                original_point_id=point_id,
                **self._extract_migration_metadata(payload)
            )
            
            # Mark as migrated content
            await self.metadata_store.add_processing_stage(
                document_id=document_metadata.document_id,
                stage=ProcessingStage.INDEXING_COMPLETE,
                status="migrated",
                details={
                    "migration_source": "existing_vector",
                    "original_point_id": point_id,
                    "backfilled": True
                }
            )
            
            return document_metadata
            
        except Exception as e:
            logger.error(f"Failed to create metadata for point {point_data.get('id')}: {e}")
            return None
    
    async def progressive_migration(
        self, 
        batch_size: int = 50,
        max_batches: Optional[int] = None
    ) -> Dict[str, Any]:
        """Progressively migrate existing content to metadata system"""
        
        print("🔄 Starting progressive metadata migration...")
        
        migrated_count = 0
        error_count = 0
        batch_count = 0
        
        try:
            # Process in batches
            offset = None
            
            while True:
                if max_batches and batch_count >= max_batches:
                    break
                
                # Get batch of points
                points = await self.qdrant_manager.scroll_points(
                    limit=batch_size,
                    offset=offset
                )
                
                if not points:
                    break
                
                batch_count += 1
                print(f"📦 Processing batch {batch_count} ({len(points)} points)...")
                
                # Process each point in batch
                for point in points:
                    try:
                        # Check if metadata already exists
                        source_path = (
                            point.get("payload", {}).get("source") or
                            f"unknown_{point.get('id')}"
                        )
                        
                        existing_docs = await self.metadata_store.search_by_source(source_path)
                        
                        if not existing_docs:
                            # Create new metadata
                            metadata = await self.create_metadata_for_existing_point(point)
                            if metadata:
                                migrated_count += 1
                            else:
                                error_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing point {point.get('id')}: {e}")
                        error_count += 1
                
                # Update offset for next batch
                if len(points) < batch_size:
                    break
                
                # Small delay between batches
                await asyncio.sleep(0.1)
            
            migration_summary = {
                "status": "completed",
                "total_batches": batch_count,
                "migrated_documents": migrated_count,
                "errors": error_count,
                "migration_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            print(f"✅ Migration completed: {migrated_count} documents, {error_count} errors")
            return migration_summary
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "migrated_documents": migrated_count,
                "errors": error_count
            }
    
    async def create_hybrid_query_handler(self) -> callable:
        """Create a query handler that works with both metadata and legacy content"""
        
        async def hybrid_query(query: str, **kwargs):
            """Query handler that enriches results with metadata when available"""
            
            # Your existing query logic here
            # This would integrate with your current RAG pipeline
            
            # After getting search results, enrich with metadata
            
            # Pseudo-code for integration:
            # raw_results = await your_existing_query_method(query, **kwargs)
            # enriched_results = await metadata_integration.enrich_search_results(raw_results)
            # return enriched_results
            
        
        return hybrid_query
    
    def _detect_source_type(self, source_path: str, payload: Dict[str, Any]) -> SourceType:
        """Detect source type from path and payload"""
        
        source_lower = source_path.lower()
        
        # Check for Mattermost channels
        if "mattermost" in source_lower or payload.get("metadata", {}).get("channel_id"):
            return SourceType.MATTERMOST_CHANNEL
        
        # Check file extensions
        if source_path.startswith("http"):
            return SourceType.URL_WEBPAGE
        elif source_path.endswith(".md"):
            return SourceType.MARKDOWN_FILE
        elif source_path.endswith(".pdf"):
            return SourceType.PDF_DOCUMENT
        elif source_path.endswith(".docx"):
            return SourceType.WORD_DOCUMENT
        elif source_path.endswith(".csv"):
            return SourceType.CSV_DATA
        elif source_path.endswith(".json"):
            return SourceType.JSON_DATA
        elif source_path.endswith(".py"):
            return SourceType.CODE_FILE
        else:
            return SourceType.UNKNOWN
    
    def _extract_migration_metadata(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant metadata from existing payload"""
        
        metadata = {}
        
        # Extract channel/team info for Mattermost content
        existing_metadata = payload.get("metadata", {})
        
        if "channel_id" in existing_metadata:
            metadata["channel_id"] = existing_metadata["channel_id"]
        if "team_id" in existing_metadata:
            metadata["team_id"] = existing_metadata["team_id"]
        if "user_id" in existing_metadata:
            metadata["user_id"] = existing_metadata["user_id"]
        
        # Extract timestamps
        if "timestamp" in existing_metadata:
            metadata["extraction_timestamp"] = existing_metadata["timestamp"]
        
        # Extract file info
        if "file_size" in existing_metadata:
            metadata["file_size"] = existing_metadata["file_size"]
        
        return metadata


# Global migrator instance
metadata_migrator = MetadataMigrator()


async def quick_migration_check():
    """Quick check to see if migration is needed"""
    
    print("🔍 Checking existing content for metadata migration...")
    
    analysis = await metadata_migrator.analyze_existing_content()
    
    if "error" in analysis:
        print(f"ℹ️  {analysis['error']}")
        return False
    
    print(f"📊 Analysis Results:")
    print(f"  • Total vectors: {analysis['total_vectors']}")
    print(f"  • Source types: {', '.join(analysis['source_types_found'])}")
    print(f"  • Missing source info: {analysis['missing_source_info']}")
    
    if analysis["migration_recommended"]:
        print("\n💡 Recommendation: Run progressive migration for full metadata benefits")
        print("   This will create metadata for existing content without re-ingestion")
    else:
        print("\n✅ No migration needed - start fresh with metadata tracking")
    
    return analysis["migration_recommended"]


if __name__ == "__main__":
    # Quick check
    asyncio.run(quick_migration_check())