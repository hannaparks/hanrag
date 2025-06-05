"""
Metadata Management System

This module provides comprehensive metadata management for the RAG system,
including source attribution tracking, content classification, quality scores,
and complete audit trail from raw data to indexed vectors.

Key Features:
- Source attribution tracking with complete lineage
- Content classification and categorization  
- Quality score management and analysis
- Audit trail from ingestion to vector storage
- Cross-reference tracking between documents
- Metadata enrichment and validation
- Performance analytics and reporting
"""

import hashlib
import logging
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum

# Configuration import (can be optional for standalone testing)
try:
    from ..config.settings import settings
except ImportError:
    # Fallback for standalone testing
    class MockSettings:
        pass
    settings = MockSettings()

logger = logging.getLogger(__name__)


class SourceType(str, Enum):
    """Source types for content classification"""
    MATTERMOST_CHANNEL = "mattermost_channel"
    URL_WEBPAGE = "url_webpage"
    PDF_DOCUMENT = "pdf_document"
    WORD_DOCUMENT = "word_document"
    MARKDOWN_FILE = "markdown_file"
    TEXT_FILE = "text_file"
    CSV_DATA = "csv_data"
    JSON_DATA = "json_data"
    XML_DATA = "xml_data"
    CODE_FILE = "code_file"
    EMAIL_MESSAGE = "email_message"
    UNKNOWN = "unknown"


class ContentCategory(str, Enum):
    """Content categories for classification"""
    DOCUMENTATION = "documentation"
    CONVERSATION = "conversation"
    CODE = "code"
    CONFIGURATION = "configuration"
    KNOWLEDGE_BASE = "knowledge_base"
    TUTORIAL = "tutorial"
    REFERENCE = "reference"
    DISCUSSION = "discussion"
    ANNOUNCEMENT = "announcement"
    QUESTION_ANSWER = "question_answer"
    TECHNICAL_SPEC = "technical_spec"
    USER_GUIDE = "user_guide"
    TROUBLESHOOTING = "troubleshooting"
    CHANGELOG = "changelog"
    MEETING_NOTES = "meeting_notes"
    OTHER = "other"


class ProcessingStage(str, Enum):
    """Processing stages in the ingestion pipeline"""
    RAW_INGESTION = "raw_ingestion"
    FORMAT_DETECTION = "format_detection"
    CONTENT_EXTRACTION = "content_extraction"
    PREPROCESSING = "preprocessing"
    CHUNKING = "chunking"
    QUALITY_ASSESSMENT = "quality_assessment"
    EMBEDDING_GENERATION = "embedding_generation"
    VECTOR_STORAGE = "vector_storage"
    INDEXING_COMPLETE = "indexing_complete"


@dataclass
class SourceAttribution:
    """Complete source attribution information"""
    source_id: str
    source_type: SourceType
    original_path: str
    source_url: Optional[str] = None
    parent_source_id: Optional[str] = None  # For nested sources
    extraction_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Mattermost-specific fields
    channel_id: Optional[str] = None
    team_id: Optional[str] = None
    message_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # File-specific fields
    file_size: Optional[int] = None
    file_hash: Optional[str] = None
    file_modified: Optional[datetime] = None
    
    # Web-specific fields
    domain: Optional[str] = None
    page_title: Optional[str] = None
    last_crawled: Optional[datetime] = None
    
    # Additional metadata
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentClassification:
    """Content classification and categorization"""
    category: ContentCategory
    subcategory: Optional[str] = None
    confidence: float = 0.0
    keywords: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    language: str = "en"
    technical_level: str = "unknown"  # beginner, intermediate, advanced, expert
    audience: str = "general"  # general, technical, administrative, developer
    
    # Classification metadata
    classifier_version: str = "1.0"
    classification_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    manual_override: bool = False


@dataclass
class QualityMetrics:
    """Comprehensive quality scoring"""
    overall_score: float = 0.0
    relevance_score: float = 0.0
    authority_score: float = 0.0
    readability_score: float = 0.0
    completeness_score: float = 0.0
    freshness_score: float = 0.0
    uniqueness_score: float = 0.0
    
    # Quality indicators
    has_duplicates: bool = False
    duplicate_count: int = 0
    is_truncated: bool = False
    has_formatting_issues: bool = False
    
    # Quality details
    quality_issues: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    
    # Assessment metadata
    assessment_version: str = "1.0"
    assessment_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ProcessingHistory:
    """Audit trail of processing stages"""
    stage: ProcessingStage
    status: str  # success, warning, error
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processing_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Processing results
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    chunks_created: Optional[int] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ChunkMetadata:
    """Metadata for individual content chunks"""
    chunk_id: str
    parent_chunk_id: Optional[str] = None
    chunk_index: int = 0
    total_chunks: int = 1
    
    # Content properties
    content_hash: str = ""
    content_length: int = 0
    token_count: int = 0
    embedding_vector_id: Optional[str] = None
    
    # Positioning information
    start_position: int = 0
    end_position: int = 0
    overlap_with_previous: int = 0
    overlap_with_next: int = 0
    
    # Hierarchical relationships
    sibling_chunks: List[str] = field(default_factory=list)
    child_chunks: List[str] = field(default_factory=list)
    
    # Processing metadata
    chunking_strategy: str = "hierarchical"
    chunking_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DocumentMetadata:
    """Complete metadata for a document"""
    document_id: str
    source_attribution: SourceAttribution
    content_classification: ContentClassification
    quality_metrics: QualityMetrics
    processing_history: List[ProcessingHistory] = field(default_factory=list)
    chunk_metadata: List[ChunkMetadata] = field(default_factory=list)
    
    # Cross-references
    related_documents: List[str] = field(default_factory=list)
    referenced_by: List[str] = field(default_factory=list)
    duplicate_of: Optional[str] = None
    
    # Version control
    version: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Access and usage
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    ingested_by: Optional[str] = None
    
    # Custom metadata
    tags: List[str] = field(default_factory=list)
    annotations: Dict[str, Any] = field(default_factory=dict)


class MetadataStore:
    """Centralized metadata management system"""
    
    def __init__(self):
        """Initialize metadata store"""
        self.documents: Dict[str, DocumentMetadata] = {}
        self.source_index: Dict[str, Set[str]] = {}  # source_id -> document_ids
        self.category_index: Dict[ContentCategory, Set[str]] = {}
        self.tag_index: Dict[str, Set[str]] = {}
        self.quality_index: Dict[str, Set[str]] = {}  # quality_tier -> document_ids
        
        # Performance tracking
        self._stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "quality_distribution": {},
            "category_distribution": {},
            "source_type_distribution": {},
            "last_updated": datetime.now(timezone.utc)
        }
    
    async def create_document_metadata(
        self,
        source_path: str,
        source_type: SourceType,
        content: str,
        **kwargs
    ) -> DocumentMetadata:
        """Create comprehensive metadata for a new document"""
        
        # Generate unique document ID
        document_id = self._generate_document_id(source_path, source_type)
        
        # Create source attribution
        source_attribution = SourceAttribution(
            source_id=document_id,
            source_type=source_type,
            original_path=source_path,
            **{k: v for k, v in kwargs.items() if k in SourceAttribution.__dataclass_fields__}
        )
        
        # Classify content
        content_classification = await self._classify_content(content, source_attribution)
        
        # Assess quality
        quality_metrics = await self._assess_quality(content, source_attribution, content_classification)
        
        # Create document metadata
        metadata = DocumentMetadata(
            document_id=document_id,
            source_attribution=source_attribution,
            content_classification=content_classification,
            quality_metrics=quality_metrics,
            ingested_by=kwargs.get("user_id"),
            tags=kwargs.get("tags", [])
        )
        
        # Add initial processing history
        await self._add_processing_stage(
            metadata,
            ProcessingStage.RAW_INGESTION,
            "success",
            {"source_path": source_path, "content_length": len(content)}
        )
        
        # Store and index
        await self._store_document_metadata(metadata)
        
        logger.info(f"Created metadata for document {document_id} from {source_path}")
        return metadata
    
    async def add_chunk_metadata(
        self,
        document_id: str,
        chunk_id: str,
        content: str,
        chunk_index: int,
        total_chunks: int,
        **kwargs
    ) -> ChunkMetadata:
        """Add metadata for a content chunk"""
        
        if document_id not in self.documents:
            raise ValueError(f"Document {document_id} not found in metadata store")
        
        # Create content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Create chunk metadata
        chunk_metadata = ChunkMetadata(
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            content_hash=content_hash,
            content_length=len(content),
            token_count=kwargs.get("token_count", len(content.split())),
            **{k: v for k, v in kwargs.items() if k in ChunkMetadata.__dataclass_fields__ and k != "token_count"}
        )
        
        # Add to document
        self.documents[document_id].chunk_metadata.append(chunk_metadata)
        self.documents[document_id].last_updated = datetime.now(timezone.utc)
        
        # Update stats
        self._stats["total_chunks"] += 1
        
        logger.debug(f"Added chunk metadata {chunk_id} to document {document_id}")
        return chunk_metadata
    
    async def add_processing_stage(
        self,
        document_id: str,
        stage: ProcessingStage,
        status: str,
        details: Optional[Dict[str, Any]] = None,
        processing_time: float = 0.0
    ) -> None:
        """Add a processing stage to the audit trail"""
        
        if document_id not in self.documents:
            raise ValueError(f"Document {document_id} not found in metadata store")
        
        await self._add_processing_stage(
            self.documents[document_id],
            stage,
            status,
            details or {},
            processing_time
        )
    
    async def update_quality_metrics(
        self,
        document_id: str,
        quality_metrics: QualityMetrics
    ) -> None:
        """Update quality metrics for a document"""
        
        if document_id not in self.documents:
            raise ValueError(f"Document {document_id} not found in metadata store")
        
        # Update metrics
        old_quality_tier = self._get_quality_tier(self.documents[document_id].quality_metrics.overall_score)
        self.documents[document_id].quality_metrics = quality_metrics
        self.documents[document_id].last_updated = datetime.now(timezone.utc)
        
        # Update quality index
        new_quality_tier = self._get_quality_tier(quality_metrics.overall_score)
        if old_quality_tier != new_quality_tier:
            self.quality_index.setdefault(old_quality_tier, set()).discard(document_id)
            self.quality_index.setdefault(new_quality_tier, set()).add(document_id)
        
        logger.debug(f"Updated quality metrics for document {document_id}")
    
    async def get_document_metadata(self, document_id: str) -> Optional[DocumentMetadata]:
        """Retrieve document metadata"""
        return self.documents.get(document_id)
    
    async def search_by_source(self, source_id: str) -> List[DocumentMetadata]:
        """Find documents by source ID"""
        document_ids = self.source_index.get(source_id, set())
        return [self.documents[doc_id] for doc_id in document_ids if doc_id in self.documents]
    
    async def search_by_category(self, category: ContentCategory) -> List[DocumentMetadata]:
        """Find documents by content category"""
        document_ids = self.category_index.get(category, set())
        return [self.documents[doc_id] for doc_id in document_ids if doc_id in self.documents]
    
    async def search_by_quality(self, min_score: float = 0.0, max_score: float = 1.0) -> List[DocumentMetadata]:
        """Find documents by quality score range"""
        results = []
        for doc_id, metadata in self.documents.items():
            score = metadata.quality_metrics.overall_score
            if min_score <= score <= max_score:
                results.append(metadata)
        return results
    
    async def search_by_tags(self, tags: List[str], match_all: bool = False) -> List[DocumentMetadata]:
        """Find documents by tags"""
        results = []
        for doc_id, metadata in self.documents.items():
            if match_all:
                if all(tag in metadata.tags for tag in tags):
                    results.append(metadata)
            else:
                if any(tag in metadata.tags for tag in tags):
                    results.append(metadata)
        return results
    
    async def get_related_documents(self, document_id: str, max_results: int = 10) -> List[DocumentMetadata]:
        """Find documents related to the given document"""
        if document_id not in self.documents:
            return []
        
        document = self.documents[document_id]
        related_ids = set(document.related_documents)
        
        # Add documents with same category
        category_docs = self.category_index.get(document.content_classification.category, set())
        related_ids.update(list(category_docs)[:max_results])
        
        # Remove self
        related_ids.discard(document_id)
        
        # Return top results
        results = [self.documents[doc_id] for doc_id in related_ids if doc_id in self.documents]
        return results[:max_results]
    
    async def get_audit_trail(self, document_id: str) -> List[ProcessingHistory]:
        """Get complete audit trail for a document"""
        if document_id not in self.documents:
            return []
        
        return self.documents[document_id].processing_history.copy()
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive metadata statistics"""
        
        # Update real-time stats
        self._stats["total_documents"] = len(self.documents)
        self._stats["total_chunks"] = sum(len(doc.chunk_metadata) for doc in self.documents.values())
        
        # Calculate distributions
        category_dist = {}
        source_type_dist = {}
        quality_dist = {"high": 0, "medium": 0, "low": 0}
        
        for metadata in self.documents.values():
            # Category distribution
            cat = metadata.content_classification.category.value
            category_dist[cat] = category_dist.get(cat, 0) + 1
            
            # Source type distribution
            src_type = metadata.source_attribution.source_type.value
            source_type_dist[src_type] = source_type_dist.get(src_type, 0) + 1
            
            # Quality distribution
            quality_tier = self._get_quality_tier(metadata.quality_metrics.overall_score)
            quality_dist[quality_tier] += 1
        
        self._stats.update({
            "category_distribution": category_dist,
            "source_type_distribution": source_type_dist,
            "quality_distribution": quality_dist,
            "last_updated": datetime.now(timezone.utc)
        })
        
        return self._stats.copy()
    
    async def export_metadata(self, document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Export metadata for backup or analysis"""
        
        if document_ids is None:
            documents_to_export = self.documents
        else:
            documents_to_export = {
                doc_id: self.documents[doc_id] 
                for doc_id in document_ids 
                if doc_id in self.documents
            }
        
        export_data = {
            "metadata_version": "1.0",
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_documents": len(documents_to_export),
            "documents": {}
        }
        
        for doc_id, metadata in documents_to_export.items():
            export_data["documents"][doc_id] = self._serialize_metadata(metadata)
        
        return export_data
    
    async def import_metadata(self, import_data: Dict[str, Any]) -> bool:
        """Import metadata from backup or external source"""
        
        try:
            documents_data = import_data.get("documents", {})
            
            for doc_id, metadata_dict in documents_data.items():
                metadata = self._deserialize_metadata(metadata_dict)
                await self._store_document_metadata(metadata)
            
            logger.info(f"Imported metadata for {len(documents_data)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import metadata: {e}")
            return False
    
    # Private helper methods
    
    def _generate_document_id(self, source_path: str, source_type: SourceType) -> str:
        """Generate unique document ID"""
        content_hash = hashlib.md5(f"{source_path}:{source_type.value}".encode()).hexdigest()
        return f"doc_{content_hash}_{int(datetime.now().timestamp())}"
    
    async def _classify_content(
        self, 
        content: str, 
        source_attribution: SourceAttribution
    ) -> ContentClassification:
        """Classify content based on content and source"""
        
        # Simple heuristic-based classification
        content_lower = content.lower()
        
        # Determine category based on keywords and source type
        if source_attribution.source_type == SourceType.MATTERMOST_CHANNEL:
            if any(word in content_lower for word in ["question", "help", "how to", "?"]):
                category = ContentCategory.QUESTION_ANSWER
            elif any(word in content_lower for word in ["announcement", "update", "release"]):
                category = ContentCategory.ANNOUNCEMENT
            else:
                category = ContentCategory.CONVERSATION
        elif "readme" in source_attribution.original_path.lower():
            category = ContentCategory.DOCUMENTATION
        elif any(ext in source_attribution.original_path.lower() for ext in [".py", ".js", ".java", ".cpp"]):
            category = ContentCategory.CODE
        elif "config" in source_attribution.original_path.lower():
            category = ContentCategory.CONFIGURATION
        else:
            category = ContentCategory.OTHER
        
        # Extract keywords (simple approach)
        keywords = self._extract_keywords(content)
        
        return ContentClassification(
            category=category,
            confidence=0.7,  # Default confidence
            keywords=keywords[:10],  # Top 10 keywords
            language="en",  # Default to English
            technical_level="intermediate"  # Default level
        )
    
    async def _assess_quality(
        self,
        content: str,
        source_attribution: SourceAttribution,
        content_classification: ContentClassification
    ) -> QualityMetrics:
        """Assess content quality across multiple dimensions"""
        
        # Simple quality assessment
        metrics = QualityMetrics()
        
        # Length-based completeness
        content_length = len(content)
        if content_length > 1000:
            metrics.completeness_score = 0.9
        elif content_length > 500:
            metrics.completeness_score = 0.7
        else:
            metrics.completeness_score = 0.5
        
        # Readability (simple heuristic)
        words = content.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        metrics.readability_score = max(0.1, min(1.0, 1.0 - (avg_word_length - 5) / 10))
        
        # Authority based on source type
        authority_weights = {
            SourceType.MATTERMOST_CHANNEL: 0.8,
            SourceType.PDF_DOCUMENT: 0.9,
            SourceType.WORD_DOCUMENT: 0.8,
            SourceType.MARKDOWN_FILE: 0.7,
            SourceType.URL_WEBPAGE: 0.6
        }
        metrics.authority_score = authority_weights.get(source_attribution.source_type, 0.5)
        
        # Relevance based on classification confidence
        metrics.relevance_score = content_classification.confidence
        
        # Freshness (recent is better)
        if source_attribution.extraction_timestamp:
            days_old = (datetime.now(timezone.utc) - source_attribution.extraction_timestamp).days
            metrics.freshness_score = max(0.1, 1.0 - days_old / 365)
        else:
            metrics.freshness_score = 0.5
        
        # Overall quality (weighted average)
        metrics.overall_score = (
            metrics.completeness_score * 0.25 +
            metrics.readability_score * 0.2 +
            metrics.authority_score * 0.25 +
            metrics.relevance_score * 0.15 +
            metrics.freshness_score * 0.15
        )
        
        return metrics
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content"""
        # Simple keyword extraction
        words = content.lower().split()
        # Filter out common words and short words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Count frequency and return most common
        from collections import Counter
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(20)]
    
    async def _add_processing_stage(
        self,
        metadata: DocumentMetadata,
        stage: ProcessingStage,
        status: str,
        details: Dict[str, Any],
        processing_time: float = 0.0
    ) -> None:
        """Add processing stage to audit trail"""
        
        history_entry = ProcessingHistory(
            stage=stage,
            status=status,
            processing_time=processing_time,
            details=details
        )
        
        metadata.processing_history.append(history_entry)
        metadata.last_updated = datetime.now(timezone.utc)
    
    async def _store_document_metadata(self, metadata: DocumentMetadata) -> None:
        """Store and index document metadata"""
        
        document_id = metadata.document_id
        
        # Store document
        self.documents[document_id] = metadata
        
        # Update source index
        source_id = metadata.source_attribution.source_id
        self.source_index.setdefault(source_id, set()).add(document_id)
        
        # Update category index
        category = metadata.content_classification.category
        self.category_index.setdefault(category, set()).add(document_id)
        
        # Update tag index
        for tag in metadata.tags:
            self.tag_index.setdefault(tag, set()).add(document_id)
        
        # Update quality index
        quality_tier = self._get_quality_tier(metadata.quality_metrics.overall_score)
        self.quality_index.setdefault(quality_tier, set()).add(document_id)
        
        # Update stats
        self._stats["total_documents"] = len(self.documents)
    
    def _get_quality_tier(self, score: float) -> str:
        """Get quality tier from score"""
        if score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _serialize_metadata(self, metadata: DocumentMetadata) -> Dict[str, Any]:
        """Serialize metadata to JSON-compatible format"""
        return asdict(metadata)
    
    def _deserialize_metadata(self, data: Dict[str, Any]) -> DocumentMetadata:
        """Deserialize metadata from JSON format"""
        # This is a simplified implementation
        # In practice, you'd need proper deserialization with type conversion
        return DocumentMetadata(**data)


# Global metadata store instance
metadata_store = MetadataStore()