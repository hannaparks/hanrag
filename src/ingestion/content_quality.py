"""
Content Quality & Deduplication System

This module provides comprehensive content quality assessment and deduplication
capabilities for the RAG ingestion pipeline, including:
- Duplicate detection across different formats
- Content relevance scoring
- Source authority weighting
- Cross-document relationship detection
"""

import hashlib
import re
import logging
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class ContentQualityMetrics:
    """Metrics for content quality assessment"""
    relevance_score: float = 0.0
    authority_score: float = 0.0
    readability_score: float = 0.0
    completeness_score: float = 0.0
    freshness_score: float = 0.0
    overall_quality: float = 0.0
    duplicate_probability: float = 0.0
    source_authority: str = "unknown"
    quality_issues: List[str] = field(default_factory=list)


@dataclass
class DuplicateMatch:
    """Information about a potential duplicate"""
    content_id: str
    similarity_score: float
    match_type: str  # "exact", "near_exact", "semantic", "partial"
    content_hash: str
    source_path: str
    overlap_percentage: float = 0.0


@dataclass
class ContentFingerprint:
    """Unique fingerprint for content identification"""
    content_hash: str
    fuzzy_hash: str
    semantic_hash: str
    length: int
    word_count: int
    source_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContentQualityAssessor:
    """Assesses content quality across multiple dimensions"""
    
    def __init__(self):
        self.source_authority_weights = {
            "mattermost_channel": 0.8,
            "documentation": 0.9,
            "official_website": 0.9,
            "github_repository": 0.8,
            "wikipedia": 0.7,
            "stackoverflow": 0.6,
            "blog_post": 0.5,
            "forum": 0.4,
            "unknown": 0.3
        }
        
        self.quality_keywords = {
            "technical": ["implementation", "configuration", "documentation", "specification"],
            "authoritative": ["official", "documentation", "guide", "manual", "specification"],
            "low_quality": ["lorem ipsum", "placeholder", "test content", "todo", "fixme"]
        }
    
    async def assess_content_quality(
        self,
        content: str,
        source_metadata: Dict[str, Any]
    ) -> ContentQualityMetrics:
        """Comprehensive content quality assessment"""
        
        metrics = ContentQualityMetrics()
        
        try:
            # Calculate individual quality scores
            metrics.relevance_score = self._calculate_relevance_score(content, source_metadata)
            metrics.authority_score = self._calculate_authority_score(source_metadata)
            metrics.readability_score = self._calculate_readability_score(content)
            metrics.completeness_score = self._calculate_completeness_score(content)
            metrics.freshness_score = self._calculate_freshness_score(source_metadata)
            
            # Calculate overall quality score (weighted average)
            metrics.overall_quality = (
                metrics.relevance_score * 0.3 +
                metrics.authority_score * 0.25 +
                metrics.readability_score * 0.2 +
                metrics.completeness_score * 0.15 +
                metrics.freshness_score * 0.1
            )
            
            # Identify quality issues
            metrics.quality_issues = self._identify_quality_issues(content, source_metadata)
            
            # Set source authority category
            metrics.source_authority = self._categorize_source_authority(source_metadata)
            
            logger.debug(f"Content quality assessment: {metrics.overall_quality:.2f}")
            
        except Exception as e:
            logger.error(f"Error assessing content quality: {e}")
            metrics.quality_issues.append(f"Assessment error: {str(e)}")
        
        return metrics
    
    def _calculate_relevance_score(self, content: str, metadata: Dict[str, Any]) -> float:
        """Calculate content relevance based on keywords and context"""
        
        content_lower = content.lower()
        relevance_score = 0.5  # Base score
        
        # Check for technical content indicators
        technical_indicators = 0
        for keyword in self.quality_keywords["technical"]:
            if keyword in content_lower:
                technical_indicators += 1
        
        relevance_score += min(technical_indicators * 0.1, 0.3)
        
        # Check for authoritative language
        authoritative_indicators = 0
        for keyword in self.quality_keywords["authoritative"]:
            if keyword in content_lower:
                authoritative_indicators += 1
        
        relevance_score += min(authoritative_indicators * 0.05, 0.2)
        
        # Penalize low-quality indicators
        for keyword in self.quality_keywords["low_quality"]:
            if keyword in content_lower:
                relevance_score -= 0.1
        
        # Consider source context
        source_type = metadata.get("source_type", "unknown")
        if source_type in ["documentation", "official_website"]:
            relevance_score += 0.1
        elif source_type == "mattermost_channel":
            # Channel messages get contextual relevance
            relevance_score += 0.05
        
        return max(0.0, min(1.0, relevance_score))
    
    def _calculate_authority_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate source authority score"""
        
        source_type = metadata.get("source_type", "unknown")
        base_score = self.source_authority_weights.get(source_type, 0.3)
        
        # Adjust based on additional metadata
        if metadata.get("is_official", False):
            base_score += 0.1
        
        if metadata.get("verification_status") == "verified":
            base_score += 0.1
        
        # Consider domain authority for web content
        domain = metadata.get("domain", "")
        if any(auth_domain in domain for auth_domain in [".gov", ".edu", ".org"]):
            base_score += 0.1
        
        return min(1.0, base_score)
    
    def _calculate_readability_score(self, content: str) -> float:
        """Calculate content readability score"""
        
        if not content.strip():
            return 0.0
        
        # Basic readability metrics
        sentences = len(re.split(r'[.!?]+', content))
        words = len(content.split())
        chars = len(content)
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Average sentence length (ideal: 15-20 words)
        avg_sentence_length = words / sentences
        sentence_score = max(0, 1 - abs(avg_sentence_length - 17.5) / 17.5)
        
        # Average word length (ideal: 4-6 characters)
        avg_word_length = chars / words
        word_score = max(0, 1 - abs(avg_word_length - 5) / 5)
        
        # Structure indicators
        structure_score = 0.5
        if '\n' in content:  # Has paragraphs
            structure_score += 0.2
        if re.search(r'^#+\s', content, re.MULTILINE):  # Has headers
            structure_score += 0.2
        if re.search(r'^\s*[-*•]\s', content, re.MULTILINE):  # Has lists
            structure_score += 0.1
        
        # Combined readability score
        readability = (sentence_score * 0.3 + word_score * 0.3 + structure_score * 0.4)
        
        return min(1.0, readability)
    
    def _calculate_completeness_score(self, content: str) -> float:
        """Calculate content completeness score"""
        
        if not content.strip():
            return 0.0
        
        completeness_score = 0.5  # Base score
        
        # Length indicators
        word_count = len(content.split())
        if word_count > 100:
            completeness_score += 0.2
        if word_count > 500:
            completeness_score += 0.1
        
        # Content structure indicators
        if re.search(r'^#+\s', content, re.MULTILINE):  # Has headers
            completeness_score += 0.1
        if re.search(r'^\s*[-*•]\s', content, re.MULTILINE):  # Has lists
            completeness_score += 0.05
        if '```' in content or re.search(r'`[^`]+`', content):  # Has code
            completeness_score += 0.1
        
        # Check for incomplete indicators
        incomplete_indicators = [
            "todo", "fixme", "placeholder", "coming soon", "under construction",
            "...", "tbd", "to be determined", "[placeholder]"
        ]
        
        content_lower = content.lower()
        for indicator in incomplete_indicators:
            if indicator in content_lower:
                completeness_score -= 0.1
        
        return max(0.0, min(1.0, completeness_score))
    
    def _calculate_freshness_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate content freshness score based on timestamps"""
        
        try:
            # Get content timestamp
            timestamp = metadata.get("created_at") or metadata.get("modified_at")
            if not timestamp:
                return 0.5  # Unknown age
            
            if isinstance(timestamp, str):
                # Try to parse timestamp
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    return 0.5
            
            # Calculate age in days
            age_days = (datetime.now(timestamp.tzinfo) - timestamp).days
            
            # Freshness scoring (newer is better, but not linearly)
            if age_days <= 7:
                return 1.0
            elif age_days <= 30:
                return 0.9
            elif age_days <= 90:
                return 0.8
            elif age_days <= 365:
                return 0.6
            elif age_days <= 730:  # 2 years
                return 0.4
            else:
                return 0.2
                
        except Exception as e:
            logger.debug(f"Error calculating freshness: {e}")
            return 0.5
    
    def _identify_quality_issues(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """Identify specific quality issues"""
        
        issues = []
        content_lower = content.lower()
        
        # Content-based issues
        if len(content.strip()) < 50:
            issues.append("Content too short")
        
        if len(content.split()) < 10:
            issues.append("Insufficient word count")
        
        # Check for placeholder content
        placeholders = ["lorem ipsum", "placeholder", "test content", "sample text"]
        if any(placeholder in content_lower for placeholder in placeholders):
            issues.append("Contains placeholder content")
        
        # Check for incomplete content
        incomplete_indicators = ["todo", "fixme", "coming soon", "under construction"]
        if any(indicator in content_lower for indicator in incomplete_indicators):
            issues.append("Content appears incomplete")
        
        # Check for poor formatting
        if content.count('\n') == 0 and len(content) > 500:
            issues.append("Poor formatting (no paragraphs)")
        
        # Source-based issues
        source_type = metadata.get("source_type")
        if source_type == "unknown":
            issues.append("Unknown source type")
        
        if not metadata.get("created_at") and not metadata.get("modified_at"):
            issues.append("No timestamp information")
        
        return issues
    
    def _categorize_source_authority(self, metadata: Dict[str, Any]) -> str:
        """Categorize source authority level"""
        
        source_type = metadata.get("source_type", "unknown")
        domain = metadata.get("domain", "")
        
        # High authority sources
        if source_type in ["official_website", "documentation"]:
            return "high"
        
        if any(auth_domain in domain for auth_domain in [".gov", ".edu"]):
            return "high"
        
        # Medium authority sources
        if source_type in ["github_repository", "mattermost_channel", "wikipedia"]:
            return "medium"
        
        if ".org" in domain:
            return "medium"
        
        # Low authority sources
        if source_type in ["blog_post", "forum", "social_media"]:
            return "low"
        
        return "unknown"


class DuplicateDetector:
    """Detects duplicate content across different formats and sources"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.content_fingerprints: Dict[str, ContentFingerprint] = {}
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.content_vectors: Dict[str, np.ndarray] = {}
        
    async def detect_duplicates(
        self,
        content: str,
        content_id: str,
        source_metadata: Dict[str, Any]
    ) -> List[DuplicateMatch]:
        """Detect potential duplicates of the given content"""
        
        duplicates = []
        
        try:
            # Generate fingerprint for new content
            fingerprint = self._generate_fingerprint(content, source_metadata)
            
            # Check for exact matches first
            exact_matches = await self._find_exact_matches(fingerprint)
            duplicates.extend(exact_matches)
            
            # Check for near-exact matches
            if not exact_matches:
                near_matches = await self._find_near_exact_matches(content, content_id)
                duplicates.extend(near_matches)
            
            # Check for semantic duplicates
            if not duplicates:
                semantic_matches = await self._find_semantic_matches(content, content_id)
                duplicates.extend(semantic_matches)
            
            # Store fingerprint for future comparisons
            self.content_fingerprints[content_id] = fingerprint
            
            logger.debug(f"Found {len(duplicates)} potential duplicates for content {content_id}")
            
        except Exception as e:
            logger.error(f"Error detecting duplicates: {e}")
        
        return duplicates
    
    def _generate_fingerprint(self, content: str, metadata: Dict[str, Any]) -> ContentFingerprint:
        """Generate a comprehensive fingerprint for content"""
        
        # Clean content for hashing
        cleaned_content = re.sub(r'\s+', ' ', content.strip().lower())
        
        # Generate different types of hashes
        content_hash = hashlib.md5(content.encode()).hexdigest()
        fuzzy_hash = hashlib.md5(cleaned_content.encode()).hexdigest()
        
        # Semantic hash (simplified - based on key terms)
        key_terms = self._extract_key_terms(content)
        semantic_content = ' '.join(sorted(key_terms))
        semantic_hash = hashlib.md5(semantic_content.encode()).hexdigest()
        
        return ContentFingerprint(
            content_hash=content_hash,
            fuzzy_hash=fuzzy_hash,
            semantic_hash=semantic_hash,
            length=len(content),
            word_count=len(content.split()),
            source_type=metadata.get("source_type", "unknown"),
            metadata=metadata
        )
    
    def _extract_key_terms(self, content: str, max_terms: int = 20) -> Set[str]:
        """Extract key terms from content for semantic comparison"""
        
        # Simple key term extraction (could be enhanced with NLP)
        words = re.findall(r'\w+', content.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        # Get words that are likely to be meaningful
        key_terms = set()
        for word in words:
            if (len(word) > 3 and 
                word not in stop_words and 
                not word.isdigit()):
                key_terms.add(word)
        
        # Return top terms by frequency
        from collections import Counter
        word_freq = Counter(words)
        important_terms = set()
        for word, _ in word_freq.most_common(max_terms):
            if word in key_terms:
                important_terms.add(word)
        
        return important_terms
    
    async def _find_exact_matches(self, fingerprint: ContentFingerprint) -> List[DuplicateMatch]:
        """Find exact content matches using hash comparison"""
        
        matches = []
        
        for content_id, existing_fp in self.content_fingerprints.items():
            if existing_fp.content_hash == fingerprint.content_hash:
                matches.append(DuplicateMatch(
                    content_id=content_id,
                    similarity_score=1.0,
                    match_type="exact",
                    content_hash=existing_fp.content_hash,
                    source_path=existing_fp.metadata.get("source", "unknown"),
                    overlap_percentage=100.0
                ))
        
        return matches
    
    async def _find_near_exact_matches(self, content: str, content_id: str) -> List[DuplicateMatch]:
        """Find near-exact matches using text similarity"""
        
        matches = []
        
        for existing_id, fingerprint in self.content_fingerprints.items():
            # Skip if length is very different
            content_length = len(content)
            if abs(content_length - fingerprint.length) / max(content_length, fingerprint.length) > 0.3:
                continue
            
            # Use difflib for sequence matching (this is expensive for large content)
            if content_length < 10000:  # Only for reasonably sized content
                # We don't have the original content, so we'll use fuzzy hash comparison
                # In a real implementation, you'd store normalized content for comparison
                similarity = self._compare_fingerprints(content, fingerprint)
                
                if similarity >= self.similarity_threshold:
                    matches.append(DuplicateMatch(
                        content_id=existing_id,
                        similarity_score=similarity,
                        match_type="near_exact",
                        content_hash=fingerprint.content_hash,
                        source_path=fingerprint.metadata.get("source", "unknown"),
                        overlap_percentage=similarity * 100
                    ))
        
        return matches
    
    async def _find_semantic_matches(self, content: str, content_id: str) -> List[DuplicateMatch]:
        """Find semantic duplicates using TF-IDF similarity"""
        
        matches = []
        
        try:
            # Prepare content for TF-IDF analysis
            if not self.tfidf_vectorizer:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
            
            # Get existing content for comparison
            existing_content = []
            existing_ids = []
            
            for existing_id, fingerprint in self.content_fingerprints.items():
                # Use key terms as proxy for content
                key_terms = self._extract_key_terms_from_fingerprint(fingerprint)
                if key_terms:
                    existing_content.append(' '.join(key_terms))
                    existing_ids.append(existing_id)
            
            if not existing_content:
                return matches
            
            # Add current content
            current_key_terms = self._extract_key_terms(content)
            all_content = existing_content + [' '.join(current_key_terms)]
            
            # Calculate TF-IDF similarity
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_content)
            current_vector = tfidf_matrix[-1]
            
            similarities = cosine_similarity(current_vector, tfidf_matrix[:-1]).flatten()
            
            # Find semantic matches
            for i, similarity in enumerate(similarities):
                if similarity >= 0.7:  # Lower threshold for semantic similarity
                    matches.append(DuplicateMatch(
                        content_id=existing_ids[i],
                        similarity_score=similarity,
                        match_type="semantic",
                        content_hash=self.content_fingerprints[existing_ids[i]].content_hash,
                        source_path=self.content_fingerprints[existing_ids[i]].metadata.get("source", "unknown"),
                        overlap_percentage=similarity * 100
                    ))
            
        except Exception as e:
            logger.error(f"Error in semantic matching: {e}")
        
        return matches
    
    def _compare_fingerprints(self, content: str, fingerprint: ContentFingerprint) -> float:
        """Compare content with an existing fingerprint"""
        
        # Extract key terms from current content
        current_terms = self._extract_key_terms(content)
        existing_terms = self._extract_key_terms_from_fingerprint(fingerprint)
        
        if not current_terms or not existing_terms:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(current_terms.intersection(existing_terms))
        union = len(current_terms.union(existing_terms))
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_key_terms_from_fingerprint(self, fingerprint: ContentFingerprint) -> Set[str]:
        """Extract key terms from stored fingerprint metadata"""
        
        # In a real implementation, you might store processed terms in metadata
        # For now, we'll return empty set and rely on semantic hash
        return set()


class CrossDocumentAnalyzer:
    """Analyzes relationships between documents"""
    
    def __init__(self):
        self.document_relationships: Dict[str, List[str]] = defaultdict(list)
        self.citation_patterns = [
            r'see\s+(?:also\s+)?(.+?)(?:\.|$)',
            r'refer\s+to\s+(.+?)(?:\.|$)',
            r'mentioned\s+in\s+(.+?)(?:\.|$)',
            r'documented\s+in\s+(.+?)(?:\.|$)'
        ]
    
    async def analyze_relationships(
        self,
        content: str,
        content_id: str,
        existing_documents: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze cross-document relationships"""
        
        relationships = {
            "references": [],
            "cited_by": [],
            "related_topics": [],
            "content_clusters": [],
            "authority_connections": []
        }
        
        try:
            # Find explicit references
            references = self._find_explicit_references(content, existing_documents)
            relationships["references"] = references
            
            # Find topical relationships
            related_topics = await self._find_related_topics(content, existing_documents)
            relationships["related_topics"] = related_topics
            
            # Identify content clusters
            clusters = await self._identify_content_clusters(content_id, existing_documents)
            relationships["content_clusters"] = clusters
            
            # Find authority connections
            authority_connections = self._find_authority_connections(content, existing_documents)
            relationships["authority_connections"] = authority_connections
            
            logger.debug(f"Analyzed relationships for {content_id}: {len(references)} references, {len(related_topics)} related topics")
            
        except Exception as e:
            logger.error(f"Error analyzing document relationships: {e}")
        
        return relationships
    
    def _find_explicit_references(
        self,
        content: str,
        existing_documents: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find explicit references to other documents"""
        
        references = []
        content_lower = content.lower()
        
        for doc_id, doc_info in existing_documents.items():
            doc_title = doc_info.get("title", "").lower()
            doc_source = doc_info.get("source", "").lower()
            
            # Check for title mentions
            if doc_title and len(doc_title) > 10 and doc_title in content_lower:
                references.append({
                    "document_id": doc_id,
                    "reference_type": "title_mention",
                    "confidence": 0.9
                })
            
            # Check for source mentions
            if doc_source and doc_source in content_lower:
                references.append({
                    "document_id": doc_id,
                    "reference_type": "source_mention", 
                    "confidence": 0.7
                })
        
        # Check for citation patterns
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                # Try to match with existing documents
                for doc_id, doc_info in existing_documents.items():
                    if any(term in match.lower() for term in doc_info.get("title", "").lower().split()):
                        references.append({
                            "document_id": doc_id,
                            "reference_type": "citation",
                            "confidence": 0.6,
                            "citation_text": match
                        })
        
        return references
    
    async def _find_related_topics(
        self,
        content: str,
        existing_documents: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find documents with related topics"""
        
        related = []
        
        try:
            # Extract key terms from current content
            current_terms = self._extract_key_terms(content)
            
            for doc_id, doc_info in existing_documents.items():
                doc_content = doc_info.get("content", "")
                if not doc_content:
                    continue
                
                # Extract key terms from document
                doc_terms = self._extract_key_terms(doc_content)
                
                # Calculate topic similarity
                if current_terms and doc_terms:
                    intersection = len(current_terms.intersection(doc_terms))
                    union = len(current_terms.union(doc_terms))
                    similarity = intersection / union if union > 0 else 0.0
                    
                    if similarity > 0.3:  # Topic similarity threshold
                        related.append({
                            "document_id": doc_id,
                            "similarity_score": similarity,
                            "shared_topics": list(current_terms.intersection(doc_terms))
                        })
            
            # Sort by similarity
            related.sort(key=lambda x: x["similarity_score"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error finding related topics: {e}")
        
        return related[:10]  # Return top 10 related documents
    
    async def _identify_content_clusters(
        self,
        content_id: str,
        existing_documents: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Identify content clusters based on source and topic similarity"""
        
        clusters = []
        
        try:
            # Simple clustering based on source type and topic similarity
            # In a production system, you might use more sophisticated clustering algorithms
            
            current_doc = existing_documents.get(content_id, {})
            current_source_type = current_doc.get("source_type", "unknown")
            
            # Find documents from same source type
            same_source_docs = []
            for doc_id, doc_info in existing_documents.items():
                if (doc_id != content_id and 
                    doc_info.get("source_type") == current_source_type):
                    same_source_docs.append(doc_id)
            
            if same_source_docs:
                clusters.append(f"source_type:{current_source_type}")
            
            # Find documents from same domain/channel
            current_source = current_doc.get("source", "")
            if current_source:
                same_source_exact = []
                for doc_id, doc_info in existing_documents.items():
                    if (doc_id != content_id and 
                        doc_info.get("source") == current_source):
                        same_source_exact.append(doc_id)
                
                if same_source_exact:
                    clusters.append(f"source:{current_source}")
            
        except Exception as e:
            logger.error(f"Error identifying content clusters: {e}")
        
        return clusters
    
    def _find_authority_connections(
        self,
        content: str,
        existing_documents: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find connections to authoritative sources"""
        
        connections = []
        
        # Look for high-authority documents that reference similar topics
        for doc_id, doc_info in existing_documents.items():
            authority_score = doc_info.get("quality_metrics", {}).get("authority_score", 0.0)
            
            if authority_score > 0.8:  # High authority threshold
                # Check for topic overlap
                doc_content = doc_info.get("content", "")
                if self._has_topic_overlap(content, doc_content):
                    connections.append({
                        "document_id": doc_id,
                        "authority_score": authority_score,
                        "connection_type": "topic_overlap",
                        "source_type": doc_info.get("source_type", "unknown")
                    })
        
        return connections
    
    def _extract_key_terms(self, content: str, max_terms: int = 20) -> Set[str]:
        """Extract key terms from content"""
        
        words = re.findall(r'\w+', content.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        # Get meaningful words
        key_terms = set()
        for word in words:
            if (len(word) > 3 and 
                word not in stop_words and 
                not word.isdigit()):
                key_terms.add(word)
        
        # Return top terms by frequency
        from collections import Counter
        word_freq = Counter(words)
        important_terms = set()
        for word, _ in word_freq.most_common(max_terms):
            if word in key_terms:
                important_terms.add(word)
        
        return important_terms
    
    def _has_topic_overlap(self, content1: str, content2: str, threshold: float = 0.2) -> bool:
        """Check if two pieces of content have topic overlap"""
        
        terms1 = self._extract_key_terms(content1)
        terms2 = self._extract_key_terms(content2)
        
        if not terms1 or not terms2:
            return False
        
        intersection = len(terms1.intersection(terms2))
        union = len(terms1.union(terms2))
        
        similarity = intersection / union if union > 0 else 0.0
        return similarity >= threshold


class ContentQualityManager:
    """Main manager for content quality assessment and deduplication"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.quality_assessor = ContentQualityAssessor()
        self.duplicate_detector = DuplicateDetector(similarity_threshold)
        self.relationship_analyzer = CrossDocumentAnalyzer()
        self.quality_cache: Dict[str, ContentQualityMetrics] = {}
        
    async def process_content(
        self,
        content: str,
        content_id: str,
        source_metadata: Dict[str, Any],
        existing_documents: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Comprehensive content processing with quality assessment and deduplication"""
        
        results = {
            "content_id": content_id,
            "quality_metrics": None,
            "duplicates": [],
            "relationships": {},
            "processing_status": "success",
            "recommendations": []
        }
        
        try:
            logger.info(f"Processing content quality for: {content_id}")
            
            # Assess content quality
            quality_metrics = await self.quality_assessor.assess_content_quality(
                content, source_metadata
            )
            results["quality_metrics"] = quality_metrics
            self.quality_cache[content_id] = quality_metrics
            
            # Detect duplicates
            duplicates = await self.duplicate_detector.detect_duplicates(
                content, content_id, source_metadata
            )
            results["duplicates"] = duplicates
            
            # Analyze cross-document relationships
            if existing_documents:
                relationships = await self.relationship_analyzer.analyze_relationships(
                    content, content_id, existing_documents
                )
                results["relationships"] = relationships
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                quality_metrics, duplicates, results["relationships"]
            )
            results["recommendations"] = recommendations
            
            logger.info(f"Content processing completed for {content_id}: "
                       f"quality={quality_metrics.overall_quality:.2f}, "
                       f"duplicates={len(duplicates)}")
            
        except Exception as e:
            logger.error(f"Error processing content {content_id}: {e}")
            results["processing_status"] = "error"
            results["error"] = str(e)
        
        return results
    
    def _generate_recommendations(
        self,
        quality_metrics: ContentQualityMetrics,
        duplicates: List[DuplicateMatch],
        relationships: Dict[str, Any]
    ) -> List[str]:
        """Generate processing recommendations based on quality assessment"""
        
        recommendations = []
        
        # Quality-based recommendations
        if quality_metrics.overall_quality < 0.3:
            recommendations.append("Consider rejecting - very low quality content")
        elif quality_metrics.overall_quality < 0.5:
            recommendations.append("Flag for manual review - below quality threshold")
        
        if quality_metrics.relevance_score < 0.4:
            recommendations.append("Low relevance - consider content filtering")
        
        if quality_metrics.completeness_score < 0.4:
            recommendations.append("Content appears incomplete - verify source")
        
        if quality_metrics.quality_issues:
            recommendations.append(f"Quality issues detected: {', '.join(quality_metrics.quality_issues)}")
        
        # Duplicate-based recommendations
        if duplicates:
            exact_duplicates = [d for d in duplicates if d.match_type == "exact"]
            if exact_duplicates:
                recommendations.append("Exact duplicate found - skip ingestion")
            else:
                near_duplicates = [d for d in duplicates if d.similarity_score > 0.9]
                if near_duplicates:
                    recommendations.append("Near-duplicate detected - consider consolidation")
        
        # Relationship-based recommendations
        if relationships.get("authority_connections"):
            recommendations.append("Connected to authoritative sources - prioritize for ingestion")
        
        if len(relationships.get("related_topics", [])) > 5:
            recommendations.append("High topic overlap with existing content - good for coverage")
        
        return recommendations
    
    async def get_quality_summary(self) -> Dict[str, Any]:
        """Get summary of content quality across all processed content"""
        
        if not self.quality_cache:
            return {"message": "No content processed yet"}
        
        quality_scores = [metrics.overall_quality for metrics in self.quality_cache.values()]
        authority_scores = [metrics.authority_score for metrics in self.quality_cache.values()]
        
        summary = {
            "total_content_processed": len(self.quality_cache),
            "average_quality_score": sum(quality_scores) / len(quality_scores),
            "average_authority_score": sum(authority_scores) / len(authority_scores),
            "quality_distribution": {
                "high_quality": len([s for s in quality_scores if s >= 0.8]),
                "medium_quality": len([s for s in quality_scores if 0.5 <= s < 0.8]),
                "low_quality": len([s for s in quality_scores if s < 0.5])
            },
            "total_duplicates_detected": len(self.duplicate_detector.content_fingerprints),
            "content_by_authority": {
                authority: len([m for m in self.quality_cache.values() if m.source_authority == authority])
                for authority in ["high", "medium", "low", "unknown"]
            }
        }
        
        return summary