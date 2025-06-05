import math
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
import re
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from loguru import logger

from .vector_retriever import RetrievalResult
from ...storage.qdrant_client import QdrantManager
from ...config.settings import settings


@dataclass
class BM25Document:
    """Represents a document in the BM25 index"""
    doc_id: str
    content: str
    terms: List[str]
    term_frequencies: Dict[str, int]
    metadata: Dict[str, Any]
    source: str


class BM25Retriever:
    """BM25 (Best Matching 25) lexical retriever for keyword-based search"""
    
    def __init__(self, k1: float = None, b: float = None, index_path: Optional[str] = None):
        """
        Initialize BM25 retriever
        
        Args:
            k1: Controls term frequency saturation (typically 1.2-2.0)
            b: Controls field length normalization (typically 0.75)
            index_path: Path to store/load BM25 index
        """
        self.k1 = k1 if k1 is not None else settings.BM25_K1
        self.b = b if b is not None else settings.BM25_B
        self.qdrant_manager = QdrantManager()
        
        # Index persistence path
        self.index_path = Path(index_path or getattr(settings, 'BM25_INDEX_PATH', './data/bm25_index.pkl'))
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # BM25 index structures
        self.documents: Dict[str, BM25Document] = {}
        self.term_document_frequency: Dict[str, int] = defaultdict(int)
        self.inverted_index: Dict[str, Set[str]] = defaultdict(set)
        self.average_document_length: float = 0.0
        self.total_documents: int = 0
        self.index_checksum: Optional[str] = None
        self.index_timestamp: Optional[datetime] = None
        
        # Stopwords for preprocessing
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'would', 'you', 'your', 'i', 'we',
            'they', 'them', 'this', 'these', 'those', 'what', 'when', 'where',
            'who', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'don',
            'should', 'now'
        }
        
        # Initialize from existing Qdrant collection or load from disk
        asyncio.create_task(self._initialize())
    
    def _preprocess_text(self, text: str, keep_positions: bool = False) -> List[str]:
        """Preprocess text into terms for BM25 indexing
        
        Args:
            text: Text to preprocess
            keep_positions: If True, preserve word positions for phrase/proximity search
        """
        if not text:
            return []
        
        # Convert to lowercase and extract words
        text = text.lower()
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        
        if keep_positions:
            # Keep all words with positions for phrase/proximity search
            return words
        else:
            # Remove stopwords for normal indexing
            terms = [word for word in words if word not in self.stopwords]
            return terms
    
    def _extract_phrases(self, query: str) -> Tuple[List[str], List[str]]:
        """Extract phrase queries and regular terms from query
        
        Returns:
            Tuple of (phrase_queries, regular_terms)
        """
        phrases = []
        regular_terms = []
        
        # Find all quoted phrases
        phrase_pattern = r'"([^"]+)"'
        phrase_matches = re.findall(phrase_pattern, query)
        
        # Remove phrases from query to get regular terms
        remaining_query = re.sub(phrase_pattern, '', query)
        
        # Process phrases
        for phrase in phrase_matches:
            # Keep stopwords in phrases for exact matching
            phrase_terms = self._preprocess_text(phrase, keep_positions=True)
            if phrase_terms:
                phrases.append(phrase_terms)
        
        # Process regular terms (remove stopwords)
        if remaining_query.strip():
            regular_terms = self._preprocess_text(remaining_query, keep_positions=False)
        
        return phrases, regular_terms
    
    def _find_phrase_in_document(self, phrase_terms: List[str], doc: BM25Document) -> int:
        """Find exact phrase occurrences in document
        
        Returns:
            Number of times the phrase appears in the document
        """
        if not phrase_terms or not doc.content:
            return 0
        
        # Get all words in document with positions preserved
        doc_words = self._preprocess_text(doc.content, keep_positions=True)
        
        # Search for phrase
        phrase_count = 0
        phrase_len = len(phrase_terms)
        
        for i in range(len(doc_words) - phrase_len + 1):
            # Check if phrase matches at position i
            if doc_words[i:i + phrase_len] == phrase_terms:
                phrase_count += 1
        
        return phrase_count
    
    def _calculate_proximity_score(self, terms: List[str], doc: BM25Document, max_distance: int = 5) -> float:
        """Calculate proximity score for terms appearing near each other
        
        Args:
            terms: Terms to check proximity for
            doc: Document to search in
            max_distance: Maximum word distance to consider as "near"
            
        Returns:
            Proximity score (0-1)
        """
        if len(terms) < 2 or not doc.content:
            return 0.0
        
        # Get all words in document with positions preserved
        doc_words = self._preprocess_text(doc.content, keep_positions=True)
        
        # Find positions of each term
        term_positions = {}
        for term in terms:
            positions = []
            for i, word in enumerate(doc_words):
                if word == term:
                    positions.append(i)
            if positions:
                term_positions[term] = positions
        
        # If not all terms are in document, no proximity score
        if len(term_positions) < len(terms):
            return 0.0
        
        # Calculate minimum distance between all term pairs
        min_distances = []
        
        for i, term1 in enumerate(terms[:-1]):
            for term2 in terms[i+1:]:
                if term1 in term_positions and term2 in term_positions:
                    # Find minimum distance between any occurrence of term1 and term2
                    min_dist = float('inf')
                    for pos1 in term_positions[term1]:
                        for pos2 in term_positions[term2]:
                            dist = abs(pos2 - pos1) - 1  # Subtract 1 for adjacent words
                            min_dist = min(min_dist, dist)
                    
                    min_distances.append(min_dist)
        
        if not min_distances:
            return 0.0
        
        # Calculate proximity score based on average minimum distance
        avg_min_distance = sum(min_distances) / len(min_distances)
        
        # Score decreases as distance increases
        if avg_min_distance == 0:  # Adjacent words
            proximity_score = 1.0
        elif avg_min_distance <= max_distance:
            proximity_score = 1.0 - (avg_min_distance / max_distance)
        else:
            proximity_score = 0.1  # Small score for words that appear but are far apart
        
        return proximity_score
    
    async def _calculate_collection_checksum(self) -> str:
        """Calculate checksum of the current collection state"""
        try:
            # Get all document IDs and their modification times
            all_points = await self.qdrant_manager.scroll_all_points()
            
            # Create a sorted list of document IDs and content hashes
            doc_signatures = []
            for point in all_points:
                content = point.payload.get('content', '')
                content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
                doc_signatures.append(f"{point.id}:{content_hash}")
            
            doc_signatures.sort()
            combined = ";".join(doc_signatures)
            return hashlib.sha256(combined.encode()).hexdigest()[:16]
            
        except Exception as e:
            logger.error(f"Failed to calculate collection checksum: {e}")
            return "unknown"
    
    async def _initialize(self):
        """Initialize BM25 index from disk or collection"""
        try:
            # Try to load from disk first
            if await self._load_index():
                # Verify the index is still valid
                current_checksum = await self._calculate_collection_checksum()
                if current_checksum == self.index_checksum:
                    logger.info("Loaded valid BM25 index from disk")
                    return
                else:
                    logger.info("Collection has changed, rebuilding BM25 index")
            
            # Build from collection if load failed or index is stale
            await self._initialize_from_collection()
            
        except Exception as e:
            logger.error(f"Failed to initialize BM25 index: {e}")
    
    async def _initialize_from_collection(self):
        """Initialize BM25 index from existing Qdrant collection"""
        try:
            # Get all documents from Qdrant
            all_points = await self.qdrant_manager.scroll_all_points()
            
            if not all_points:
                logger.info("No documents found in collection for BM25 indexing")
                return
            
            logger.info(f"Building BM25 index from {len(all_points)} documents")
            
            # Build BM25 index
            document_lengths = []
            
            for point in all_points:
                payload = point.payload
                content = payload.get('content', '')
                
                if not content.strip():
                    continue
                
                # Create BM25 document
                terms = self._preprocess_text(content)
                term_frequencies = Counter(terms)
                
                doc = BM25Document(
                    doc_id=str(point.id),
                    content=content,
                    terms=terms,
                    term_frequencies=term_frequencies,
                    metadata=payload.get('metadata', {}),
                    source=payload.get('source', 'unknown')
                )
                
                self.documents[doc.doc_id] = doc
                document_lengths.append(len(terms))
                
                # Update inverted index and term document frequencies
                unique_terms = set(terms)
                for term in unique_terms:
                    self.term_document_frequency[term] += 1
                    self.inverted_index[term].add(doc.doc_id)
            
            # Calculate average document length
            self.total_documents = len(self.documents)
            self.average_document_length = sum(document_lengths) / len(document_lengths) if document_lengths else 0
            
            logger.info(f"BM25 index initialized: {self.total_documents} documents, "
                       f"avg length: {self.average_document_length:.1f} terms, "
                       f"vocabulary: {len(self.term_document_frequency)} terms")
            
            # Calculate checksum and save index
            self.index_checksum = await self._calculate_collection_checksum()
            self.index_timestamp = datetime.utcnow()
            await self._save_index()
            
        except Exception as e:
            logger.error(f"Failed to initialize BM25 index: {e}")
    
    def _calculate_bm25_score(self, query_terms: List[str], doc_id: str) -> float:
        """Calculate BM25 score for a document given query terms"""
        if doc_id not in self.documents:
            return 0.0
        
        doc = self.documents[doc_id]
        score = 0.0
        doc_length = len(doc.terms)
        
        for term in query_terms:
            if term not in doc.term_frequencies:
                continue
            
            # Term frequency in document
            tf = doc.term_frequencies[term]
            
            # Document frequency (number of documents containing the term)
            df = self.term_document_frequency.get(term, 0)
            
            if df == 0:
                continue
            
            # Inverse document frequency
            idf = math.log((self.total_documents - df + 0.5) / (df + 0.5))
            
            # BM25 term score
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.average_document_length))
            
            term_score = idf * (numerator / denominator)
            score += term_score
        
        return score
    
    async def search(
        self,
        query: str,
        top_k: int = 50,
        collection_name: Optional[str] = None,
        enable_phrase_search: bool = True,
        enable_proximity_boost: bool = True,
        proximity_distance: int = 5
    ) -> List[RetrievalResult]:
        """Perform BM25 lexical search with phrase and proximity support
        
        Args:
            query: Search query (can include "quoted phrases")
            top_k: Number of results to return
            collection_name: Optional collection name
            enable_phrase_search: Enable exact phrase matching
            enable_proximity_boost: Boost scores for terms appearing near each other
            proximity_distance: Maximum word distance for proximity boost
        """
        
        try:
            if not self.documents:
                logger.warning("BM25 index is empty, reinitializing...")
                await self._initialize_from_collection()
                
                if not self.documents:
                    logger.warning("No documents available for BM25 search")
                    return []
            
            # Extract phrases and regular terms from query
            if enable_phrase_search:
                phrases, query_terms = self._extract_phrases(query)
            else:
                phrases = []
                query_terms = self._preprocess_text(query)
            
            if not query_terms and not phrases:
                logger.warning("No valid terms or phrases in query after preprocessing")
                return []
            
            logger.debug(f"BM25 search - Regular terms: {query_terms}, Phrases: {phrases}")
            
            # Find candidate documents
            candidate_docs = set()
            
            # Add documents containing regular terms
            for term in query_terms:
                if term in self.inverted_index:
                    candidate_docs.update(self.inverted_index[term])
            
            # For phrases, we need to check all documents that contain all words in the phrase
            for phrase_terms in phrases:
                phrase_candidates = None
                for term in phrase_terms:
                    term_lower = term.lower()
                    if term_lower in self.inverted_index:
                        if phrase_candidates is None:
                            phrase_candidates = set(self.inverted_index[term_lower])
                        else:
                            phrase_candidates &= self.inverted_index[term_lower]
                
                if phrase_candidates:
                    candidate_docs.update(phrase_candidates)
            
            if not candidate_docs:
                logger.info("No documents found containing query terms or phrases")
                return []
            
            # Calculate scores for candidate documents
            scored_docs = []
            for doc_id in candidate_docs:
                doc = self.documents[doc_id]
                
                # Calculate base BM25 score for regular terms
                bm25_score = self._calculate_bm25_score(query_terms, doc_id) if query_terms else 0.0
                
                # Add phrase matching boost
                phrase_score = 0.0
                if phrases:
                    for phrase_terms in phrases:
                        phrase_count = self._find_phrase_in_document(phrase_terms, doc)
                        if phrase_count > 0:
                            # Boost score significantly for exact phrase matches
                            phrase_score += phrase_count * 2.0
                
                # Add proximity boost
                proximity_score = 0.0
                if enable_proximity_boost and len(query_terms) > 1:
                    prox = self._calculate_proximity_score(query_terms, doc, proximity_distance)
                    proximity_score = prox * 0.5  # Moderate boost for proximity
                
                # Combine scores
                total_score = bm25_score + phrase_score + proximity_score
                
                if total_score > 0:
                    scored_docs.append((doc_id, total_score, {
                        'bm25': bm25_score,
                        'phrase': phrase_score,
                        'proximity': proximity_score
                    }))
            
            # Sort by total score (descending) and take top_k
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            top_docs = scored_docs[:top_k]
            
            # Convert to RetrievalResult objects
            results = []
            for doc_id, score, score_breakdown in top_docs:
                doc = self.documents[doc_id]
                
                # Add score breakdown to metadata
                enhanced_metadata = {
                    **doc.metadata,
                    'bm25_score_breakdown': score_breakdown
                }
                
                result = RetrievalResult(
                    content=doc.content,
                    score=score,
                    metadata=enhanced_metadata,
                    chunk_id=doc_id,
                    source=doc.source
                )
                results.append(result)
            
            logger.info(f"BM25 search found {len(results)} relevant documents "
                       f"from {len(candidate_docs)} candidates "
                       f"(phrases: {len(phrases)}, proximity boost: {enable_proximity_boost})")
            
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    async def update_document(self, doc_id: str, content: str, metadata: Dict[str, Any], source: str):
        """Update or add a document to the BM25 index"""
        
        try:
            # Remove old document if it exists
            if doc_id in self.documents:
                await self._remove_document(doc_id)
            
            # Add new document
            terms = self._preprocess_text(content)
            term_frequencies = Counter(terms)
            
            doc = BM25Document(
                doc_id=doc_id,
                content=content,
                terms=terms,
                term_frequencies=term_frequencies,
                metadata=metadata,
                source=source
            )
            
            self.documents[doc_id] = doc
            
            # Update inverted index and term document frequencies
            unique_terms = set(terms)
            for term in unique_terms:
                self.term_document_frequency[term] += 1
                self.inverted_index[term].add(doc_id)
            
            # Recalculate average document length
            self.total_documents = len(self.documents)
            if self.total_documents > 0:
                total_length = sum(len(doc.terms) for doc in self.documents.values())
                self.average_document_length = total_length / self.total_documents
            
            logger.debug(f"Updated BM25 index with document {doc_id}")
            
        except Exception as e:
            logger.error(f"Failed to update BM25 document {doc_id}: {e}")
    
    async def _remove_document(self, doc_id: str):
        """Remove a document from the BM25 index"""
        
        if doc_id not in self.documents:
            return
        
        doc = self.documents[doc_id]
        
        # Update term document frequencies and inverted index
        unique_terms = set(doc.terms)
        for term in unique_terms:
            self.term_document_frequency[term] -= 1
            self.inverted_index[term].discard(doc_id)
            
            # Remove term completely if no documents contain it
            if self.term_document_frequency[term] <= 0:
                del self.term_document_frequency[term]
                del self.inverted_index[term]
        
        # Remove document
        del self.documents[doc_id]
        
        logger.debug(f"Removed document {doc_id} from BM25 index")
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the BM25 index"""
        
        return {
            'total_documents': self.total_documents,
            'vocabulary_size': len(self.term_document_frequency),
            'average_document_length': self.average_document_length,
            'top_terms': sorted(
                self.term_document_frequency.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20],  # Top 20 most common terms
            'parameters': {
                'k1': self.k1,
                'b': self.b
            }
        }
    
    async def search_with_filters(
        self,
        query: str,
        metadata_filters: Optional[Dict[str, Any]] = None,
        top_k: int = 50,
        enable_phrase_search: bool = True,
        enable_proximity_boost: bool = True,
        proximity_distance: int = 5
    ) -> List[RetrievalResult]:
        """Perform BM25 search with metadata filtering and advanced features"""
        
        # First get all BM25 results with phrase and proximity support
        all_results = await self.search(
            query, 
            top_k * 2,  # Get more to account for filtering
            enable_phrase_search=enable_phrase_search,
            enable_proximity_boost=enable_proximity_boost,
            proximity_distance=proximity_distance
        )
        
        if not metadata_filters:
            return all_results[:top_k]
        
        # Filter by metadata
        filtered_results = []
        for result in all_results:
            match = True
            for key, value in metadata_filters.items():
                if key not in result.metadata or result.metadata[key] != value:
                    match = False
                    break
            
            if match:
                filtered_results.append(result)
                
                if len(filtered_results) >= top_k:
                    break
        
        logger.info(f"BM25 filtered search returned {len(filtered_results)} results")
        return filtered_results
    
    def get_document_terms(self, doc_id: str) -> List[str]:
        """Get the processed terms for a document"""
        if doc_id in self.documents:
            return self.documents[doc_id].terms
        return []
    
    def explain_score(self, query: str, doc_id: str) -> Dict[str, Any]:
        """Explain the BM25 score calculation for debugging"""
        
        if doc_id not in self.documents:
            return {'error': 'Document not found'}
        
        query_terms = self._preprocess_text(query)
        doc = self.documents[doc_id]
        doc_length = len(doc.terms)
        
        explanation = {
            'query_terms': query_terms,
            'document_length': doc_length,
            'average_document_length': self.average_document_length,
            'total_documents': self.total_documents,
            'term_scores': {},
            'total_score': 0.0
        }
        
        total_score = 0.0
        
        for term in query_terms:
            if term not in doc.term_frequencies:
                explanation['term_scores'][term] = {
                    'tf': 0,
                    'df': 0,
                    'idf': 0,
                    'score': 0
                }
                continue
            
            tf = doc.term_frequencies[term]
            df = self.term_document_frequency.get(term, 0)
            idf = math.log((self.total_documents - df + 0.5) / (df + 0.5))
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.average_document_length))
            term_score = idf * (numerator / denominator)
            
            explanation['term_scores'][term] = {
                'tf': tf,
                'df': df,
                'idf': idf,
                'numerator': numerator,
                'denominator': denominator,
                'score': term_score
            }
            
            total_score += term_score
        
        explanation['total_score'] = total_score
        return explanation
    
    async def _save_index(self) -> bool:
        """Save BM25 index to disk"""
        try:
            # Prepare data for serialization
            index_data = {
                'version': '1.0',
                'checksum': self.index_checksum,
                'timestamp': self.index_timestamp.isoformat() if self.index_timestamp else None,
                'total_documents': self.total_documents,
                'average_document_length': self.average_document_length,
                'k1': self.k1,
                'b': self.b,
                'documents': {},
                'term_document_frequency': dict(self.term_document_frequency),
                'inverted_index': {term: list(docs) for term, docs in self.inverted_index.items()}
            }
            
            # Convert documents to serializable format
            for doc_id, doc in self.documents.items():
                index_data['documents'][doc_id] = {
                    'doc_id': doc.doc_id,
                    'content': doc.content,
                    'terms': doc.terms,
                    'term_frequencies': dict(doc.term_frequencies),
                    'metadata': doc.metadata,
                    'source': doc.source
                }
            
            # Save to disk
            with open(self.index_path, 'wb') as f:
                pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"Saved BM25 index to {self.index_path} ({self.total_documents} documents)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")
            return False
    
    async def _load_index(self) -> bool:
        """Load BM25 index from disk"""
        try:
            if not self.index_path.exists():
                logger.info(f"No BM25 index found at {self.index_path}")
                return False
            
            # Load from disk
            with open(self.index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            # Verify version compatibility
            if index_data.get('version') != '1.0':
                logger.warning(f"Incompatible BM25 index version: {index_data.get('version')}")
                return False
            
            # Restore index state
            self.index_checksum = index_data['checksum']
            self.index_timestamp = datetime.fromisoformat(index_data['timestamp']) if index_data['timestamp'] else None
            self.total_documents = index_data['total_documents']
            self.average_document_length = index_data['average_document_length']
            self.k1 = index_data['k1']
            self.b = index_data['b']
            
            # Restore documents
            self.documents = {}
            for doc_id, doc_data in index_data['documents'].items():
                self.documents[doc_id] = BM25Document(
                    doc_id=doc_data['doc_id'],
                    content=doc_data['content'],
                    terms=doc_data['terms'],
                    term_frequencies=doc_data['term_frequencies'],
                    metadata=doc_data['metadata'],
                    source=doc_data['source']
                )
            
            # Restore indices
            self.term_document_frequency = defaultdict(int, index_data['term_document_frequency'])
            self.inverted_index = defaultdict(set)
            for term, doc_list in index_data['inverted_index'].items():
                self.inverted_index[term] = set(doc_list)
            
            logger.info(f"Loaded BM25 index from {self.index_path} ({self.total_documents} documents)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False
    
    async def force_rebuild(self) -> bool:
        """Force rebuild of BM25 index from collection"""
        try:
            logger.info("Force rebuilding BM25 index...")
            
            # Clear existing index
            self.documents.clear()
            self.term_document_frequency.clear()
            self.inverted_index.clear()
            self.average_document_length = 0.0
            self.total_documents = 0
            
            # Rebuild from collection
            await self._initialize_from_collection()
            
            # Save new index
            self.index_checksum = await self._calculate_collection_checksum()
            self.index_timestamp = datetime.utcnow()
            await self._save_index()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to rebuild BM25 index: {e}")
            return False
    
    async def clear_index(self) -> bool:
        """Clear the BM25 index completely"""
        try:
            logger.info("Clearing BM25 index...")
            
            # Clear all in-memory structures
            self.documents = {}
            self.inverted_index = defaultdict(set)
            self.term_document_frequency = defaultdict(int)
            self.total_documents = 0
            self.average_document_length = 0.0
            self.index_checksum = None
            self.index_timestamp = None
            
            # Remove persisted index file if it exists
            if self.index_path.exists():
                logger.info(f"Removing persisted index at {self.index_path}")
                self.index_path.unlink()
            
            logger.info("BM25 index cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear BM25 index: {e}")
            return False