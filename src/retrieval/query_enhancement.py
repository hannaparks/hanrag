import re
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from ..generation.claude_client import ClaudeClient
from ..config.settings import settings


class QueryType(Enum):
    """Types of queries for routing and optimization"""
    SIMPLE_FACTUAL = "simple_factual"
    COMPLEX_ANALYTICAL = "complex_analytical"
    MULTI_FACETED = "multi_faceted"
    PROCEDURAL = "procedural"
    COMPARISON = "comparison"
    TEMPORAL = "temporal"


@dataclass
class EnhancedQuery:
    """Enhanced query with optimization metadata"""
    original_query: str
    enhanced_query: str
    query_type: QueryType
    sub_queries: List[str]
    keywords: List[str]
    suggested_params: Dict[str, Any]
    confidence: float


class QueryEnhancer:
    """Query enhancement and routing system"""
    
    def __init__(self):
        self.claude_client = ClaudeClient()
        
        # Query type patterns
        self.query_patterns = {
            QueryType.SIMPLE_FACTUAL: [
                r'\bwhat is\b',
                r'\bwho is\b',
                r'\bwhen was\b',
                r'\bwhere is\b',
                r'\bdefine\b',
                r'\bexplain\b'
            ],
            QueryType.COMPLEX_ANALYTICAL: [
                r'\banalyze\b',
                r'\bcompare and contrast\b',
                r'\bwhy does\b',
                r'\bhow does.*work\b',
                r'\bwhat are the implications\b',
                r'\bwhat factors\b'
            ],
            QueryType.MULTI_FACETED: [
                r'\band\b.*\band\b',
                r'\bor\b.*\bor\b',
                r'\bmultiple\b',
                r'\bvarious\b',
                r'\ball.*aspects\b'
            ],
            QueryType.PROCEDURAL: [
                r'\bhow to\b',
                r'\bsteps to\b',
                r'\bprocess for\b',
                r'\binstructions\b',
                r'\bguide\b'
            ],
            QueryType.COMPARISON: [
                r'\bvs\b',
                r'\bversus\b',
                r'\bcompare\b',
                r'\bdifference between\b',
                r'\bbetter than\b',
                r'\bsimilar to\b'
            ],
            QueryType.TEMPORAL: [
                r'\brecent\b',
                r'\blatest\b',
                r'\bupdated\b',
                r'\bcurrent\b',
                r'\btoday\b',
                r'\byesterday\b',
                r'\blast week\b',
                r'\bin \d+\b'
            ]
        }
    
    async def enhance_query(self, query: str) -> EnhancedQuery:
        """Enhance a query with type detection and optimization"""
        
        try:
            # Detect query type
            query_type = self._detect_query_type(query)
            
            # Extract keywords
            keywords = self._extract_keywords(query)
            
            # Generate sub-queries if needed
            sub_queries = await self._generate_sub_queries(query, query_type)
            
            # Enhance the query
            enhanced_query = await self._enhance_query_text(query, query_type)
            
            # Get suggested parameters
            suggested_params = self._get_suggested_params(query_type, query)
            
            result = EnhancedQuery(
                original_query=query,
                enhanced_query=enhanced_query,
                query_type=query_type,
                sub_queries=sub_queries,
                keywords=keywords,
                suggested_params=suggested_params,
                confidence=0.8  # Default confidence
            )
            
            logger.info(f"Enhanced query: {query_type.value} with {len(sub_queries)} sub-queries")
            return result
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")
            # Return minimal enhancement
            return EnhancedQuery(
                original_query=query,
                enhanced_query=query,
                query_type=QueryType.SIMPLE_FACTUAL,
                sub_queries=[],
                keywords=self._extract_keywords(query),
                suggested_params={"top_k": 25, "rerank_top_n": 10},
                confidence=0.5
            )
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of query based on patterns"""
        
        query_lower = query.lower()
        
        # Score each query type
        type_scores = {}
        for query_type, patterns in self.query_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches
            type_scores[query_type] = score
        
        # Return the type with highest score
        best_type = max(type_scores, key=type_scores.get)
        
        # If no clear pattern match, use heuristics
        if type_scores[best_type] == 0:
            if len(query.split()) <= 5:
                return QueryType.SIMPLE_FACTUAL
            elif '?' in query:
                return QueryType.COMPLEX_ANALYTICAL
            else:
                return QueryType.SIMPLE_FACTUAL
        
        return best_type
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'could', 'can', 'may', 'might', 'must', 'what', 'when', 'where', 'why',
            'how', 'who', 'which', 'that', 'this', 'these', 'those'
        }
        
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for word in keywords:
            if word not in seen:
                seen.add(word)
                unique_keywords.append(word)
        
        return unique_keywords[:10]  # Limit to top 10 keywords
    
    async def _generate_sub_queries(self, query: str, query_type: QueryType) -> List[str]:
        """Generate sub-queries for complex queries"""
        
        # Only generate sub-queries for complex types
        if query_type not in [QueryType.COMPLEX_ANALYTICAL, QueryType.MULTI_FACETED, QueryType.COMPARISON]:
            return []
        
        try:
            prompt = f"""Break down this complex question into 2-4 simpler sub-questions that would help answer the main question comprehensively:

Main Question: {query}

Generate specific sub-questions that:
1. Focus on different aspects of the main question
2. Can be answered independently
3. Together provide a complete answer to the main question

Format: Return only the sub-questions, one per line, without numbering."""

            response = await self.claude_client.generate_simple_response(prompt)
            
            # Parse sub-queries
            sub_queries = []
            for line in response.split('\n'):
                line = line.strip()
                if line and not line.startswith('Sub-question') and '?' in line:
                    sub_queries.append(line)
            
            return sub_queries[:4]  # Limit to 4 sub-queries
            
        except Exception as e:
            logger.error(f"Sub-query generation failed: {e}")
            return []
    
    async def _enhance_query_text(self, query: str, query_type: QueryType) -> str:
        """Enhance the query text for better retrieval"""
        
        try:
            enhancement_prompts = {
                QueryType.SIMPLE_FACTUAL: f"Rephrase this question to be more specific and include relevant keywords: {query}",
                QueryType.COMPLEX_ANALYTICAL: f"Expand this analytical question to include related concepts and specific aspects to analyze: {query}",
                QueryType.PROCEDURAL: f"Rephrase this how-to question to be more specific about the process and expected outcomes: {query}",
                QueryType.COMPARISON: f"Expand this comparison question to specify the criteria and aspects to compare: {query}",
                QueryType.TEMPORAL: f"Clarify this time-related question and specify the relevant time period: {query}"
            }
            
            prompt = enhancement_prompts.get(query_type, f"Improve this question for better search results: {query}")
            prompt += "\n\nReturn only the enhanced question, no explanation."
            
            enhanced = await self.claude_client.generate_simple_response(prompt)
            
            # Clean up the response
            enhanced = enhanced.strip().strip('"\'')
            
            # If enhancement failed or is too similar, return original
            if not enhanced or len(enhanced) < len(query) * 0.5:
                return query
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Query text enhancement failed: {e}")
            return query
    
    def _get_suggested_params(self, query_type: QueryType, query: str) -> Dict[str, Any]:
        """Get suggested retrieval parameters based on query type"""
        
        base_params = {
            "top_k": settings.SIMILARITY_TOP_K,
            "rerank_top_n": settings.RERANK_TOP_N,
            "mmr": False,
            "temperature": settings.TEMPERATURE
        }
        
        # Adjust parameters based on query type
        type_adjustments = {
            QueryType.SIMPLE_FACTUAL: {
                "top_k": 25,
                "rerank_top_n": 10,
                "mmr": False,
                "temperature": 0.0
            },
            QueryType.COMPLEX_ANALYTICAL: {
                "top_k": 50,
                "rerank_top_n": 25,
                "mmr": True,
                "temperature": 0.1
            },
            QueryType.MULTI_FACETED: {
                "top_k": 75,
                "rerank_top_n": 35,
                "mmr": True,
                "temperature": 0.2
            },
            QueryType.PROCEDURAL: {
                "top_k": 30,
                "rerank_top_n": 15,
                "mmr": False,
                "temperature": 0.0
            },
            QueryType.COMPARISON: {
                "top_k": 60,
                "rerank_top_n": 30,
                "mmr": True,
                "temperature": 0.1
            },
            QueryType.TEMPORAL: {
                "top_k": 40,
                "rerank_top_n": 20,
                "mmr": False,
                "temperature": 0.0
            }
        }
        
        # Apply adjustments
        if query_type in type_adjustments:
            base_params.update(type_adjustments[query_type])
        
        # Additional adjustments based on query length
        query_words = len(query.split())
        if query_words > 20:  # Long query
            base_params["top_k"] = min(base_params["top_k"] * 1.5, 100)
            base_params["mmr"] = True
        elif query_words < 5:  # Short query
            base_params["top_k"] = max(base_params["top_k"] * 0.7, 15)
        
        return base_params
    
    async def expand_query_with_synonyms(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms"""
        
        try:
            prompt = f"""For the following search query, generate 3-5 alternative phrasings using synonyms and related terms:

Original query: {query}

Generate variations that:
1. Use synonyms for key terms
2. Rephrase the question structure
3. Include related concepts
4. Maintain the same intent

Return only the alternative queries, one per line."""

            response = await self.claude_client.generate_simple_response(prompt)
            
            variations = []
            for line in response.split('\n'):
                line = line.strip()
                if line and line != query:
                    variations.append(line)
            
            return [query] + variations[:4]  # Original + up to 4 variations
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return [query]
    
    def get_retrieval_strategy(self, enhanced_query: EnhancedQuery) -> Dict[str, Any]:
        """Get the optimal retrieval strategy for an enhanced query"""
        
        strategy = {
            "use_mmr": enhanced_query.suggested_params.get("mmr", False),
            "top_k": enhanced_query.suggested_params.get("top_k", 50),
            "rerank_top_n": enhanced_query.suggested_params.get("rerank_top_n", 25),
            "use_sub_queries": len(enhanced_query.sub_queries) > 0,
            "multi_hop": enhanced_query.query_type in [QueryType.COMPLEX_ANALYTICAL, QueryType.MULTI_FACETED],
            "temperature": enhanced_query.suggested_params.get("temperature", 0.1)
        }
        
        return strategy