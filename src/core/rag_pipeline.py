import aiohttp
from typing import Dict, Any, List, Optional, AsyncIterator
from loguru import logger
from datetime import datetime
import uuid

from ..storage.qdrant_client import QdrantManager
from ..generation.claude_client import ClaudeClient
from ..retrieval.retrievers.vector_retriever import VectorRetriever
from ..retrieval.retrievers.hybrid_retriever import HybridRetriever, HybridSearchConfig
from ..retrieval.query_enhancement import QueryEnhancer
from ..ingestion.parsers.document_parser import DocumentParser
from ..ingestion.chunking.text_chunker import TextChunker
from ..ingestion.channel_processor import ChannelMessageProcessor
from ..mattermost.channel_client import MattermostClient
from ..utils.source_manager import SourceManager
from ..config.settings import settings
from ..cache import RedisCache, get_cache
from ..monitoring.metrics import track_query, track_ingestion
from ..api.models import ConversationHistory


class RAGPipeline:
    """Main RAG pipeline orchestrating ingestion, retrieval, and generation"""
    
    def __init__(self, use_hybrid_search: bool = True):
        self.qdrant_manager = QdrantManager()
        self.claude_client = ClaudeClient()
        self.vector_retriever = VectorRetriever()
        self.query_enhancer = QueryEnhancer()
        self.document_parser = DocumentParser()
        self.text_chunker = TextChunker()
        self.source_manager = SourceManager()
        
        # Channel processing components
        self.channel_processor = ChannelMessageProcessor()
        self.mattermost_client = MattermostClient()
        
        # Cache instance (will be set during initialization)
        self.cache: Optional[RedisCache] = None
        
        # Hybrid search configuration
        self.use_hybrid_search = use_hybrid_search
        if use_hybrid_search:
            self.hybrid_retriever = HybridRetriever(
                config=HybridSearchConfig(
                    vector_weight=settings.HYBRID_VECTOR_WEIGHT,
                    bm25_weight=settings.HYBRID_BM25_WEIGHT,
                    rrf_k=settings.HYBRID_RRF_K,
                    normalize_scores=True
                )
            )
        else:
            self.hybrid_retriever = None
        
        # Conversation history storage (in-memory for now, could be Redis/DB in production)
        self.conversations: Dict[str, ConversationHistory] = {}
        
        # Initialize components
        self._initialized = False
    
    async def initialize(self):
        """Initialize the RAG pipeline components"""
        if self._initialized:
            return
        
        try:
            # Ensure Qdrant collection exists
            await self.qdrant_manager.create_collection()
            
            # Initialize cache
            self.cache = await get_cache()
            if self.cache:
                logger.info("Redis cache initialized successfully")
            else:
                logger.info("Running without Redis cache")
            
            logger.info("RAG pipeline initialized successfully")
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    
    async def ingest_url(self, url: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Ingest content from a URL using the advanced ingestion pipeline"""
        
        try:
            logger.info(f"Starting URL ingestion: {url}")
            
            # Use the new ingestion pipeline for URL processing
            from ..ingestion.pipeline import IngestionPipeline, IngestionConfig
            
            # Create pipeline instance with appropriate config
            config = IngestionConfig(
                chunk_size=512,
                chunk_overlap=64,
                max_concurrent_tasks=1,
                retry_attempts=3,
                request_timeout=60
            )
            pipeline = IngestionPipeline(config)
            
            # Start URL ingestion
            task_id = await pipeline.ingest_url(url)
            
            # Wait for completion with timeout
            task = await pipeline.wait_for_task(task_id, timeout=120)
            
            if task.status.value == "completed":
                logger.info(f"URL ingestion completed successfully: {task_id}")
                return {
                    "success": True,
                    "document_id": task.result.get("document_id") if task.result else task_id,
                    "chunks_created": task.result.get("chunks_created", 0) if task.result else 0,
                    "task_id": task_id,
                    "processing_time": task.duration_seconds,
                    "metadata": task.metadata
                }
            else:
                error_msg = task.error_message or "Unknown error during ingestion"
                logger.error(f"URL ingestion failed: {error_msg}")
                return {"success": False, "error": error_msg}
            
        except Exception as e:
            logger.error(f"URL ingestion failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def ingest_channel_enhanced(
        self,
        channel_id: str,
        team_id: str,
        max_messages: int = 1000,
        include_user_profiles: bool = True,
        incremental: bool = False,
        since_timestamp: Optional[int] = None
    ) -> Dict[str, Any]:
        """Enhanced channel ingestion with complete metadata and conversation analysis"""
        
        try:
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"Starting enhanced channel ingestion for {channel_id}")
            
            # Use the advanced channel processor
            if incremental and since_timestamp:
                result = await self.channel_processor.process_channel_incremental(
                    channel_id=channel_id,
                    team_id=team_id,
                    since_timestamp=since_timestamp,
                    max_messages=max_messages
                )
            else:
                result = await self.channel_processor.process_channel_complete(
                    channel_id=channel_id,
                    team_id=team_id,
                    max_messages=max_messages,
                    include_user_profiles=include_user_profiles
                )
            
            if result.get("success"):
                logger.info(f"Enhanced channel ingestion completed successfully")
                return {
                    "success": True,
                    "method": "enhanced_processor",
                    "channel_id": channel_id,
                    "document_id": result.get("document_id"),
                    "metrics": result.get("metrics"),
                    "channel_context": result.get("channel_context"),
                    "enhanced_metadata": result.get("enhanced_metadata")
                }
            else:
                logger.error(f"Enhanced channel ingestion failed: {result.get('error')}")
                return {"success": False, "error": result.get("error")}
                
        except Exception as e:
            logger.error(f"Enhanced channel ingestion failed: {e}")
            return {"success": False, "error": str(e)}
    
    @track_ingestion("mattermost_channel")
    async def ingest_channel_messages(
        self, 
        messages: List[Dict[str, Any]], 
        channel_info: Dict[str, Any], 
        team_info: Dict[str, Any], 
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Ingest Mattermost channel messages into the knowledge base"""
        
        try:
            logger.info(f"Starting channel message ingestion: {len(messages)} messages from channel {channel_info.get('display_name', 'Unknown')}")
            
            if not messages:
                return {"success": False, "error": "No messages to ingest"}
            
            # Sort messages by creation time (oldest first for chronological context)
            sorted_messages = sorted(messages, key=lambda m: m.get("create_at", 0))
            
            # Group messages into conversation threads for better context
            conversation_groups = self._group_messages_by_context(sorted_messages)
            
            total_chunks = 0
            processed_groups = 0
            
            for group_id, message_group in conversation_groups.items():
                try:
                    # Convert message group to text content
                    conversation_text = self._format_messages_as_text(message_group, channel_info, team_info)
                    
                    if len(conversation_text.strip()) < 50:  # Skip very short conversations
                        continue
                    
                    # Create document ID for this conversation group
                    document_id = f"channel_{channel_info.get('id', 'unknown')}_{group_id}"
                    
                    # Chunk the conversation
                    chunks, parent_chunks = self.text_chunker.hierarchical_chunk(
                        text=conversation_text,
                        document_id=document_id
                    )
                    
                    # Enhance metadata with Mattermost-specific information
                    enhanced_metadata = {
                        **(metadata or {}),
                        "source_type": "mattermost_channel",
                        "channel_id": channel_info.get("id"),
                        "channel_name": channel_info.get("display_name"),
                        "team_id": team_info.get("id"),
                        "team_name": team_info.get("display_name"),
                        "message_count": len(message_group),
                        "conversation_group": group_id,
                        "first_message_time": message_group[0].get("create_at"),
                        "last_message_time": message_group[-1].get("create_at"),
                        "participants": list(set(msg.get("user_id") for msg in message_group if msg.get("user_id")))
                    }
                    
                    # Store chunks with enhanced metadata
                    source_identifier = f"mattermost://{team_info.get('name', 'unknown')}/{channel_info.get('name', 'unknown')}"
                    await self._store_chunks(chunks, parent_chunks, source_identifier, enhanced_metadata)
                    
                    total_chunks += len(chunks)
                    processed_groups += 1
                    
                    logger.debug(f"Processed conversation group {group_id}: {len(chunks)} chunks")
                    
                except Exception as e:
                    logger.error(f"Failed to process conversation group {group_id}: {e}")
                    continue
            
            logger.info(f"Successfully ingested channel messages: {processed_groups} conversation groups, {total_chunks} total chunks")
            
            return {
                "success": True,
                "chunks_processed": total_chunks,
                "conversation_groups": processed_groups,
                "messages_processed": len(messages),
                "channel_id": channel_info.get("id"),
                "channel_name": channel_info.get("display_name")
            }
            
        except Exception as e:
            logger.error(f"Channel message ingestion failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _group_messages_by_context(self, messages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group messages into conversation contexts for better chunking"""
        
        groups = {}
        current_group = []
        current_group_id = f"group_{uuid.uuid4().hex[:8]}"
        last_message_time = 0
        
        # Configuration for grouping logic
        MAX_TIME_GAP_MINUTES = 30  # Max gap between messages in same group
        MAX_GROUP_SIZE = 20        # Max messages per group
        
        for message in messages:
            message_time = message.get("create_at", 0)
            time_gap_minutes = (message_time - last_message_time) / (1000 * 60)  # Convert ms to minutes
            
            # Start new group if:
            # 1. Time gap is too large
            # 2. Current group is too big
            # 3. Message is a thread reply to a different root
            if (current_group and 
                (time_gap_minutes > MAX_TIME_GAP_MINUTES or 
                 len(current_group) >= MAX_GROUP_SIZE)):
                
                groups[current_group_id] = current_group
                current_group = []
                current_group_id = f"group_{uuid.uuid4().hex[:8]}"
            
            current_group.append(message)
            last_message_time = message_time
        
        # Add the last group
        if current_group:
            groups[current_group_id] = current_group
        
        return groups
    
    def _format_messages_as_text(
        self, 
        messages: List[Dict[str, Any]], 
        channel_info: Dict[str, Any], 
        team_info: Dict[str, Any]
    ) -> str:
        """Format a group of messages as readable text for embedding"""
        
        lines = []
        
        # Add context header
        channel_name = channel_info.get("display_name", "Unknown Channel")
        team_name = team_info.get("display_name", "Unknown Team")
        lines.append(f"## Conversation in #{channel_name} ({team_name})")
        lines.append("")
        
        # Add channel purpose if available
        if channel_info.get("purpose"):
            lines.append(f"**Channel Purpose:** {channel_info['purpose']}")
            lines.append("")
        
        # Format each message
        for message in messages:
            message_content = message.get("message", "").strip()
            if not message_content:
                continue
            
            # Convert timestamp to readable format
            timestamp = message.get("create_at", 0)
            if timestamp:
                dt = datetime.fromtimestamp(timestamp / 1000)  # Convert ms to seconds
                time_str = dt.strftime("%Y-%m-%d %H:%M")
            else:
                time_str = "Unknown time"
            
            user_id = message.get("user_id", "unknown_user")
            
            # Handle threaded messages
            if message.get("root_id") and message.get("root_id") != message.get("id"):
                lines.append(f"  ↳ **Reply by {user_id}** ({time_str}): {message_content}")
            else:
                lines.append(f"**{user_id}** ({time_str}): {message_content}")
            
            # Add hashtags if present
            if message.get("hashtags"):
                lines.append(f"  Tags: {message['hashtags']}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    async def _fetch_url_content(self, url: str) -> Optional[str]:
        """Fetch content from URL with basic text/binary handling (legacy method)"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status == 200:
                        content_type = response.content_type.lower() if response.content_type else ""
                        
                        # For binary files, delegate to new ingestion pipeline
                        if any(binary_type in content_type for binary_type in [
                            'application/pdf', 'application/octet-stream'
                        ]) or url.lower().endswith(('.pdf', '.doc', '.docx', '.xls', '.xlsx')):
                            logger.warning(f"Binary content detected - recommend using ingest_url() method instead")
                            return None
                        
                        # Handle text content with encoding fallback
                        try:
                            content = await response.text()
                            logger.debug(f"Fetched {len(content)} characters from {url}")
                            return content
                        except UnicodeDecodeError as e:
                            logger.warning(f"UTF-8 decode failed, trying with latin-1: {e}")
                            # Fallback to latin-1 encoding
                            raw_content = await response.read()
                            content = raw_content.decode('latin-1', errors='ignore')
                            logger.debug(f"Fetched {len(content)} characters from {url} using latin-1")
                            return content
                    else:
                        logger.error(f"HTTP {response.status} when fetching {url}")
                        return None
        except Exception as e:
            logger.error(f"Failed to fetch URL {url}: {e}")
            return None
    
    async def _store_chunks(self, chunks: List, parent_chunks: List, source: str, metadata: Dict[str, Any]):
        """Store hierarchical chunks in vector database"""
        
        try:
            # Prepare all chunks for embedding (children + parents)
            all_chunks = chunks + parent_chunks
            chunk_texts = [chunk.content for chunk in all_chunks]
            
            # Get embeddings for all chunks
            embeddings = await self.vector_retriever._get_embeddings(chunk_texts)
            
            # Prepare points for Qdrant
            points = []
            
            # Store child chunks first
            for i, chunk in enumerate(chunks):
                point = {
                    "id": chunk.chunk_id,
                    "vector": embeddings[i],
                    "payload": {
                        "content": chunk.content,
                        "source": source,
                        "document_id": chunk.metadata.get("document_id"),
                        "chunk_type": chunk.metadata.get("chunk_type", "child"),
                        "parent_id": chunk.parent_id,
                        "start_index": chunk.start_index,
                        "end_index": chunk.end_index,
                        "token_count": chunk.metadata.get("token_count", 0),
                        "char_count": chunk.metadata.get("char_count", len(chunk.content)),
                        "hierarchy_level": chunk.metadata.get("hierarchy_level", 1),
                        "metadata": {**chunk.metadata, **metadata},
                        "timestamp": datetime.now().isoformat()
                    }
                }
                points.append(point)
            
            # Store parent chunks
            for i, parent_chunk in enumerate(parent_chunks):
                parent_point = {
                    "id": parent_chunk.chunk_id,
                    "vector": embeddings[len(chunks) + i],  # Parent embeddings start after child embeddings
                    "payload": {
                        "content": parent_chunk.content,
                        "source": source,
                        "document_id": parent_chunk.metadata.get("document_id"),
                        "chunk_type": parent_chunk.metadata.get("chunk_type", "parent"),
                        "parent_id": None,  # Parents don't have parents
                        "start_index": parent_chunk.start_index,
                        "end_index": parent_chunk.end_index,
                        "token_count": parent_chunk.metadata.get("token_count", 0),
                        "char_count": parent_chunk.metadata.get("char_count", len(parent_chunk.content)),
                        "hierarchy_level": parent_chunk.metadata.get("hierarchy_level", 0),
                        "metadata": {**parent_chunk.metadata, **metadata},
                        "timestamp": datetime.now().isoformat()
                    }
                }
                points.append(parent_point)
            
            # Store all points in Qdrant
            await self.qdrant_manager.upsert_points(points)
            
            logger.info(f"Stored {len(chunks)} child chunks and {len(parent_chunks)} parent chunks")
            
        except Exception as e:
            logger.error(f"Failed to store hierarchical chunks: {e}")
            raise
    
    async def _assemble_hierarchical_context(self, query_vector: List[float], max_results: int = 50) -> List[str]:
        """Assemble context using hierarchical chunk relationships"""
        
        try:
            # Get hierarchical search results (both children and parents)
            search_results = await self.qdrant_manager.search_with_parent_context(
                query_vector=query_vector,
                top_k=max_results,
                include_parents=True
            )
            
            children = search_results.get("children", [])
            parents = search_results.get("parents", [])
            
            # Create a map of parent content for quick lookup
            parent_map = {parent.id: parent.payload.get("content", "") for parent in parents}
            
            # Assemble context with hierarchical relationships
            context_texts = []
            processed_parents = set()
            
            for child in children[:10]:  # Top 10 child results
                child_content = child.payload.get("content", "")
                parent_id = child.payload.get("parent_id")
                
                if parent_id and parent_id in parent_map and parent_id not in processed_parents:
                    # Use parent content with child highlighted
                    parent_content = parent_map[parent_id]
                    
                    # Highlight the child content within parent
                    if child_content in parent_content:
                        highlighted_content = parent_content.replace(
                            child_content,
                            f"**[RELEVANT: {child_content}]**"
                        )
                        context_texts.append(highlighted_content)
                    else:
                        # Fallback: parent + child
                        context_texts.append(f"{parent_content}\n\n--- FOCUS ---\n{child_content}")
                    
                    processed_parents.add(parent_id)
                else:
                    # No parent or already processed, use child content
                    context_texts.append(child_content)
            
            return context_texts
            
        except Exception as e:
            logger.error(f"Failed to assemble hierarchical context: {e}")
            # Fallback to simple vector search
            results = await self.qdrant_manager.search_vectors(
                query_vector=query_vector,
                top_k=max_results
            )
            return [result.payload.get("content", "") for result in results[:10]]
    
    @track_query("hybrid")
    async def query(
        self, 
        query: str, 
        max_results: int = 50,
        use_mmr: bool = True,
        temperature: float = None,
        use_hybrid: Optional[bool] = None,
        use_cache: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        use_streaming: bool = None
    ) -> Dict[str, Any]:
        """Query the knowledge base with optional hybrid search, caching, and filtering
        
        Args:
            query: The search query
            max_results: Maximum number of results to retrieve
            use_mmr: Whether to use MMR for diversity
            temperature: Temperature for generation (overrides default)
            use_hybrid: Whether to use hybrid search (overrides default)
            use_cache: Whether to use cache
            filters: Optional filters to apply (e.g., {"channel_id": "123", "date_range": {"start": "2024-01-01"}})
            use_streaming: Whether to use streaming for generation (auto-detected if None)
        """
        
        try:
            if not self._initialized:
                await self.initialize()
            
            # Check cache first
            if use_cache and self.cache:
                cache_params = {
                    "max_results": max_results,
                    "use_mmr": use_mmr,
                    "temperature": temperature,
                    "use_hybrid": use_hybrid,
                    "filters": filters
                }
                
                cached_result = await self.cache.get(query, cache_params)
                if cached_result:
                    logger.info(f"Returning cached result for query: {query[:50]}...")
                    return cached_result
            
            # Enhance query
            enhanced_query = await self.query_enhancer.enhance_query(query)
            
            # Determine search method
            use_hybrid_for_query = use_hybrid if use_hybrid is not None else self.use_hybrid_search
            
            # Get search results
            if use_hybrid_for_query and self.hybrid_retriever:
                logger.info("Using hybrid search (vector + BM25)")
                
                if use_mmr:
                    # Hybrid search with MMR diversification
                    results = await self.hybrid_retriever.search_with_mmr(
                        query=enhanced_query.enhanced_query,
                        top_k=max_results,
                        mmr_lambda=enhanced_query.suggested_params.get("mmr_lambda", 0.7),
                        filters=filters
                    )
                else:
                    # Standard hybrid search
                    results = await self.hybrid_retriever.search(
                        query=enhanced_query.enhanced_query,
                        top_k=max_results,
                        filters=filters
                    )
                
                search_method = "hybrid_mmr" if use_mmr else "hybrid"
            else:
                logger.info("Using vector search only")
                
                if use_mmr:
                    # Vector search with MMR
                    results = await self.vector_retriever.mmr_search(
                        query=enhanced_query.enhanced_query,
                        top_k=max_results,
                        mmr_lambda=enhanced_query.suggested_params.get("mmr_lambda", 0.7)
                    )
                else:
                    # Standard vector search
                    results = await self.vector_retriever.similarity_search(
                        query=enhanced_query.enhanced_query,
                        top_k=max_results,
                        filters=filters
                    )
                
                search_method = "vector_mmr" if use_mmr else "vector"
            
            if not results:
                return {
                    "response": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "context_count": 0,
                    "query_type": enhanced_query.query_type.value,
                    "search_method": search_method
                }
            
            # Extract context from results
            context_texts = [result.content for result in results[:10]]  # Top 10 for context
            
            # Process sources with enhanced deduplication and ranking
            processed_sources, citation_mapping = self.source_manager.process_retrieval_results(
                results, 
                max_sources=5,
                min_relevance_threshold=0.05  # Lowered threshold for typical retrieval scores
            )
            
            # Generate context and citation instructions for Claude
            context_text, citation_guide = self.source_manager.generate_source_context_for_llm(
                processed_sources, 
                citation_mapping
            )
            
            # Build source metadata for the Claude prompt
            source_metadata = []
            for source in processed_sources:
                metadata_item = {
                    "citation_key": source.citation_key,
                    "title": source.source_info.title,
                    "source_type": source.source_info.source_type,
                    "relevance_score": source.combined_relevance,
                    "authority_score": source.source_info.authority_score,
                    "chunk_count": source.source_info.chunk_count,
                    "timestamp": source.source_info.timestamp.isoformat() if source.source_info.timestamp else None
                }
                source_metadata.append(metadata_item)
            
            # Auto-detect if streaming should be used for long operations
            if use_streaming is None:
                # Enable streaming for queries that might be complex/long
                context_length = len(context_text)
                query_length = len(query)
                total_sources = len(processed_sources)
                
                # Use streaming if context is large, many sources, or complex query
                use_streaming = (
                    context_length > 8000 or  # Large context
                    total_sources > 3 or      # Many sources to cite
                    query_length > 200 or     # Complex query
                    any(keyword in query.lower() for keyword in ['analyze', 'compare', 'summarize', 'categorize', 'list all', 'explain'])
                )
                
                if use_streaming:
                    logger.info(f"Auto-enabling streaming: context_len={context_length}, sources={total_sources}, query_len={query_length}")
            
            # Generate response with enhanced quality assessment
            try:
                if use_streaming:
                    logger.info("Using streaming generation for potentially long response")
                    # Use streaming API to avoid 10-minute timeout
                    prompt = self.claude_client._build_rag_prompt([context_text], query, citation_guide)
                    
                    stream = await self.claude_client.client.messages.create(
                        model=self.claude_client.model,
                        max_tokens=self.claude_client.max_tokens,
                        temperature=temperature or enhanced_query.suggested_params.get("temperature", 0.1),
                        messages=[{"role": "user", "content": prompt}],
                        stream=True
                    )
                    
                    # Collect streaming response
                    full_response = ""
                    async for event in stream:
                        if event.type == "content_block_delta" and hasattr(event.delta, 'text'):
                            full_response += event.delta.text
                        elif event.type == "message_stop":
                            break
                    
                    response_data = {
                        "response": full_response,
                        "model": self.claude_client.model,
                        "usage": {"estimated_tokens": len(full_response.split())},
                        "sources": self.claude_client._extract_citations(full_response)
                    }
                else:
                    # Use regular generation
                    response_data = await self.claude_client.generate_response(
                        context=[context_text],  # Use the processed context
                        query=query,
                        temperature=temperature or enhanced_query.suggested_params.get("temperature", 0.1),
                        source_metadata=source_metadata,
                        citation_guide=citation_guide
                    )
            except Exception as e:
                if use_streaming and "streaming" in str(e).lower():
                    logger.warning(f"Streaming failed, falling back to regular generation: {e}")
                    # Fallback to regular generation
                    response_data = await self.claude_client.generate_response(
                        context=[context_text],
                        query=query,
                        temperature=temperature or enhanced_query.suggested_params.get("temperature", 0.1),
                        source_metadata=source_metadata,
                        citation_guide=citation_guide
                    )
                else:
                    raise
            
            # Validate citations in response
            citation_validation = self.source_manager.validate_citations_in_response(
                response_data["response"], 
                processed_sources
            )
            
            # Format sources for display
            sources = self.source_manager.format_sources_for_display(processed_sources)
            
            # Get the generation model from response
            generation_model = response_data.get("model", getattr(settings, 'GENERATION_MODEL', settings.CLAUDE_MODEL))
            logger.debug(f"Generation model from response: {response_data.get('model')}, fallback: {getattr(settings, 'GENERATION_MODEL', settings.CLAUDE_MODEL)}, final: {generation_model}")
            
            result = {
                "response": response_data["response"],
                "sources": sources,
                "context_count": len(processed_sources),
                "query_type": enhanced_query.query_type.value,
                "search_method": search_method,
                "usage": response_data.get("usage"),
                "citation_validation": citation_validation,
                "source_count": len(processed_sources),
                "generation_model": generation_model
            }
            
            # Cache the result
            if use_cache and self.cache:
                cache_params = {
                    "max_results": max_results,
                    "use_mmr": use_mmr,
                    "temperature": temperature,
                    "use_hybrid": use_hybrid,
                    "filters": filters
                }
                
                await self.cache.set(query, result, cache_params)
            
            return result
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "response": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "context_count": 0,
                "query_type": "error",
                "search_method": "error"
            }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        
        try:
            if not self._initialized:
                await self.initialize()
            
            # Get collection info
            collection_info = await self.qdrant_manager.get_collection_info()
            point_count = await self.qdrant_manager.count_points()
            
            return {
                "total_documents": point_count,
                "total_chunks": point_count,
                "total_teams": 1,  # Placeholder
                "total_channels": 1,  # Placeholder
                "embedding_dimension": settings.EMBEDDING_DIMENSION,
                "last_updated": datetime.now().isoformat(),
                "collection_info": collection_info.__dict__ if collection_info else {}
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    async def query_stream(
        self, 
        query: str, 
        max_results: int = 50,
        use_mmr: bool = True,
        temperature: float = None,
        use_hybrid: Optional[bool] = None,
        filters: Optional[Dict[str, Any]] = None
    ):
        """Stream a query response for real-time results
        
        Args:
            query: The search query
            max_results: Maximum number of results to retrieve
            use_mmr: Whether to use MMR for diversity
            temperature: Temperature for generation (overrides default)
            use_hybrid: Whether to use hybrid search (overrides default)
            filters: Optional filters to apply
            
        Yields:
            Dict[str, Any]: Streaming response chunks
        """
        
        try:
            if not self._initialized:
                await self.initialize()
            
            # Enhanced query analysis
            enhanced_query = self.query_enhancer.enhance_query(query)
            
            # Determine search method
            use_hybrid_search = use_hybrid if use_hybrid is not None else self._should_use_hybrid(enhanced_query)
            search_method = "hybrid" if use_hybrid_search else "vector"
            
            # Perform retrieval
            if use_hybrid_search:
                retriever = self.hybrid_retriever
                results = await retriever.search_with_mmr(
                    query, 
                    top_k=max_results,
                    mmr_diversity_threshold=0.5,
                    semantic_weight=0.7,
                    filters=filters
                )
            else:
                retriever = self.vector_retriever
                results = await retriever.search_with_mmr(
                    query,
                    top_k=max_results,
                    diversity_threshold=0.5,
                    filters=filters
                )
            
            if not results:
                yield {
                    "type": "complete",
                    "response": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "context_count": 0,
                    "query_type": enhanced_query.query_type.value,
                    "search_method": search_method
                }
                return
            
            # Extract context from results
            context_texts = [result.content for result in results[:10]]
            
            # Process sources
            processed_sources, citation_mapping = self.source_manager.process_retrieval_results(
                results, 
                max_sources=5,
                min_relevance_threshold=0.05
            )
            
            # Generate context and citation instructions
            context_text, citation_guide = self.source_manager.generate_source_context_for_llm(
                processed_sources, 
                citation_mapping
            )
            
            # Build source metadata
            source_metadata = []
            for source in processed_sources:
                metadata_item = {
                    "citation_key": source.citation_key,
                    "title": source.source_info.title,
                    "source_type": source.source_info.source_type,
                    "relevance_score": source.combined_relevance,
                    "authority_score": source.source_info.authority_score,
                    "chunk_count": source.source_info.chunk_count,
                    "timestamp": source.source_info.timestamp.isoformat() if source.source_info.timestamp else None
                }
                source_metadata.append(metadata_item)
            
            # Stream the response
            full_response = ""
            async for chunk in self.claude_client.generate_response_stream(
                context=[context_text],
                query=query,
                temperature=temperature or enhanced_query.suggested_params.get("temperature", 0.1),
                source_metadata=source_metadata,
                citation_guide=citation_guide
            ):
                if chunk["type"] == "content":
                    full_response += chunk["content"]
                    yield chunk
                elif chunk["type"] == "complete":
                    # Validate citations and format final response
                    citation_validation = self.source_manager.validate_citations_in_response(
                        chunk["response"], 
                        processed_sources
                    )
                    
                    sources = self.source_manager.format_sources_for_display(processed_sources)
                    
                    final_chunk = {
                        "type": "complete",
                        "response": chunk["response"],
                        "sources": sources,
                        "context_count": len(processed_sources),
                        "query_type": enhanced_query.query_type.value,
                        "search_method": search_method,
                        "usage": chunk.get("usage"),
                        "citation_validation": citation_validation,
                        "source_count": len(processed_sources),
                        "generation_model": chunk.get("model", self.claude_client.model)
                    }
                    
                    yield final_chunk
                    break
                else:
                    yield chunk
                    
        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def search(
        self,
        query: str,
        top_k: int = 50,
        filters: Dict[str, Any] = None
    ) -> List:
        """Perform similarity search"""
        
        try:
            if not self._initialized:
                await self.initialize()
            
            results = await self.vector_retriever.similarity_search(
                query=query,
                top_k=top_k,
                filters=filters
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def purge_database(
        self, 
        filters: Optional[Dict[str, Any]] = None,
        preview_only: bool = False,
        confirm_purge: bool = False
    ) -> Dict[str, Any]:
        """Purge database with safety checks and preview"""
        
        try:
            if not self._initialized:
                await self.initialize()
            
            if preview_only:
                # Just show what would be deleted
                return await self.qdrant_manager.get_purge_preview(filters=filters)
            
            if not confirm_purge:
                # Require explicit confirmation for actual purge
                preview = await self.qdrant_manager.get_purge_preview(filters=filters)
                preview["confirmation_required"] = True
                preview["message"] = "Add confirm_purge=True to actually perform the purge operation"
                return preview
            
            # Perform the actual purge
            result = await self.qdrant_manager.purge_collection(
                filters=filters,
                confirm=True
            )
            
            # Also clear BM25 index if it exists
            if self.hybrid_retriever and hasattr(self.hybrid_retriever, 'bm25_retriever'):
                logger.info("Clearing BM25 index...")
                bm25_result = await self.hybrid_retriever.bm25_retriever.clear_index()
                result["bm25_cleared"] = bm25_result
            
            # Clear Redis cache if enabled
            if hasattr(self, 'cache') and self.cache:
                logger.info("Clearing Redis cache...")
                try:
                    await self.cache.clear_all()
                    result["cache_cleared"] = True
                except Exception as e:
                    logger.warning(f"Failed to clear cache: {e}")
                    result["cache_cleared"] = False
            
            # Clear metadata store if applicable
            if hasattr(self, 'metadata_store') and self.metadata_store:
                logger.info("Clearing metadata store...")
                try:
                    # This would need to be implemented in metadata_store
                    # For now, we'll log that it's not implemented
                    logger.warning("Metadata store clearing not implemented")
                    result["metadata_cleared"] = "Not implemented"
                except Exception as e:
                    logger.warning(f"Failed to clear metadata: {e}")
                    result["metadata_cleared"] = False
            
            logger.info(f"Database purge completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Database purge failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_channel_status(self, channel_id: str) -> Dict[str, Any]:
        """Get processing status for a channel"""
        
        try:
            if not self._initialized:
                await self.initialize()
            
            return await self.channel_processor.get_channel_processing_status(channel_id)
            
        except Exception as e:
            logger.error(f"Failed to get channel status: {e}")
            return {"processed": False, "error": str(e)}
    
    async def ingest_channel_by_id(
        self,
        channel_id: str,
        team_id: str,
        use_enhanced: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Convenience method to ingest channel by ID with automatic method selection"""
        
        try:
            if use_enhanced:
                return await self.ingest_channel_enhanced(
                    channel_id=channel_id,
                    team_id=team_id,
                    **kwargs
                )
            else:
                # Fallback to basic method
                messages = await self.mattermost_client.get_channel_history(
                    channel_id=channel_id,
                    max_messages=kwargs.get("max_messages", 1000)
                )
                
                channel_info = await self.mattermost_client.get_channel_info(channel_id)
                team_info = await self.mattermost_client.get_team_info(team_id)
                
                return await self.ingest_channel_messages(
                    messages=messages,
                    channel_info=channel_info or {"id": channel_id, "display_name": "Unknown"},
                    team_info=team_info or {"id": team_id, "display_name": "Unknown"}
                )
                
        except Exception as e:
            logger.error(f"Channel ingestion by ID failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def query_with_channel_context(
        self,
        query: str,
        channel_id: Optional[str] = None,
        team_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Query with optional channel/team context filtering"""
        
        try:
            filters = {}
            
            if channel_id:
                filters["channel_id"] = channel_id
            if team_id:
                filters["team_id"] = team_id
            
            # Add filters to the query parameters
            if filters:
                kwargs["filters"] = filters
            
            return await self.query(query, **kwargs)
            
        except Exception as e:
            logger.error(f"Query with channel context failed: {e}")
            return {
                "response": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "context_count": 0,
                "query_type": "error",
                "search_method": "error"
            }
    
    async def query_stream(
        self, 
        query: str, 
        max_results: int = 50,
        use_mmr: bool = True,
        temperature: float = None,
        use_hybrid: Optional[bool] = None,
        use_cache: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Query the knowledge base with streaming response"""
        
        try:
            if not self._initialized:
                await self.initialize()
            
            # Yield initial status
            yield {
                "type": "status",
                "message": "Initializing query..."
            }
            
            # Check cache first (for retrieval phase)
            cache_key = None
            if use_cache and self.cache:
                cache_params = {
                    "max_results": max_results,
                    "use_mmr": use_mmr,
                    "temperature": temperature,
                    "use_hybrid": use_hybrid
                }
                
                # For streaming, we only cache the retrieval results, not the generation
                cache_key = f"retrieval_{query}"
                cached_retrieval = await self.cache.get(cache_key, cache_params)
                
                if cached_retrieval:
                    logger.info(f"Using cached retrieval results for query: {query[:50]}...")
                    retrieved_docs = cached_retrieval.get("retrieved_docs")
                    enhanced_query = cached_retrieval.get("enhanced_query")
                    search_method = cached_retrieval.get("search_method")
                    yield {
                        "type": "status",
                        "message": "Using cached retrieval results..."
                    }
                else:
                    # Perform retrieval
                    yield {
                        "type": "status",
                        "message": "Enhancing query..."
                    }
                    
                    # Enhance query
                    enhanced_query = await self.query_enhancer.enhance_query(query)
                    
                    yield {
                        "type": "status",
                        "message": "Searching knowledge base..."
                    }
                    
                    # Determine search method
                    use_hybrid_for_query = use_hybrid if use_hybrid is not None else self.use_hybrid_search
                    
                    # Get search results
                    if use_hybrid_for_query and self.hybrid_retriever:
                        search_method = "hybrid"
                        logger.info("Using hybrid search (vector + BM25)")
                        
                        if use_mmr:
                            results = await self.hybrid_retriever.search_with_mmr(
                                query=enhanced_query.enhanced_query,
                                top_k=max_results,
                                lambda_param=settings.MMR_LAMBDA,
                                filters=filters
                            )
                        else:
                            results = await self.hybrid_retriever.search(
                                query=enhanced_query.enhanced_query,
                                top_k=max_results,
                                filters=filters
                            )
                    else:
                        search_method = "vector"
                        logger.info("Using vector search")
                        
                        if use_mmr:
                            results = await self.vector_retriever.similarity_search_with_mmr(
                                query=enhanced_query.enhanced_query,
                                top_k=max_results,
                                lambda_param=settings.MMR_LAMBDA
                            )
                        else:
                            results = await self.vector_retriever.similarity_search(
                                query=enhanced_query.enhanced_query,
                                top_k=max_results,
                                filters=filters
                            )
                    
                    retrieved_docs = results
                    
                    # Cache retrieval results
                    if use_cache and self.cache and cache_key:
                        retrieval_cache = {
                            "retrieved_docs": retrieved_docs,
                            "enhanced_query": enhanced_query,
                            "search_method": search_method
                        }
                        await self.cache.set(cache_key, retrieval_cache, cache_params)
            
            if not retrieved_docs:
                yield {
                    "type": "error",
                    "message": "No relevant context found for your question."
                }
                return
            
            yield {
                "type": "status",
                "message": "Processing sources..."
            }
            
            # Process and rank sources
            processed_sources = self.source_manager.process_retrieved_documents(
                retrieved_docs=retrieved_docs,
                query=query,
                enhanced_query=enhanced_query
            )
            
            # Create citation mapping
            citation_mapping = self.source_manager.create_citation_mapping(processed_sources)
            
            # Prepare context for Claude
            context_text, citation_guide = self.source_manager.generate_source_context_for_llm(
                processed_sources, 
                citation_mapping
            )
            
            # Build source metadata
            source_metadata = []
            for source in processed_sources:
                metadata_item = {
                    "citation_key": source.citation_key,
                    "title": source.source_info.title,
                    "source_type": source.source_info.source_type,
                    "relevance_score": source.combined_relevance,
                    "authority_score": source.source_info.authority_score,
                    "chunk_count": source.source_info.chunk_count,
                    "timestamp": source.source_info.timestamp.isoformat() if source.source_info.timestamp else None
                }
                source_metadata.append(metadata_item)
            
            yield {
                "type": "status",
                "message": "Generating response..."
            }
            
            # Stream the response generation
            async for chunk in self.claude_client.generate_response_stream(
                context=[context_text],
                query=query,
                temperature=temperature or enhanced_query.suggested_params.get("temperature", 0.1),
                source_metadata=source_metadata,
                citation_guide=citation_guide
            ):
                # Pass through streaming chunks
                if chunk["type"] == "content":
                    yield chunk
                elif chunk["type"] == "complete":
                    # Add additional metadata to the complete response
                    response_data = chunk
                    
                    # Validate citations
                    citation_validation = self.source_manager.validate_citations_in_response(
                        response_data["response"], 
                        processed_sources
                    )
                    
                    # Format sources
                    sources = self.source_manager.format_sources_for_display(processed_sources)
                    
                    # Yield the final complete result
                    yield {
                        "type": "complete",
                        "response": response_data["response"],
                        "sources": sources,
                        "context_count": len(processed_sources),
                        "query_type": enhanced_query.query_type.value,
                        "search_method": search_method,
                        "usage": response_data.get("usage"),
                        "citation_validation": citation_validation,
                        "source_count": len(processed_sources)
                    }
                else:
                    # Pass through other event types
                    yield chunk
            
        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "message": f"I encountered an error while processing your question: {str(e)}"
            }
    
    @track_query("conversational")
    async def query_conversational(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_history_turns: int = 10,
        max_results: int = 50,
        use_mmr: bool = True,
        temperature: float = None,
        use_hybrid: Optional[bool] = None,
        use_cache: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        maintain_context: bool = True
    ) -> Dict[str, Any]:
        """Query with conversation history support
        
        Args:
            query: Current question
            conversation_id: ID of existing conversation to continue
            conversation_history: Previous conversation turns (alternative to conversation_id)
            max_history_turns: Maximum conversation history to include
            maintain_context: Whether to maintain conversation context for future queries
            ... other standard query parameters
        """
        
        try:
            if not self._initialized:
                await self.initialize()
            
            # Get or create conversation
            if conversation_id and conversation_id in self.conversations:
                conversation = self.conversations[conversation_id]
            elif conversation_history:
                # Create new conversation from provided history
                conversation = ConversationHistory()
                for turn in conversation_history:
                    conversation.add_turn(turn["role"], turn["content"])
            else:
                # New conversation
                conversation = ConversationHistory()
                conversation_id = conversation.conversation_id
            
            # Get conversation context window
            history_turns = conversation.get_context_window(max_history_turns)
            history_messages = [{"role": turn.role, "content": turn.content} for turn in history_turns]
            
            # Enhance query considering conversation context
            if history_messages:
                # Add conversation context to query enhancement
                context_aware_query = f"Previous conversation:\n"
                for msg in history_messages[-3:]:  # Last 3 exchanges for query enhancement
                    context_aware_query += f"{msg['role']}: {msg['content']}\n"
                context_aware_query += f"\nCurrent question: {query}"
                enhanced_query = await self.query_enhancer.enhance_query(context_aware_query)
            else:
                enhanced_query = await self.query_enhancer.enhance_query(query)
            
            # Determine search method
            use_hybrid_for_query = use_hybrid if use_hybrid is not None else self.use_hybrid_search
            
            # Get search results (same as regular query)
            if use_hybrid_for_query and self.hybrid_retriever:
                logger.info("Using hybrid search (vector + BM25) for conversational query")
                
                if use_mmr:
                    results = await self.hybrid_retriever.search_with_mmr(
                        query=enhanced_query.enhanced_query,
                        top_k=max_results,
                        mmr_lambda=enhanced_query.suggested_params.get("mmr_lambda", 0.7),
                        filters=filters
                    )
                else:
                    results = await self.hybrid_retriever.search(
                        query=enhanced_query.enhanced_query,
                        top_k=max_results,
                        filters=filters
                    )
                
                search_method = "hybrid_mmr" if use_mmr else "hybrid"
            else:
                logger.info("Using vector search only for conversational query")
                
                if use_mmr:
                    results = await self.vector_retriever.mmr_search(
                        query=enhanced_query.enhanced_query,
                        top_k=max_results,
                        mmr_lambda=enhanced_query.suggested_params.get("mmr_lambda", 0.7)
                    )
                else:
                    results = await self.vector_retriever.similarity_search(
                        query=enhanced_query.enhanced_query,
                        top_k=max_results,
                        filters=filters
                    )
                
                search_method = "vector_mmr" if use_mmr else "vector"
            
            if not results:
                response = "I couldn't find any relevant information to answer your question."
                # Still maintain conversation
                if maintain_context:
                    conversation.add_turn("user", query)
                    conversation.add_turn("assistant", response)
                    self.conversations[conversation_id] = conversation
                
                return {
                    "response": response,
                    "conversation_id": conversation_id,
                    "sources": [],
                    "context_count": 0,
                    "query_type": enhanced_query.query_type.value,
                    "search_method": search_method,
                    "conversation_metadata": {
                        "turn_count": len(conversation.turns),
                        "created_at": conversation.created_at.isoformat()
                    }
                }
            
            # Process sources
            processed_sources, citation_mapping = self.source_manager.process_retrieval_results(
                results, 
                max_sources=5,
                min_relevance_threshold=0.05
            )
            
            # Generate context and citation instructions
            context_text, citation_guide = self.source_manager.generate_source_context_for_llm(
                processed_sources, 
                citation_mapping
            )
            
            # Build source metadata
            source_metadata = []
            for source in processed_sources:
                metadata_item = {
                    "citation_key": source.citation_key,
                    "title": source.source_info.title,
                    "source_type": source.source_info.source_type,
                    "relevance_score": source.combined_relevance,
                    "authority_score": source.source_info.authority_score,
                    "chunk_count": source.source_info.chunk_count,
                    "timestamp": source.source_info.timestamp.isoformat() if source.source_info.timestamp else None
                }
                source_metadata.append(metadata_item)
            
            # Generate response with conversation history
            response_data = await self.claude_client.generate_conversational_response(
                context=[context_text],
                query=query,
                conversation_history=history_messages,
                temperature=temperature or enhanced_query.suggested_params.get("temperature", 0.1),
                source_metadata=source_metadata,
                citation_guide=citation_guide
            )
            
            # Validate citations
            citation_validation = self.source_manager.validate_citations_in_response(
                response_data["response"], 
                processed_sources
            )
            
            # Format sources
            sources = self.source_manager.format_sources_for_display(processed_sources)
            
            # Update conversation history if maintaining context
            if maintain_context:
                conversation.add_turn("user", query, {
                    "query_type": enhanced_query.query_type.value,
                    "sources_retrieved": len(results)
                })
                conversation.add_turn("assistant", response_data["response"], {
                    "sources_used": len(sources)
                })
                self.conversations[conversation_id] = conversation
            
            result = {
                "response": response_data["response"],
                "conversation_id": conversation_id,
                "sources": sources,
                "context_count": len(processed_sources),
                "query_type": enhanced_query.query_type.value,
                "search_method": search_method,
                "usage": response_data.get("usage"),
                "citation_validation": citation_validation,
                "source_count": len(processed_sources),
                "conversation_metadata": {
                    "turn_count": len(conversation.turns),
                    "created_at": conversation.created_at.isoformat(),
                    "updated_at": conversation.updated_at.isoformat()
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Conversational query failed: {e}")
            return {
                "response": f"I encountered an error while processing your question: {str(e)}",
                "conversation_id": conversation_id or str(uuid.uuid4()),
                "sources": [],
                "context_count": 0,
                "query_type": "error",
                "search_method": "error",
                "conversation_metadata": {}
            }
    
    async def query_conversational_stream(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_history_turns: int = 10,
        max_results: int = 50,
        use_mmr: bool = True,
        temperature: float = None,
        use_hybrid: Optional[bool] = None,
        use_cache: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        maintain_context: bool = True
    ) -> AsyncIterator[Dict[str, Any]]:
        """Query with conversation history support and streaming response"""
        
        try:
            if not self._initialized:
                await self.initialize()
            
            # Yield initial status
            yield {
                "type": "status",
                "message": "Initializing conversational query..."
            }
            
            # Get or create conversation
            if conversation_id and conversation_id in self.conversations:
                conversation = self.conversations[conversation_id]
            elif conversation_history:
                # Create new conversation from provided history
                conversation = ConversationHistory()
                for turn in conversation_history:
                    conversation.add_turn(turn["role"], turn["content"])
            else:
                # New conversation
                conversation = ConversationHistory()
                conversation_id = conversation.conversation_id
            
            # Get conversation context window
            history_turns = conversation.get_context_window(max_history_turns)
            history_messages = [{"role": turn.role, "content": turn.content} for turn in history_turns]
            
            yield {
                "type": "status",
                "message": "Enhancing query with conversation context..."
            }
            
            # Enhance query considering conversation context
            if history_messages:
                # Add conversation context to query enhancement
                context_aware_query = f"Previous conversation:\n"
                for msg in history_messages[-3:]:  # Last 3 exchanges for query enhancement
                    context_aware_query += f"{msg['role']}: {msg['content']}\n"
                context_aware_query += f"\nCurrent question: {query}"
                enhanced_query = await self.query_enhancer.enhance_query(context_aware_query)
            else:
                enhanced_query = await self.query_enhancer.enhance_query(query)
            
            yield {
                "type": "status",
                "message": "Searching knowledge base..."
            }
            
            # Determine search method
            use_hybrid_for_query = use_hybrid if use_hybrid is not None else self.use_hybrid_search
            
            # Get search results (same as regular query)
            if use_hybrid_for_query and self.hybrid_retriever:
                search_method = "hybrid"
                logger.info("Using hybrid search (vector + BM25) for conversational query")
                
                if use_mmr:
                    results = await self.hybrid_retriever.search_with_mmr(
                        query=enhanced_query.enhanced_query,
                        top_k=max_results,
                        lambda_param=settings.MMR_LAMBDA,
                        filters=filters
                    )
                else:
                    results = await self.hybrid_retriever.search(
                        query=enhanced_query.enhanced_query,
                        top_k=max_results,
                        filters=filters
                    )
            else:
                search_method = "vector"
                logger.info("Using vector search for conversational query")
                
                if use_mmr:
                    results = await self.vector_retriever.similarity_search_with_mmr(
                        query=enhanced_query.enhanced_query,
                        top_k=max_results,
                        lambda_param=settings.MMR_LAMBDA
                    )
                else:
                    results = await self.vector_retriever.similarity_search(
                        query=enhanced_query.enhanced_query,
                        top_k=max_results,
                        filters=filters
                    )
            
            if not results:
                yield {
                    "type": "error",
                    "message": "No relevant context found for your question."
                }
                return
            
            yield {
                "type": "status",
                "message": "Processing sources..."
            }
            
            # Process and rank sources
            processed_sources, citation_mapping = self.source_manager.process_retrieval_results(
                results, 
                max_sources=5,
                min_relevance_threshold=0.05
            )
            
            # Generate context and citation instructions
            context_text, citation_guide = self.source_manager.generate_source_context_for_llm(
                processed_sources, 
                citation_mapping
            )
            
            # Build source metadata
            source_metadata = []
            for source in processed_sources:
                metadata_item = {
                    "citation_key": source.citation_key,
                    "title": source.source_info.title,
                    "source_type": source.source_info.source_type,
                    "relevance_score": source.combined_relevance,
                    "authority_score": source.source_info.authority_score,
                    "chunk_count": source.source_info.chunk_count,
                    "timestamp": source.source_info.timestamp.isoformat() if source.source_info.timestamp else None
                }
                source_metadata.append(metadata_item)
            
            yield {
                "type": "status",
                "message": "Generating response with conversation context..."
            }
            
            # Keep track of the full response for conversation history
            full_response = ""
            
            # Stream the response generation
            async for chunk in self.claude_client.generate_conversational_response_stream(
                context=[context_text],
                query=query,
                conversation_history=history_messages,
                temperature=temperature or enhanced_query.suggested_params.get("temperature", 0.1),
                source_metadata=source_metadata,
                citation_guide=citation_guide
            ):
                # Pass through streaming chunks
                if chunk["type"] == "content":
                    full_response += chunk["content"]
                    yield chunk
                elif chunk["type"] == "complete":
                    # Add additional metadata to the complete response
                    response_data = chunk
                    
                    # Validate citations
                    citation_validation = self.source_manager.validate_citations_in_response(
                        response_data["response"], 
                        processed_sources
                    )
                    
                    # Format sources
                    sources = self.source_manager.format_sources_for_display(processed_sources)
                    
                    # Update conversation history if maintaining context
                    if maintain_context:
                        conversation.add_turn("user", query, {
                            "query_type": enhanced_query.query_type.value,
                            "sources_retrieved": len(results)
                        })
                        conversation.add_turn("assistant", full_response, {
                            "sources_used": len(sources)
                        })
                        self.conversations[conversation_id] = conversation
                    
                    # Yield the final complete result
                    yield {
                        "type": "complete",
                        "response": response_data["response"],
                        "conversation_id": conversation_id,
                        "sources": sources,
                        "context_count": len(processed_sources),
                        "query_type": enhanced_query.query_type.value,
                        "search_method": search_method,
                        "usage": response_data.get("usage"),
                        "citation_validation": citation_validation,
                        "source_count": len(processed_sources),
                        "conversation_metadata": {
                            "turn_count": len(conversation.turns),
                            "created_at": conversation.created_at.isoformat(),
                            "updated_at": conversation.updated_at.isoformat()
                        }
                    }
                else:
                    # Pass through other event types
                    yield chunk
            
        except Exception as e:
            logger.error(f"Streaming conversational query failed: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "message": f"I encountered an error while processing your question: {str(e)}"
            }