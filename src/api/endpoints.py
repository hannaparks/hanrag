from fastapi import FastAPI, Form, HTTPException, BackgroundTasks, Depends, Security, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
from typing import Optional, Dict, Any, AsyncIterator
import uuid
from datetime import datetime
from loguru import logger
from pathlib import Path
import os
import json

from .models import (
    SlashCommandResponse, QueryRequest, QueryResponse, UrlIngestionRequest,
    ChannelIngestionRequest, IngestionResponse, HealthResponse, SearchRequest,
    SearchResponse, StatsResponse, ErrorResponse, SystemConfig,
    ConversationalQueryRequest, ConversationalQueryResponse
)
from ..mattermost.slash_commands import SlashCommandHandler
from ..config.settings import settings
from ..config.logging import setup_logging
from .auth import get_api_key, require_write, require_admin, optional_api_key, api_key_manager
from .rate_limit import require_default_rate_limit, require_expensive_rate_limit
from ..monitoring.webhook_notifications import webhook_service
<<<<<<< HEAD
=======
from ..monitoring.monitoring_integration import MonitoringManager
from ..monitoring.dashboard import MonitoringDashboard
from ..utils.path_security import (
    secure_file_response_path, get_static_files_dir, 
    ALLOWED_STATIC_EXTENSIONS, ALLOWED_CONFIG_EXTENSIONS
)
>>>>>>> 66c74c8

# Initialize logging
setup_logging()

# Create FastAPI app
app = FastAPI(
    title="RAG Assistant API",
    description="Retrieval Augmented Generation system for Mattermost",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Add middleware to include rate limit headers
@app.middleware("http")
async def add_rate_limit_headers(request: Request, call_next):
    """Add rate limit headers to response if available"""
    response = await call_next(request)
    
    # Check if rate limit headers were set
    if hasattr(request.state, 'rate_limit_headers'):
        for header, value in request.state.rate_limit_headers.items():
            response.headers[header] = value
    
    return response

# Global instances (will be initialized on startup)
slash_handler: Optional[SlashCommandHandler] = None
rag_pipeline = None
metrics_collector = None
monitoring_manager = None
monitoring_dashboard = None

# Task storage (in production, use Redis or database)
background_tasks: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize application components"""
    global slash_handler, rag_pipeline, metrics_collector, monitoring_manager, monitoring_dashboard
    
    logger.info("Starting RAG Assistant API...")
    
    try:
        # Initialize cache first
        from ..cache import initialize_cache
        await initialize_cache()
        
        # Initialize slash command handler
        slash_handler = SlashCommandHandler()
        
        # Initialize RAG pipeline
        from ..core.rag_pipeline import RAGPipeline
        rag_pipeline = RAGPipeline()
        await rag_pipeline.initialize()
        slash_handler.set_rag_pipeline(rag_pipeline)
        
        # Initialize metrics collector
        from ..monitoring.metrics import MetricsCollector, set_system_info
        metrics_collector = MetricsCollector(rag_pipeline)
        await metrics_collector.start()
        
        # Set system info for metrics
        set_system_info(version="1.0.0", environment=settings.ENVIRONMENT)
        
        # Initialize webhook service
        await webhook_service.start()
        
        # Initialize monitoring manager and dashboard
        monitoring_manager = MonitoringManager()
        monitoring_dashboard = MonitoringDashboard(monitoring_manager)
        
        logger.info("RAG Assistant API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start RAG Assistant API: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down RAG Assistant API...")
    
    # Stop metrics collector
    if metrics_collector:
        await metrics_collector.stop()
    
    # Stop webhook service
    await webhook_service.stop()
    
    # Cleanup monitoring manager
    if monitoring_manager:
        monitoring_manager.cleanup()
    
    # Disconnect from cache
    from ..cache import cache_manager
    if cache_manager:
        await cache_manager.disconnect()


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    components = {
        "api": "healthy",
        "mattermost": "unknown",
        "qdrant": "unknown",
        "openai": "unknown",
        "claude": "unknown"
    }
    
    # TODO: Add actual component health checks
    
    return HealthResponse(
        status="healthy",
        service="RAG Assistant",
        timestamp=datetime.now(),
        version="1.0.0",
        components=components
    )


@app.get("/config", response_class=HTMLResponse)
async def configuration_page():
    """Serve the configuration page"""
    config_path = Path(__file__).parent / "static" / "configuration.html"
    if config_path.exists():
        return FileResponse(config_path)
    else:
        raise HTTPException(status_code=404, detail="Configuration page not found")


@app.get("/metrics")
async def get_metrics(api_key: Optional[str] = Security(optional_api_key)):
    """Prometheus metrics endpoint
    
    Returns metrics in Prometheus text format.
    Authentication is optional to allow Prometheus scraping.
    """
    from ..monitoring.metrics import get_metrics
    
    try:
        metrics_data = get_metrics()
        return Response(
            content=metrics_data,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate metrics")


# Mattermost slash command endpoints
@app.post("/mattermost/inject", response_model=SlashCommandResponse)
async def handle_inject_command(
    token: str = Form(...),
    team_id: str = Form(...),
    team_domain: str = Form(...),
    channel_id: str = Form(...),
    channel_name: str = Form(...),
    user_id: str = Form(...),
    user_name: str = Form(...),
    command: str = Form(...),
    text: str = Form(default=""),
    response_url: str = Form(...),
    trigger_id: str = Form(...)
):
    """Handle /inject slash command"""
    
    if not slash_handler:
        raise HTTPException(status_code=503, detail="Service not available")
    
    try:
        response = await slash_handler.handle_inject_command(
            token=token,
            team_id=team_id,
            channel_id=channel_id,
            user_id=user_id,
            text=text
        )
        
        return SlashCommandResponse(**response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inject command failed: {e}")
        return SlashCommandResponse(
            response_type="ephemeral",
            text=f"❌ **Error processing inject command**\n```\n{str(e)}\n```",
            username="RAG Assistant",
            icon_emoji=":x:"
        )


@app.post("/mattermost/ask", response_model=SlashCommandResponse)
async def handle_ask_command(
    token: str = Form(...),
    team_id: str = Form(...),
    team_domain: str = Form(...),
    channel_id: str = Form(...),
    channel_name: str = Form(...),
    user_id: str = Form(...),
    user_name: str = Form(...),
    command: str = Form(...),
    text: str = Form(...),
    response_url: str = Form(...),
    trigger_id: str = Form(...)
):
    """Handle /ask slash command"""
    
    if not slash_handler:
        raise HTTPException(status_code=503, detail="Service not available")
    
    try:
        response = await slash_handler.handle_ask_command(
            token=token,
            team_id=team_id,
            channel_id=channel_id,
            user_id=user_id,
            text=text
        )
        
        return SlashCommandResponse(**response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ask command failed: {e}")
        return SlashCommandResponse(
            response_type="ephemeral",
            text=f"❌ **Error processing question**\n```\n{str(e)}\n```",
            username="RAG Assistant",
            icon_emoji=":x:"
        )


# API endpoints for direct access
@app.post("/api/query", response_model=QueryResponse, dependencies=[Depends(require_expensive_rate_limit)])
async def query_knowledge_base(
    request: QueryRequest,
    req: Request,
    api_key: str = Security(get_api_key)
):
    """Query the knowledge base directly (requires API key, rate limited)"""
    
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    try:
        start_time = asyncio.get_event_loop().time()
        
        # Execute query (no channel filtering)
        result = await rag_pipeline.query(
            query=request.question,
            max_results=request.max_results,
            use_mmr=request.use_mmr,
            temperature=request.temperature
        )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return QueryResponse(
            response=result["response"],
            sources=result.get("sources", []),
            context_count=result.get("context_count", 0),
            query_type=result.get("query_type", "simple_factual"),
            processing_time=processing_time,
            usage=result.get("usage")
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/stream", dependencies=[Depends(require_expensive_rate_limit)])
async def query_knowledge_base_stream(
    request: QueryRequest,
    req: Request,
    api_key: str = Security(get_api_key)
):
    """Query the knowledge base with streaming response (requires API key, rate limited)"""
    
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    async def generate_stream() -> AsyncIterator[str]:
        """Generate Server-Sent Events (SSE) stream"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Stream query results
            async for chunk in rag_pipeline.query_stream(
                query=request.question,
                max_results=request.max_results,
                use_mmr=request.use_mmr,
                temperature=request.temperature
            ):
                # Format as Server-Sent Event
                if chunk["type"] == "complete":
                    # Add processing time to complete event
                    chunk["processing_time"] = asyncio.get_event_loop().time() - start_time
                
                # Yield as SSE format
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # Send final done event
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            error_chunk = {
                "type": "error",
                "error": str(e),
                "message": f"Streaming failed: {str(e)}"
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@app.post("/api/query/conversational", response_model=ConversationalQueryResponse, dependencies=[Depends(require_expensive_rate_limit)])
async def query_conversational(
    request: ConversationalQueryRequest,
    req: Request,
    api_key: str = Security(get_api_key)
):
    """Query the knowledge base with conversation history support (requires API key, rate limited)"""
    
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    try:
        start_time = asyncio.get_event_loop().time()
        
        # Convert conversation history if provided
        history = None
        if request.conversation_history:
            history = [{"role": turn.role, "content": turn.content} for turn in request.conversation_history]
        
        # Execute conversational query
        result = await rag_pipeline.query_conversational(
            query=request.question,
            conversation_id=request.conversation_id,
            conversation_history=history,
            max_history_turns=request.max_history_turns,
            max_results=request.max_results,
            use_mmr=request.use_mmr,
            temperature=request.temperature,
            maintain_context=request.maintain_context
        )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return ConversationalQueryResponse(
            response=result["response"],
            conversation_id=result["conversation_id"],
            sources=result.get("sources", []),
            context_count=result.get("context_count", 0),
            query_type=result.get("query_type", "simple_factual"),
            processing_time=processing_time,
            usage=result.get("usage"),
            conversation_metadata=result.get("conversation_metadata")
        )
        
    except Exception as e:
        logger.error(f"Conversational query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/conversational/stream", dependencies=[Depends(require_expensive_rate_limit)])
async def query_conversational_stream(
    request: ConversationalQueryRequest,
    req: Request,
    api_key: str = Security(get_api_key)
):
    """Query the knowledge base with conversation history and streaming response (requires API key, rate limited)"""
    
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    async def generate_stream() -> AsyncIterator[str]:
        """Generate Server-Sent Events (SSE) stream for conversational query"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Convert conversation history if provided
            history = None
            if request.conversation_history:
                history = [{"role": turn.role, "content": turn.content} for turn in request.conversation_history]
            
            # Stream conversational query results
            async for chunk in rag_pipeline.query_conversational_stream(
                query=request.question,
                conversation_id=request.conversation_id,
                conversation_history=history,
                max_history_turns=request.max_history_turns,
                max_results=request.max_results,
                use_mmr=request.use_mmr,
                temperature=request.temperature,
                maintain_context=request.maintain_context
            ):
                # Format as Server-Sent Event
                if chunk["type"] == "complete":
                    # Add processing time to complete event
                    chunk["processing_time"] = asyncio.get_event_loop().time() - start_time
                
                # Yield as SSE format
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # Send final done event
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming conversational query failed: {e}")
            error_chunk = {
                "type": "error",
                "error": str(e),
                "message": f"Streaming failed: {str(e)}"
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    api_key: str = Security(get_api_key)
):
    """Get conversation history by ID (requires API key)"""
    
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    if conversation_id not in rag_pipeline.conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation = rag_pipeline.conversations[conversation_id]
    
    return {
        "conversation_id": conversation_id,
        "turns": [
            {
                "role": turn.role,
                "content": turn.content,
                "timestamp": turn.timestamp.isoformat(),
                "metadata": turn.metadata
            }
            for turn in conversation.turns
        ],
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
        "metadata": conversation.metadata
    }


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    api_key: str = Security(get_api_key)
):
    """Delete a conversation by ID (requires API key)"""
    
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    if conversation_id not in rag_pipeline.conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    del rag_pipeline.conversations[conversation_id]
    
    return {"message": f"Conversation {conversation_id} deleted successfully"}


@app.post("/api/ingest/url", response_model=IngestionResponse, dependencies=[Depends(require_expensive_rate_limit)])
async def ingest_url(
    request: UrlIngestionRequest,
    background_tasks: BackgroundTasks,
    req: Request,
    api_key: str = Security(require_write)
):
    """Ingest content from URL (requires write permissions, rate limited)"""
    
<<<<<<< HEAD
=======
    logger.info(f"URL ingestion request: {request.url}")
>>>>>>> 66c74c8
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    task_id = str(uuid.uuid4())
    
    # Store task info
    background_tasks_store = {
        "task_id": task_id,
        "status": "pending",
        "created_at": datetime.now(),
        "type": "url_ingestion",
        "url": str(request.url)
    }
    background_tasks[task_id] = background_tasks_store
    
    # Start background task
    background_tasks.add_task(
        _ingest_url_task,
        task_id,
        request
    )
    
    return IngestionResponse(
        status="pending",
        task_id=task_id,
        message=f"URL ingestion started: {request.url}"
    )


@app.post("/api/ingest/channel", response_model=IngestionResponse, dependencies=[Depends(require_expensive_rate_limit)])
async def ingest_channel(
    request: ChannelIngestionRequest,
    background_tasks: BackgroundTasks,
    req: Request,
    api_key: str = Security(require_write)
):
    """Ingest channel message history (requires write permissions, rate limited)"""
    
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    task_id = str(uuid.uuid4())
    
    # Store task info
    background_tasks_store = {
        "task_id": task_id,
        "status": "pending",
        "created_at": datetime.now(),
        "type": "channel_ingestion",
        "channel_id": request.channel_id
    }
    background_tasks[task_id] = background_tasks_store
    
    # Start background task
    background_tasks.add_task(
        _ingest_channel_task,
        task_id,
        request
    )
    
    return IngestionResponse(
        status="pending",
        task_id=task_id,
        message=f"Channel ingestion started: {request.channel_id}"
    )


@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    """Get background task status"""
    
    if task_id not in background_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return background_tasks[task_id]


@app.post("/api/search", response_model=SearchResponse, dependencies=[Depends(require_default_rate_limit)])
async def search_similarity(
    request: SearchRequest,
    req: Request,
    api_key: str = Security(get_api_key)
):
    """Perform similarity search (requires API key, rate limited)"""
    
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    try:
        start_time = asyncio.get_event_loop().time()
        
        # Build filters
        filters = {}
        if request.channel_filter:
            filters["channel_id"] = request.channel_filter
        if request.team_filter:
            filters["team_id"] = request.team_filter
        if request.metadata_filters:
            filters.update(request.metadata_filters)
        
        # Perform search
        results = await rag_pipeline.search(
            query=request.query,
            top_k=request.top_k,
            filters=filters
        )
        
        query_time = asyncio.get_event_loop().time() - start_time
        
        return SearchResponse(
            results=[result.to_dict() for result in results],
            total_results=len(results),
            query_time=query_time,
            query_type="simple_factual"  # TODO: Detect query type
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", response_model=StatsResponse)
async def get_system_stats(
    api_key: Optional[str] = Security(optional_api_key)
):
    """Get system statistics (optionally authenticated for more details)"""
    
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    try:
        stats = await rag_pipeline.get_stats()
        
        return StatsResponse(
            total_documents=stats.get("total_documents", 0),
            total_chunks=stats.get("total_chunks", 0),
            total_teams=stats.get("total_teams", 0),
            total_channels=stats.get("total_channels", 0),
            embedding_dimension=settings.EMBEDDING_DIMENSION,
            last_updated=stats.get("last_updated"),
            storage_size=stats.get("storage_size")
        )
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/config", response_model=SystemConfig)
async def get_system_config():
    """Get system configuration
    
    Reads configuration from both settings and .env file to ensure
    we show the most current values.
    """
    
    # Import env manager to read current .env values
    from ..utils.env_manager import EnvFileManager
    env_manager = EnvFileManager()
    env_vars = env_manager.read_env_file()
    
    # Mask sensitive API keys for display
    def mask_api_key(key: str) -> str:
        if not key:
            return None
        if len(key) > 8:
            return f"{key[:4]}...{key[-4:]}"
        return "***"
    
    # Helper to get value from env or settings
    def get_value(env_key: str, settings_value: Any, is_int: bool = False, is_float: bool = False, is_bool: bool = False):
        """Get value from env file first, then fall back to settings"""
        env_value = env_vars.get(env_key)
        if env_value is not None:
            if is_int:
                try:
                    return int(env_value)
                except:
                    pass
            elif is_float:
                try:
                    return float(env_value)
                except:
                    pass
            elif is_bool:
                return env_value.lower() in ('true', '1', 'yes', 'on')
            return env_value
        return settings_value
    
    return SystemConfig(
        # API Keys (masked for security)
        openai_api_key=mask_api_key(get_value('OPENAI_API_KEY', settings.OPENAI_API_KEY)),
        anthropic_api_key=mask_api_key(get_value('ANTHROPIC_API_KEY', settings.ANTHROPIC_API_KEY)),
        
        # Mattermost Settings
        mattermost_url=get_value('MATTERMOST_URL', settings.MATTERMOST_URL),
        mattermost_personal_access_token=mask_api_key(get_value('MATTERMOST_PERSONAL_ACCESS_TOKEN', getattr(settings, 'MATTERMOST_PERSONAL_ACCESS_TOKEN', None))),
        mattermost_inject_token=mask_api_key(get_value('MATTERMOST_INJECT_TOKEN', getattr(settings, 'MATTERMOST_INJECT_TOKEN', None))),
        mattermost_ask_token=mask_api_key(get_value('MATTERMOST_ASK_TOKEN', getattr(settings, 'MATTERMOST_ASK_TOKEN', None))),
        
        # RAG Settings
        embedding={
            "model": get_value('EMBEDDING_MODEL', settings.EMBEDDING_MODEL),
            "dimension": get_value('EMBEDDING_DIMENSION', settings.EMBEDDING_DIMENSION, is_int=True),
            "max_tokens": get_value('CHUNK_SIZE', settings.CHUNK_SIZE, is_int=True)
        },
        retrieval={
            "top_k": get_value('SIMILARITY_TOP_K', settings.SIMILARITY_TOP_K, is_int=True),
            "rerank_top_n": get_value('RERANK_TOP_N', settings.RERANK_TOP_N, is_int=True),
            "mmr_lambda": get_value('MMR_LAMBDA', settings.MMR_LAMBDA, is_float=True),
            "use_hybrid_search": get_value('USE_HYBRID_SEARCH', True, is_bool=True),
            "vector_weight": get_value('HYBRID_VECTOR_WEIGHT', settings.HYBRID_VECTOR_WEIGHT, is_float=True),
            "bm25_weight": get_value('HYBRID_BM25_WEIGHT', settings.HYBRID_BM25_WEIGHT, is_float=True),
            "rrf_k": get_value('RRF_K', settings.HYBRID_RRF_K, is_int=True),
            "min_score_threshold": get_value('MIN_SCORE_THRESHOLD', settings.HYBRID_MIN_SCORE_THRESHOLD, is_float=True),
            "bm25_k1": get_value('BM25_K1', settings.BM25_K1, is_float=True),
            "bm25_b": get_value('BM25_B', settings.BM25_B, is_float=True)
        },
        generation={
            "model": get_value('GENERATION_MODEL', get_value('CLAUDE_MODEL', settings.CLAUDE_MODEL)),
            "max_tokens": get_value('GENERATION_MAX_TOKENS', get_value('MAX_TOKENS', settings.MAX_TOKENS, is_int=True), is_int=True),
            "temperature": get_value('TEMPERATURE', settings.TEMPERATURE, is_float=True),
            "enable_hybrid_mode": get_value('ENABLE_HYBRID_MODE', False, is_bool=True)
        },
        channel_processing={
            "conversation_gap_minutes": get_value('CONVERSATION_GAP_MINUTES', settings.CONVERSATION_GAP_MINUTES, is_int=True),
            "max_group_size": get_value('MAX_GROUP_SIZE', settings.MAX_GROUP_SIZE, is_int=True),
            "chunk_overlap": get_value('CHUNK_OVERLAP', settings.CHUNK_OVERLAP, is_int=True),
            "parent_chunk_size": get_value('PARENT_CHUNK_SIZE', settings.PARENT_CHUNK_SIZE, is_int=True),
            "preserve_threads": get_value('PRESERVE_THREADS', settings.PRESERVE_THREADS, is_bool=True)
        },
        database={
            "qdrant_host": get_value('QDRANT_HOST', settings.QDRANT_HOST),
            "qdrant_port": get_value('QDRANT_PORT', settings.QDRANT_PORT, is_int=True),
            "collection_name": get_value('QDRANT_COLLECTION_NAME', settings.QDRANT_COLLECTION_NAME)
        },
        server={
            "host": get_value('SERVER_HOST', settings.SERVER_HOST),
            "port": get_value('SERVER_PORT', settings.SERVER_PORT, is_int=True)
        },
        system={
            "log_level": get_value('LOG_LEVEL', settings.LOG_LEVEL)
        },
        environment=get_value('ENVIRONMENT', settings.ENVIRONMENT)
    )


@app.get("/configuration", response_class=HTMLResponse)
async def get_configuration_page():
    """Serve the configuration HTML page"""
    config_file = Path(__file__).parent / "static" / "configuration.html"
    
    if not config_file.exists():
        raise HTTPException(status_code=404, detail="Configuration page not found")
    
    return FileResponse(config_file)


@app.get("/configuration_{theme}.html", response_class=HTMLResponse)
async def get_theme_configuration_page(theme: str):
    """Serve theme-specific configuration pages with path security"""
    # Validate theme parameter (v1-v12 only)
    if not theme.startswith('v') or not theme[1:].isdigit() or int(theme[1:]) < 1 or int(theme[1:]) > 12:
        raise HTTPException(status_code=404, detail="Theme not found")
    
    # Look for theme file in project root
    theme_file = f"configuration_{theme}.html"
    project_root = Path(__file__).parent.parent.parent
    theme_path = project_root / theme_file
    
    if not theme_path.exists():
        raise HTTPException(status_code=404, detail=f"Theme configuration file {theme_file} not found")
    
    return FileResponse(theme_path)


@app.post("/api/config/ui", response_model=Dict[str, Any])
async def update_system_config_ui(
    config: SystemConfig
):
    """Update system configuration from UI (development friendly)
    
    This endpoint is specifically for the configuration UI and has relaxed
    authentication requirements in development mode.
    """
    
    # In production, this should be protected
    if settings.ENVIRONMENT == "production":
        logger.warning("Configuration UI update endpoint accessed in production mode")
    
    try:
        from ..utils.env_manager import EnvFileManager
        
        # Initialize environment manager
        env_manager = EnvFileManager()
        
        # Check if .env file exists and is writable
        if not env_manager.env_file_path.exists():
            logger.warning(f".env file not found at {env_manager.env_file_path}, creating it")
            env_manager.env_file_path.touch()
        
        if not os.access(env_manager.env_file_path, os.W_OK):
            raise HTTPException(
                status_code=500,
                detail=f".env file is not writable: {env_manager.env_file_path}"
            )
        
        # Prepare updates dictionary
        updates = {}
        
        # Update API Keys (only if provided and not masked)
        if config.openai_api_key and "..." not in config.openai_api_key:
            updates['OPENAI_API_KEY'] = config.openai_api_key
            
        if config.anthropic_api_key and "..." not in config.anthropic_api_key:
            updates['ANTHROPIC_API_KEY'] = config.anthropic_api_key
        
        # Update Mattermost Settings
        if config.mattermost_url:
            updates['MATTERMOST_URL'] = config.mattermost_url
            
        if config.mattermost_personal_access_token and "..." not in config.mattermost_personal_access_token:
            updates['MATTERMOST_PERSONAL_ACCESS_TOKEN'] = config.mattermost_personal_access_token
            
        if config.mattermost_inject_token and "..." not in config.mattermost_inject_token:
            updates['MATTERMOST_INJECT_TOKEN'] = config.mattermost_inject_token
            
        if config.mattermost_ask_token and "..." not in config.mattermost_ask_token:
            updates['MATTERMOST_ASK_TOKEN'] = config.mattermost_ask_token
        
        # Update embedding settings
        if config.embedding:
            if hasattr(config.embedding, 'model') and config.embedding.model:
                updates['EMBEDDING_MODEL'] = config.embedding.model
            if hasattr(config.embedding, 'max_tokens') and config.embedding.max_tokens:
                updates['CHUNK_SIZE'] = str(config.embedding.max_tokens)
        
        # Update retrieval settings
        if config.retrieval:
            if hasattr(config.retrieval, 'top_k') and config.retrieval.top_k:
                updates['SIMILARITY_TOP_K'] = str(config.retrieval.top_k)
            if hasattr(config.retrieval, 'rerank_top_n') and config.retrieval.rerank_top_n:
                updates['RERANK_TOP_N'] = str(config.retrieval.rerank_top_n)
            if hasattr(config.retrieval, 'mmr_lambda') and config.retrieval.mmr_lambda is not None:
                updates['MMR_LAMBDA'] = str(config.retrieval.mmr_lambda)
            if hasattr(config.retrieval, 'use_hybrid_search'):
                updates['USE_HYBRID_SEARCH'] = str(config.retrieval.use_hybrid_search)
            if hasattr(config.retrieval, 'vector_weight') and config.retrieval.vector_weight is not None:
                updates['HYBRID_VECTOR_WEIGHT'] = str(config.retrieval.vector_weight)
            if hasattr(config.retrieval, 'bm25_weight') and config.retrieval.bm25_weight is not None:
                updates['HYBRID_BM25_WEIGHT'] = str(config.retrieval.bm25_weight)
            if hasattr(config.retrieval, 'rrf_k') and config.retrieval.rrf_k:
                updates['RRF_K'] = str(config.retrieval.rrf_k)
            if hasattr(config.retrieval, 'min_score_threshold') and config.retrieval.min_score_threshold is not None:
                updates['MIN_SCORE_THRESHOLD'] = str(config.retrieval.min_score_threshold)
            if hasattr(config.retrieval, 'bm25_k1') and config.retrieval.bm25_k1 is not None:
                updates['BM25_K1'] = str(config.retrieval.bm25_k1)
            if hasattr(config.retrieval, 'bm25_b') and config.retrieval.bm25_b is not None:
                updates['BM25_B'] = str(config.retrieval.bm25_b)
        
        # Update generation settings
        if config.generation:
            if hasattr(config.generation, 'model') and config.generation.model:
                updates['GENERATION_MODEL'] = config.generation.model
            if hasattr(config.generation, 'max_tokens') and config.generation.max_tokens:
                updates['GENERATION_MAX_TOKENS'] = str(config.generation.max_tokens)
            if hasattr(config.generation, 'temperature') and config.generation.temperature is not None:
                updates['TEMPERATURE'] = str(config.generation.temperature)
            if hasattr(config.generation, 'enable_hybrid_mode'):
                updates['ENABLE_HYBRID_MODE'] = str(config.generation.enable_hybrid_mode)
        
        # Update channel processing settings
        if config.channel_processing:
            if hasattr(config.channel_processing, 'conversation_gap_minutes') and config.channel_processing.conversation_gap_minutes:
                updates['CONVERSATION_GAP_MINUTES'] = str(config.channel_processing.conversation_gap_minutes)
            if hasattr(config.channel_processing, 'max_group_size') and config.channel_processing.max_group_size:
                updates['MAX_GROUP_SIZE'] = str(config.channel_processing.max_group_size)
            if hasattr(config.channel_processing, 'chunk_overlap') and config.channel_processing.chunk_overlap is not None:
                updates['CHUNK_OVERLAP'] = str(config.channel_processing.chunk_overlap)
            if hasattr(config.channel_processing, 'parent_chunk_size') and config.channel_processing.parent_chunk_size:
                updates['PARENT_CHUNK_SIZE'] = str(config.channel_processing.parent_chunk_size)
            if hasattr(config.channel_processing, 'preserve_threads'):
                updates['PRESERVE_THREADS'] = str(config.channel_processing.preserve_threads)
        
        # Update database settings
        if config.database:
            if hasattr(config.database, 'qdrant_host') and config.database.qdrant_host:
                updates['QDRANT_HOST'] = config.database.qdrant_host
            if hasattr(config.database, 'qdrant_port') and config.database.qdrant_port:
                updates['QDRANT_PORT'] = str(config.database.qdrant_port)
            if hasattr(config.database, 'collection_name') and config.database.collection_name:
                updates['QDRANT_COLLECTION_NAME'] = config.database.collection_name
        
        # Update server settings
        if config.server:
            if hasattr(config.server, 'port') and config.server.port:
                updates['SERVER_PORT'] = str(config.server.port)
        
        # Update system settings
        if config.system:
            if hasattr(config.system, 'log_level') and config.system.log_level:
                updates['LOG_LEVEL'] = config.system.log_level
        
        # Write updates to .env file
        if updates:
            env_manager.update_env_vars(updates)
            logger.info(f"Updated {len(updates)} configuration values in .env file")
            
            # Note: Settings won't be reloaded automatically in the running process
            # A server restart would be needed for all changes to take effect
            return {
                "status": "success",
                "message": f"Updated {len(updates)} configuration values. Some changes may require a server restart to take effect.",
                "updated_keys": list(updates.keys())
            }
        else:
            return {
                "status": "success", 
                "message": "No configuration changes detected",
                "updated_keys": []
            }
        
    except Exception as e:
        logger.error(f"Configuration update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/config")
async def update_system_config(
    config: SystemConfig,
    api_key: Optional[str] = Security(optional_api_key)
):
    """Update system configuration
    
    In production, requires admin API key. In development, authentication is optional.
    """
    
    # Check authentication in production mode
    if settings.ENVIRONMENT == "production" and not api_key:
        raise HTTPException(
            status_code=401,
            detail="Authentication required in production mode"
        )
    
    # In production, verify admin permissions
    if settings.ENVIRONMENT == "production" and api_key:
        key_info = api_key_manager.get_key_info(api_key)
        if not key_info or "admin" not in key_info.get("permissions", []):
            raise HTTPException(
                status_code=403,
                detail="Admin permissions required"
            )
    try:
        from ..utils.env_manager import EnvFileManager
        
        # Initialize environment manager
        env_manager = EnvFileManager()
        
        # Prepare updates dictionary
        updates = {}
        
        # Update API Keys (only if provided and not masked)
        if config.openai_api_key and "..." not in config.openai_api_key:
            updates['OPENAI_API_KEY'] = config.openai_api_key
            
        if config.anthropic_api_key and "..." not in config.anthropic_api_key:
            updates['ANTHROPIC_API_KEY'] = config.anthropic_api_key
        
        # Update Mattermost Settings
        if config.mattermost_url:
            updates['MATTERMOST_URL'] = config.mattermost_url
            
        if config.mattermost_personal_access_token and "..." not in config.mattermost_personal_access_token:
            updates['MATTERMOST_PERSONAL_ACCESS_TOKEN'] = config.mattermost_personal_access_token
            
        if config.mattermost_inject_token and "..." not in config.mattermost_inject_token:
            updates['MATTERMOST_INJECT_TOKEN'] = config.mattermost_inject_token
            
        if config.mattermost_ask_token and "..." not in config.mattermost_ask_token:
            updates['MATTERMOST_ASK_TOKEN'] = config.mattermost_ask_token
        
        # Update embedding settings
        if config.embedding:
            if hasattr(config.embedding, 'model') and config.embedding.model:
                updates['EMBEDDING_MODEL'] = config.embedding.model
            if hasattr(config.embedding, 'max_tokens') and config.embedding.max_tokens:
                updates['CHUNK_SIZE'] = str(config.embedding.max_tokens)
        
        # Update retrieval settings
        if config.retrieval:
            if hasattr(config.retrieval, 'top_k') and config.retrieval.top_k:
                updates['SIMILARITY_TOP_K'] = str(config.retrieval.top_k)
            if hasattr(config.retrieval, 'rerank_top_n') and config.retrieval.rerank_top_n:
                updates['RERANK_TOP_N'] = str(config.retrieval.rerank_top_n)
            if hasattr(config.retrieval, 'mmr_lambda') and config.retrieval.mmr_lambda is not None:
                updates['MMR_LAMBDA'] = str(config.retrieval.mmr_lambda)
            # Advanced retrieval settings
            if hasattr(config.retrieval, 'vector_weight') and config.retrieval.vector_weight is not None:
                updates['HYBRID_VECTOR_WEIGHT'] = str(config.retrieval.vector_weight)
            if hasattr(config.retrieval, 'bm25_weight') and config.retrieval.bm25_weight is not None:
                updates['HYBRID_BM25_WEIGHT'] = str(config.retrieval.bm25_weight)
            if hasattr(config.retrieval, 'rrf_k') and config.retrieval.rrf_k:
                updates['HYBRID_RRF_K'] = str(config.retrieval.rrf_k)
            if hasattr(config.retrieval, 'min_score_threshold') and config.retrieval.min_score_threshold is not None:
                updates['HYBRID_MIN_SCORE_THRESHOLD'] = str(config.retrieval.min_score_threshold)
            if hasattr(config.retrieval, 'bm25_k1') and config.retrieval.bm25_k1 is not None:
                updates['BM25_K1'] = str(config.retrieval.bm25_k1)
            if hasattr(config.retrieval, 'bm25_b') and config.retrieval.bm25_b is not None:
                updates['BM25_B'] = str(config.retrieval.bm25_b)
        
        # Update generation settings
        if config.generation:
            if hasattr(config.generation, 'model') and config.generation.model:
                updates['CLAUDE_MODEL'] = config.generation.model
            if hasattr(config.generation, 'max_tokens') and config.generation.max_tokens:
                updates['MAX_TOKENS'] = str(config.generation.max_tokens)
            if hasattr(config.generation, 'temperature') and config.generation.temperature is not None:
                updates['TEMPERATURE'] = str(config.generation.temperature)
        
        # Update channel processing settings
        if config.channel_processing:
            if hasattr(config.channel_processing, 'conversation_gap_minutes') and config.channel_processing.conversation_gap_minutes:
                updates['CONVERSATION_GAP_MINUTES'] = str(config.channel_processing.conversation_gap_minutes)
            if hasattr(config.channel_processing, 'max_group_size') and config.channel_processing.max_group_size:
                updates['MAX_GROUP_SIZE'] = str(config.channel_processing.max_group_size)
            if hasattr(config.channel_processing, 'chunk_overlap') and config.channel_processing.chunk_overlap is not None:
                updates['CHUNK_OVERLAP'] = str(config.channel_processing.chunk_overlap)
            if hasattr(config.channel_processing, 'parent_chunk_size') and config.channel_processing.parent_chunk_size:
                updates['PARENT_CHUNK_SIZE'] = str(config.channel_processing.parent_chunk_size)
            if hasattr(config.channel_processing, 'preserve_threads') and config.channel_processing.preserve_threads is not None:
                updates['PRESERVE_THREADS'] = str(config.channel_processing.preserve_threads)
        
        # Update database settings
        if config.database:
            if hasattr(config.database, 'qdrant_host') and config.database.qdrant_host:
                updates['QDRANT_HOST'] = config.database.qdrant_host
            if hasattr(config.database, 'qdrant_port') and config.database.qdrant_port:
                updates['QDRANT_PORT'] = str(config.database.qdrant_port)
            if hasattr(config.database, 'collection_name') and config.database.collection_name:
                updates['QDRANT_COLLECTION_NAME'] = config.database.collection_name
        
        # Update server settings
        if config.server:
            if hasattr(config.server, 'port') and config.server.port:
                updates['SERVER_PORT'] = str(config.server.port)
        
        # Update system settings
        if config.system:
            if hasattr(config.system, 'log_level') and config.system.log_level:
                updates['LOG_LEVEL'] = config.system.log_level
        
        # Write updates to .env file
        if updates:
            env_manager.update_env_vars(updates)
            logger.info(f"Updated {len(updates)} configuration values in .env file")
            
            # Note: Settings won't be reloaded automatically in the running process
            # A server restart would be needed for all changes to take effect
            return {
                "status": "success",
                "message": f"Updated {len(updates)} configuration values. Some changes may require a server restart to take effect.",
                "updated_keys": list(updates.keys())
            }
        else:
            return {
                "status": "success", 
                "message": "No configuration changes detected",
                "updated_keys": []
            }
        
    except Exception as e:
        logger.error(f"Configuration update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/purge")
async def purge_database(
    filters: Optional[Dict[str, Any]] = None,
    preview_only: bool = True,
    confirm_purge: bool = False,
    api_key: str = Security(require_admin)
):
    """Purge database with safety checks"""
    
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    try:
        result = await rag_pipeline.purge_database(
            filters=filters,
            preview_only=preview_only,
            confirm_purge=confirm_purge
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Database purge failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task functions
async def _ingest_url_task(task_id: str, request: UrlIngestionRequest):
    """Background task for URL ingestion"""
    
    try:
        # Update task status
        background_tasks[task_id]["status"] = "processing"
        background_tasks[task_id]["started_at"] = datetime.now()
        
        # Perform ingestion
        result = await rag_pipeline.ingest_url(
            url=str(request.url),
            metadata={
                "channel_id": request.channel_id,
                "team_id": request.team_id,
                "ingested_by": request.user_id,
                "source_type": "url"
            }
        )
        
        # Update task status
        background_tasks[task_id]["status"] = "completed"
        background_tasks[task_id]["completed_at"] = datetime.now()
        background_tasks[task_id]["result"] = result
        
    except Exception as e:
        logger.error(f"URL ingestion task failed: {e}")
        background_tasks[task_id]["status"] = "failed"
        background_tasks[task_id]["error"] = str(e)
        background_tasks[task_id]["completed_at"] = datetime.now()


async def _ingest_channel_task(task_id: str, request: ChannelIngestionRequest):
    """Background task for channel ingestion"""
    
    try:
        # Update task status
        background_tasks[task_id]["status"] = "processing"
        background_tasks[task_id]["started_at"] = datetime.now()
        
        # Perform ingestion
        result = await rag_pipeline.ingest_channel(
            channel_id=request.channel_id,
            team_id=request.team_id,
            max_messages=request.max_messages,
            metadata={
                "ingested_by": request.user_id,
                "source_type": "channel_history"
            }
        )
        
        # Update task status
        background_tasks[task_id]["status"] = "completed"
        background_tasks[task_id]["completed_at"] = datetime.now()
        background_tasks[task_id]["result"] = result
        
    except Exception as e:
        logger.error(f"Channel ingestion task failed: {e}")
        background_tasks[task_id]["status"] = "failed"
        background_tasks[task_id]["error"] = str(e)
        background_tasks[task_id]["completed_at"] = datetime.now()


# Cache management endpoints
@app.get("/api/cache/stats")
async def get_cache_stats(
    api_key: str = Security(get_api_key)
):
    """Get cache statistics"""
    from ..cache import cache_manager
    
    if not cache_manager:
        return {"status": "disabled", "message": "Cache is not enabled"}
    
    stats = await cache_manager.get_stats()
    return stats


@app.post("/api/cache/invalidate")
async def invalidate_cache(
    query: Optional[str] = None,
    pattern: Optional[str] = None,
    all: bool = False,
    api_key: str = Security(require_admin)
):
    """Invalidate cache entries (requires admin permissions)"""
    from ..cache import cache_manager
    
    if not cache_manager:
        raise HTTPException(status_code=503, detail="Cache is not enabled")
    
    if all:
        # Invalidate all cache
        count = await cache_manager.invalidate()
        return {"message": f"Invalidated all {count} cache entries"}
    elif query:
        # Invalidate specific query
        count = await cache_manager.invalidate(query=query)
        return {"message": f"Invalidated {count} cache entries for query"}
    elif pattern:
        # Invalidate by pattern
        count = await cache_manager.invalidate(pattern=pattern)
        return {"message": f"Invalidated {count} cache entries matching pattern"}
    else:
        raise HTTPException(
            status_code=400,
            detail="Must specify either 'query', 'pattern', or 'all=true'"
        )


# BM25 index management endpoints
@app.post("/api/bm25/rebuild")
async def rebuild_bm25_index(
    api_key: str = Security(require_admin)
):
    """Rebuild BM25 index from scratch (requires admin permissions)"""
    
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    try:
        # Access the BM25 retriever through hybrid retriever
        if hasattr(rag_pipeline, 'hybrid_retriever') and rag_pipeline.hybrid_retriever:
            bm25_retriever = rag_pipeline.hybrid_retriever.bm25_retriever
            success = await bm25_retriever.force_rebuild()
            
            if success:
                return {
                    "status": "success",
                    "message": "BM25 index rebuilt successfully",
                    "documents": bm25_retriever.total_documents
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to rebuild BM25 index")
        else:
            raise HTTPException(status_code=503, detail="BM25 retriever not available")
            
    except Exception as e:
        logger.error(f"BM25 rebuild failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bm25/stats")
async def get_bm25_stats(
    api_key: str = Security(get_api_key)
):
    """Get BM25 index statistics"""
    
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    try:
        if hasattr(rag_pipeline, 'hybrid_retriever') and rag_pipeline.hybrid_retriever:
            bm25_retriever = rag_pipeline.hybrid_retriever.bm25_retriever
            
            return {
                "status": "active",
                "total_documents": bm25_retriever.total_documents,
                "vocabulary_size": len(bm25_retriever.term_document_frequency),
                "average_document_length": round(bm25_retriever.average_document_length, 2),
                "index_checksum": bm25_retriever.index_checksum,
                "index_timestamp": bm25_retriever.index_timestamp.isoformat() if bm25_retriever.index_timestamp else None,
                "index_path": str(bm25_retriever.index_path),
                "parameters": {
                    "k1": bm25_retriever.k1,
                    "b": bm25_retriever.b
                }
            }
        else:
            return {"status": "disabled", "message": "BM25 retriever not available"}
            
    except Exception as e:
        logger.error(f"Failed to get BM25 stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Monitoring Dashboard endpoints
@app.get("/monitor", response_class=HTMLResponse)
async def get_monitoring_dashboard():
    """Main monitoring dashboard page"""
    if not monitoring_dashboard:
        raise HTTPException(status_code=503, detail="Monitoring dashboard not available")
    
    return HTMLResponse(content=monitoring_dashboard._generate_dashboard_html())


@app.get("/api/monitor/health")
async def monitoring_health_check():
    """Health check endpoint for monitoring dashboard"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/monitor/status")
async def get_monitoring_status():
    """Get monitoring system status"""
    if not monitoring_manager:
        raise HTTPException(status_code=503, detail="Monitoring manager not available")
    
    return monitoring_manager.get_health_status()


@app.get("/api/monitor/report")
async def get_monitoring_report():
    """Get comprehensive monitoring report"""
    if not monitoring_manager:
        raise HTTPException(status_code=503, detail="Monitoring manager not available")
    
    return monitoring_manager.get_comprehensive_report()


@app.get("/api/monitor/performance")
async def get_performance_summary(hours: int = 24):
    """Get performance summary"""
    if not monitoring_manager or not monitoring_manager.performance_monitor:
        return {"error": "Performance monitoring not enabled"}
    
    return monitoring_manager.performance_monitor.get_performance_summary(hours)


@app.get("/api/monitor/cost")
async def get_cost_summary():
    """Get cost summary"""
    if not monitoring_manager or not monitoring_manager.cost_tracker:
        return {"error": "Cost tracking not enabled"}
    
    return monitoring_manager.cost_tracker.get_cost_summary()


@app.get("/api/monitor/cost/projection")
async def get_cost_projection(days_ahead: int = 30):
    """Get cost projection"""
    if not monitoring_manager or not monitoring_manager.cost_tracker:
        return {"error": "Cost tracking not enabled"}
    
    from dataclasses import asdict
    projection = monitoring_manager.cost_tracker.get_cost_projection(days_ahead)
    return asdict(projection)


@app.get("/api/monitor/evaluation")
async def get_evaluation_metrics():
    """Get evaluation metrics"""
    if not monitoring_manager or not monitoring_manager.evaluation_framework:
        return {"error": "Evaluation framework not enabled"}
    
    from dataclasses import asdict
    metrics = monitoring_manager.evaluation_framework.generate_evaluation_report()
    return asdict(metrics)


@app.post("/api/monitor/export")
async def export_monitoring_data():
    """Export monitoring data"""
    if not monitoring_manager:
        raise HTTPException(status_code=503, detail="Monitoring manager not available")
    
    try:
        filepath = monitoring_manager.save_monitoring_report()
        return {"success": True, "filepath": filepath}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Circuit breaker monitoring endpoints
@app.get("/api/circuit-breakers/stats")
async def get_circuit_breaker_stats(
    api_key: str = Security(get_api_key)
):
    """Get circuit breaker statistics"""
    from ..utils.circuit_breaker import get_all_circuit_breakers
    
    stats = get_all_circuit_breakers()
    return {
        "circuit_breakers": stats,
        "summary": {
            "total": len(stats),
            "open": sum(1 for cb in stats.values() if cb["state"] == "open"),
            "closed": sum(1 for cb in stats.values() if cb["state"] == "closed"),
            "half_open": sum(1 for cb in stats.values() if cb["state"] == "half_open")
        }
    }


@app.post("/api/circuit-breakers/{name}/reset")
async def reset_circuit_breaker(
    name: str,
    api_key: str = Security(require_admin)
):
    """Reset a specific circuit breaker (requires admin permissions)"""
    from ..utils.circuit_breaker import _circuit_breakers, CircuitState
    
    if name not in _circuit_breakers:
        raise HTTPException(status_code=404, detail=f"Circuit breaker '{name}' not found")
    
    breaker = _circuit_breakers[name]
    await breaker._change_state(CircuitState.CLOSED, "Manual reset via API")
    
    return {
        "status": "success",
        "message": f"Circuit breaker '{name}' has been reset",
        "new_state": "closed"
    }


# Webhook Management Endpoints

@app.get("/api/webhooks/stats")
async def get_webhook_stats(api_key: str = Depends(get_api_key)):
    """Get webhook delivery statistics"""
    try:
        stats = webhook_service.get_delivery_statistics()
        return {
            "status": "success",
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get webhook stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get webhook statistics")


@app.get("/api/webhooks/history")
async def get_webhook_history(
    limit: int = 100,
    status_filter: Optional[str] = None,
    endpoint_filter: Optional[str] = None,
    api_key: str = Depends(get_api_key)
):
    """Get webhook delivery history"""
    try:
        history = webhook_service.get_delivery_history(
            limit=limit,
            status_filter=status_filter,
            endpoint_filter=endpoint_filter
        )
        
        # Convert to serializable format
        history_data = []
        for delivery in history:
            history_data.append({
                "delivery_id": delivery.delivery_id,
                "webhook_url": delivery.webhook_url,
                "event_type": delivery.payload.event_type.value,
                "task_id": delivery.payload.task_id,
                "attempt": delivery.attempt,
                "status": delivery.status,
                "response_status": delivery.response_status,
                "duration_ms": delivery.duration_ms,
                "error_message": delivery.error_message,
                "timestamp": delivery.timestamp.isoformat()
            })
        
        return {
            "status": "success",
            "history": history_data,
            "total_count": len(history_data)
        }
    except Exception as e:
        logger.error(f"Failed to get webhook history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get webhook history")


@app.get("/api/webhooks/delivery/{delivery_id}")
async def get_webhook_delivery(
    delivery_id: str,
    api_key: str = Depends(get_api_key)
):
    """Get specific webhook delivery details"""
    try:
        delivery = webhook_service.get_delivery_status(delivery_id)
        
        if not delivery:
            raise HTTPException(status_code=404, detail="Webhook delivery not found")
        
        return {
            "status": "success",
            "delivery": {
                "delivery_id": delivery.delivery_id,
                "webhook_url": delivery.webhook_url,
                "payload": {
                    "event_type": delivery.payload.event_type.value,
                    "task_id": delivery.payload.task_id,
                    "source_type": delivery.payload.source_type,
                    "status": delivery.payload.status,
                    "progress_percentage": delivery.payload.progress_percentage,
                    "current_step": delivery.payload.current_step,
                    "timestamp": delivery.payload.timestamp
                },
                "attempt": delivery.attempt,
                "status": delivery.status,
                "response_status": delivery.response_status,
                "response_body": delivery.response_body,
                "duration_ms": delivery.duration_ms,
                "error_message": delivery.error_message,
                "timestamp": delivery.timestamp.isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get webhook delivery: {e}")
        raise HTTPException(status_code=500, detail="Failed to get webhook delivery")


@app.post("/api/webhooks/test")
async def test_webhook_notification(
    api_key: str = Depends(require_admin)
):
    """Send a test webhook notification to all configured endpoints"""
    try:
        from ..monitoring.webhook_notifications import WebhookEventType
        from ..ingestion.pipeline import IngestionTask, IngestionStatus, SourceType
        
        # Create a test task
        test_task = IngestionTask(
            task_id=f"test_{uuid.uuid4().hex[:8]}",
            source_type=SourceType.TEXT_CONTENT,
            source_identifier="webhook_test",
            status=IngestionStatus.COMPLETED,
            progress_percentage=100.0,
            current_step="Test completed",
            total_steps=1,
            completed_steps=1,
            metadata={
                "test": True,
                "purpose": "Webhook endpoint testing"
            }
        )
        
        # Send test notification
        deliveries = await webhook_service.send_notification(test_task, WebhookEventType.INGESTION_COMPLETED)
        
        return {
            "status": "success",
            "message": "Test webhook notifications sent",
            "delivery_count": len(deliveries),
            "deliveries": [d.delivery_id for d in deliveries]
        }
        
    except Exception as e:
        logger.error(f"Failed to send test webhook: {e}")
        raise HTTPException(status_code=500, detail="Failed to send test webhook")


@app.post("/api/webhooks/health")
async def send_health_webhook(
    api_key: str = Depends(require_admin)
):
    """Send system health webhook notification"""
    try:
        # Get current health data
        if rag_pipeline:
            health_data = await rag_pipeline.get_health_status()
        else:
            health_data = {
                "overall_healthy": False,
                "message": "RAG pipeline not initialized"
            }
        
        # Send health notification
        deliveries = await webhook_service.send_health_notification(health_data)
        
        return {
            "status": "success",
            "message": "Health webhook notifications sent",
            "delivery_count": len(deliveries),
            "deliveries": [d.delivery_id for d in deliveries]
        }
        
    except Exception as e:
        logger.error(f"Failed to send health webhook: {e}")
        raise HTTPException(status_code=500, detail="Failed to send health webhook")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error="not_found",
            message="The requested resource was not found"
        ).model_dump(mode='json')
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error",
            message="An internal server error occurred"
        ).model_dump(mode='json')
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.endpoints:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=settings.ENVIRONMENT == "development"
    )