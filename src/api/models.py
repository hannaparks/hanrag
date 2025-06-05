from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import uuid


class QueryType(str, Enum):
    """Types of queries"""
    SIMPLE_FACTUAL = "simple_factual"
    COMPLEX_ANALYTICAL = "complex_analytical"
    MULTI_FACETED = "multi_faceted"
    PROCEDURAL = "procedural"
    COMPARISON = "comparison"
    TEMPORAL = "temporal"


class IngestionStatus(str, Enum):
    """Status of ingestion process"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Slash Command Models
class SlashCommandBase(BaseModel):
    """Base model for slash command requests"""
    token: str
    team_id: str
    team_domain: str
    channel_id: str
    channel_name: str
    user_id: str
    user_name: str
    command: str
    text: str = ""
    response_url: str
    trigger_id: str


class InjectCommandRequest(SlashCommandBase):
    """Model for /inject command request"""


class AskCommandRequest(SlashCommandBase):
    """Model for /ask command request"""


class SlashCommandResponse(BaseModel):
    """Model for slash command response"""
    response_type: str = Field(..., description="ephemeral or in_channel")
    text: str = Field(..., description="Response text in Markdown")
    username: Optional[str] = Field(default="RAG Assistant", description="Bot username")
    icon_emoji: Optional[str] = Field(default=":robot_face:", description="Bot icon")
    attachments: Optional[List[Dict[str, Any]]] = Field(default=None, description="Rich attachments")


# Query Models
class QueryRequest(BaseModel):
    """Model for query requests"""
    question: str = Field(..., description="The question to ask")
    channel_id: Optional[str] = Field(default=None, description="Channel context (not used for filtering)")
    team_id: Optional[str] = Field(default=None, description="Team context (not used for filtering)")
    user_id: Optional[str] = Field(default=None, description="User making the request")
    max_results: Optional[int] = Field(default=50, description="Maximum results to return")
    use_mmr: Optional[bool] = Field(default=True, description="Use MMR for diversity")
    temperature: Optional[float] = Field(default=0.1, description="Response creativity")


class QueryResponse(BaseModel):
    """Model for query responses"""
    response: str = Field(..., description="Generated response")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents used")
    context_count: int = Field(..., description="Number of context sources")
    query_type: QueryType = Field(..., description="Detected query type")
    processing_time: float = Field(..., description="Processing time in seconds")
    usage: Optional[Dict[str, Any]] = Field(default=None, description="Token usage stats")


# Conversation History Models
class ConversationTurn(BaseModel):
    """Model for a single turn in a conversation"""
    role: str = Field(..., description="Role (user or assistant)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Turn timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class ConversationHistory(BaseModel):
    """Model for conversation history"""
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique conversation ID")
    turns: List[ConversationTurn] = Field(default_factory=list, description="Conversation turns")
    created_at: datetime = Field(default_factory=datetime.now, description="Conversation start time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    
    def add_turn(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a turn to the conversation"""
        turn = ConversationTurn(role=role, content=content, metadata=metadata)
        self.turns.append(turn)
        self.updated_at = datetime.now()
    
    def get_context_window(self, max_turns: int = 10) -> List[ConversationTurn]:
        """Get the most recent turns within the context window"""
        return self.turns[-max_turns:] if len(self.turns) > max_turns else self.turns
    
    def to_messages_format(self) -> List[Dict[str, str]]:
        """Convert to Claude/OpenAI messages format"""
        return [{"role": turn.role, "content": turn.content} for turn in self.turns]


class ConversationalQueryRequest(BaseModel):
    """Model for conversational query requests with history"""
    question: str = Field(..., description="The current question to ask")
    conversation_id: Optional[str] = Field(default=None, description="Existing conversation ID to continue")
    conversation_history: Optional[List[ConversationTurn]] = Field(default=None, description="Previous conversation turns")
    max_history_turns: Optional[int] = Field(default=10, description="Maximum history turns to include")
    channel_id: Optional[str] = Field(default=None, description="Channel context")
    team_id: Optional[str] = Field(default=None, description="Team context")
    user_id: Optional[str] = Field(default=None, description="User making the request")
    max_results: Optional[int] = Field(default=50, description="Maximum results to return")
    use_mmr: Optional[bool] = Field(default=True, description="Use MMR for diversity")
    temperature: Optional[float] = Field(default=0.1, description="Response creativity")
    maintain_context: Optional[bool] = Field(default=True, description="Maintain conversation context")


class ConversationalQueryResponse(BaseModel):
    """Model for conversational query responses"""
    response: str = Field(..., description="Generated response")
    conversation_id: str = Field(..., description="Conversation ID for continuation")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents used")
    context_count: int = Field(..., description="Number of context sources")
    query_type: QueryType = Field(..., description="Detected query type")
    processing_time: float = Field(..., description="Processing time in seconds")
    usage: Optional[Dict[str, Any]] = Field(default=None, description="Token usage stats")
    conversation_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Conversation metadata")


# Ingestion Models
class UrlIngestionRequest(BaseModel):
    """Model for URL ingestion requests"""
    url: HttpUrl = Field(..., description="URL to ingest")
    channel_id: Optional[str] = Field(default=None, description="Associated channel")
    team_id: Optional[str] = Field(default=None, description="Associated team")
    user_id: Optional[str] = Field(default=None, description="User requesting ingestion")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class ChannelIngestionRequest(BaseModel):
    """Model for channel history ingestion requests"""
    channel_id: str = Field(..., description="Channel to ingest from")
    team_id: str = Field(..., description="Team ID")
    user_id: Optional[str] = Field(default=None, description="User requesting ingestion")
    max_messages: Optional[int] = Field(default=1000, description="Maximum messages to ingest")
    before: Optional[str] = Field(default=None, description="Before timestamp")
    after: Optional[str] = Field(default=None, description="After timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class FileIngestionRequest(BaseModel):
    """Model for file ingestion requests"""
    file_path: str = Field(..., description="Path to file to ingest")
    channel_id: Optional[str] = Field(default=None, description="Associated channel")
    team_id: Optional[str] = Field(default=None, description="Associated team")
    user_id: Optional[str] = Field(default=None, description="User requesting ingestion")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class IngestionResponse(BaseModel):
    """Model for ingestion responses"""
    status: IngestionStatus = Field(..., description="Ingestion status")
    task_id: str = Field(..., description="Task ID for tracking")
    message: str = Field(..., description="Status message")
    chunks_processed: Optional[int] = Field(default=None, description="Number of chunks processed")
    total_tokens: Optional[int] = Field(default=None, description="Total tokens processed")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    error: Optional[str] = Field(default=None, description="Error message if failed")


# System Models
class HealthResponse(BaseModel):
    """Model for health check response"""
    status: str = Field(..., description="System status")
    service: str = Field(default="RAG Assistant", description="Service name")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    version: Optional[str] = Field(default="1.0.0", description="Service version")
    components: Optional[Dict[str, str]] = Field(default=None, description="Component statuses")


class StatsResponse(BaseModel):
    """Model for system statistics response"""
    total_documents: int = Field(..., description="Total documents in knowledge base")
    total_chunks: int = Field(..., description="Total content chunks")
    total_teams: int = Field(..., description="Total teams using system")
    total_channels: int = Field(..., description="Total channels with content")
    embedding_dimension: int = Field(..., description="Vector embedding dimension")
    last_updated: Optional[datetime] = Field(default=None, description="Last content update")
    storage_size: Optional[int] = Field(default=None, description="Storage size in bytes")


class SearchRequest(BaseModel):
    """Model for similarity search requests"""
    query: str = Field(..., description="Search query")
    top_k: Optional[int] = Field(default=50, description="Number of results")
    channel_filter: Optional[str] = Field(default=None, description="Channel filter (deprecated - not used)")
    team_filter: Optional[str] = Field(default=None, description="Team filter (deprecated - not used)")
    metadata_filters: Optional[Dict[str, Any]] = Field(default=None, description="Additional filters")


class SearchResult(BaseModel):
    """Model for search result"""
    content: str = Field(..., description="Content text")
    score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(..., description="Content metadata")
    chunk_id: str = Field(..., description="Unique chunk ID")
    source: str = Field(..., description="Source identifier")


class SearchResponse(BaseModel):
    """Model for search response"""
    results: List[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total results found")
    query_time: float = Field(..., description="Query processing time")
    query_type: QueryType = Field(..., description="Detected query type")


# Error Models
class ErrorResponse(BaseModel):
    """Model for error responses"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


class ValidationError(BaseModel):
    """Model for validation errors"""
    field: str = Field(..., description="Field with validation error")
    message: str = Field(..., description="Validation error message")
    value: Any = Field(..., description="Invalid value")


# Configuration Models
class EmbeddingConfig(BaseModel):
    """Model for embedding configuration"""
    model: Optional[str] = Field(default=None, description="Embedding model name")
    dimension: Optional[int] = Field(default=None, description="Vector dimension")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens per chunk")


class RetrievalConfig(BaseModel):
    """Model for retrieval configuration"""
    top_k: int = Field(default=50, description="Top K results")
    rerank_top_n: int = Field(default=25, description="Results to rerank")
    mmr_lambda: float = Field(default=0.7, description="MMR lambda parameter")
    use_hybrid_search: bool = Field(default=True, description="Use hybrid search")
    # Advanced retrieval settings
    vector_weight: Optional[float] = Field(default=0.7, description="Weight for vector search")
    bm25_weight: Optional[float] = Field(default=0.3, description="Weight for BM25 search")
    rrf_k: Optional[int] = Field(default=60, description="Reciprocal Rank Fusion parameter")
    min_score_threshold: Optional[float] = Field(default=0.0, description="Minimum score threshold")
    bm25_k1: Optional[float] = Field(default=1.2, description="BM25 K1 parameter")
    bm25_b: Optional[float] = Field(default=0.75, description="BM25 B parameter")


class GenerationConfig(BaseModel):
    """Model for generation configuration"""
    model: Optional[str] = Field(default=None, description="Generation model name")
    max_tokens: Optional[int] = Field(default=4096, ge=100, le=64000, description="Maximum response tokens (100-64000)")
    temperature: Optional[float] = Field(default=0.1, description="Generation temperature")
    enable_hybrid_mode: Optional[bool] = Field(default=False, description="Enable hybrid mode with extended thinking (Claude 4 only)")


class ChannelProcessingConfig(BaseModel):
    """Model for channel processing configuration"""
    conversation_gap_minutes: int = Field(default=30, description="Gap between conversation groups")
    max_group_size: int = Field(default=25, description="Maximum messages per group")
    chunk_overlap: int = Field(default=64, description="Token overlap between chunks")
    parent_chunk_size: int = Field(default=1024, description="Parent chunk size in tokens")
    preserve_threads: bool = Field(default=True, description="Preserve thread structure")


class DatabaseConfig(BaseModel):
    """Model for database configuration"""
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant port")
    collection_name: str = Field(default="documents", description="Collection name")


class ServerConfig(BaseModel):
    """Model for server configuration"""
    host: Optional[str] = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")


class SystemSettingsConfig(BaseModel):
    """Model for system settings"""
    log_level: str = Field(default="INFO", description="Logging level")


class SystemConfig(BaseModel):
    """Model for system configuration"""
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    
    # Mattermost Settings
    mattermost_url: Optional[str] = Field(default=None, description="Mattermost server URL")
    mattermost_personal_access_token: Optional[str] = Field(default=None, description="Mattermost personal access token")
    mattermost_inject_token: Optional[str] = Field(default=None, description="Mattermost /inject command token")
    mattermost_ask_token: Optional[str] = Field(default=None, description="Mattermost /ask command token")
    
    # RAG Settings
    embedding: Optional[EmbeddingConfig] = Field(default=None, description="Embedding configuration")
    retrieval: Optional[RetrievalConfig] = Field(default=None, description="Retrieval configuration")
    generation: Optional[GenerationConfig] = Field(default=None, description="Generation configuration")
    
    # Additional Settings
    channel_processing: Optional[ChannelProcessingConfig] = Field(default=None, description="Channel processing configuration")
    database: Optional[DatabaseConfig] = Field(default=None, description="Database configuration")
    server: Optional[ServerConfig] = Field(default=None, description="Server configuration")
    system: Optional[SystemSettingsConfig] = Field(default=None, description="System settings")
    
    # Environment
    environment: Optional[str] = Field(default=None, description="Environment (development/production)")


# Task Models
class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(BaseModel):
    """Model for background tasks"""
    task_id: str = Field(..., description="Unique task ID")
    task_type: str = Field(..., description="Type of task")
    status: TaskStatus = Field(..., description="Current status")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    progress: Optional[int] = Field(default=0, description="Progress percentage")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Task result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Task metadata")