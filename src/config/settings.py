from pydantic_settings import BaseSettings
from pydantic import validator
from typing import Optional


class RAGSettings(BaseSettings):
    # Environment
    ENVIRONMENT: str = "development"  # development, production
    
    # Vector Database
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "documents"
    QDRANT_POOL_SIZE: int = 10  # Number of connections in pool
    QDRANT_MAX_IDLE_TIME: int = 300  # Max idle time in seconds before reconnection
    
    # API Keys
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: str
    API_KEY: Optional[str] = None  # Main API key for REST endpoints
    
    # Mattermost Integration
    MATTERMOST_URL: str = "https://hanna-test.test.mattermost.cloud/"  # Can be overridden by env
    MATTERMOST_PERSONAL_TOKEN: str  # Must be set via environment variable
    MATTERMOST_PERSONAL_ACCESS_TOKEN: Optional[str] = None  # Alias for UI compatibility
    
    # Slash Command Tokens - Must be set via environment variables
    MATTERMOST_INJECT_TOKEN: str  # Must be set via environment variable
    MATTERMOST_ASK_TOKEN: str  # Must be set via environment variable
    
    # Server Configuration
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    
    # Production URLs (Hostinger VPS)
    PRODUCTION_HOST: str = "168.231.68.82"
    PRODUCTION_DOMAIN: str = "168.231.68.82"  # Can be updated with domain later
    
    # Retrieval Parameters
    SIMILARITY_TOP_K: int = 50
    RERANK_TOP_N: int = 25
    MMR_LAMBDA: float = 0.7
    
    # Hybrid Search Parameters
    HYBRID_VECTOR_WEIGHT: float = 0.7      # Weight for vector search in fusion
    HYBRID_BM25_WEIGHT: float = 0.3        # Weight for BM25 search in fusion
    HYBRID_RRF_K: int = 60                 # Reciprocal Rank Fusion parameter
    HYBRID_MIN_SCORE_THRESHOLD: float = 0.0  # Minimum score to include result
    
    # BM25 Parameters
    BM25_K1: float = 1.2                   # Controls term frequency saturation
    BM25_B: float = 0.75                   # Controls field length normalization
    BM25_INDEX_PATH: str = "./data/bm25_index.pkl"  # Path to persist BM25 index
    
    # Channel Processing Settings
    CONVERSATION_GAP_MINUTES: int = 30     # Max gap between messages in same conversation group
    MAX_GROUP_SIZE: int = 25               # Max messages per conversation group
    PRESERVE_THREADS: bool = True          # Whether to preserve thread structure in formatting
    
    # Chunk Settings
    CHUNK_SIZE: int = 512                  # Tokens per chunk
    PARENT_CHUNK_SIZE: int = 1024          # Tokens per parent chunk
    CHUNK_OVERLAP: int = 64                # Token overlap between chunks
    
    # Generation Settings
    CLAUDE_MODEL: str = "claude-3-5-sonnet-20241022"
    GENERATION_MODEL: str = "claude-3-5-sonnet-20241022"  # Alias for UI compatibility
    MAX_TOKENS: int = 4096
    GENERATION_MAX_TOKENS: int = 4096  # Alias for UI compatibility
    TEMPERATURE: float = 0.1
    ENABLE_HYBRID_MODE: bool = False  # Enable hybrid mode with extended thinking (Claude 4 only)
    
    # Embedding Settings
    EMBEDDING_MODEL: str = "text-embedding-3-large"  # 3072 dimensions
    EMBEDDING_DIMENSION: int = 3072
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    RATE_LIMIT_BURST_SIZE: int = 10
    EXPENSIVE_RATE_LIMIT_PER_MINUTE: int = 10
    EXPENSIVE_RATE_LIMIT_PER_HOUR: int = 100
    EXPENSIVE_RATE_LIMIT_BURST_SIZE: int = 3
    
    # Redis Cache Configuration
    ENABLE_REDIS_CACHE: bool = True
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    CACHE_TTL: int = 3600  # 1 hour default TTL
    CACHE_MAX_SIZE: int = 10000  # Max number of cached queries
    
    # Prometheus Metrics Configuration
    ENABLE_PROMETHEUS_METRICS: bool = True  # Enable Prometheus metrics export
    METRICS_PORT: int = 9090  # Port for metrics endpoint (if separate server)
    
    # Webhook Configuration
    ENABLE_WEBHOOK_NOTIFICATIONS: bool = True  # Enable webhook notifications
    WEBHOOK_ENDPOINTS: Optional[str] = None  # Comma-separated webhook URLs
    WEBHOOK_TIMEOUT: int = 30  # Webhook timeout in seconds
    WEBHOOK_RETRY_ATTEMPTS: int = 3  # Number of retry attempts
    WEBHOOK_SECRET: Optional[str] = None  # Optional webhook secret for signature verification
    
<<<<<<< HEAD
=======
    # CORS Configuration
    CORS_ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:8080,*.mattermost.cloud"  # Comma-separated allowed origins
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOWED_METHODS: str = "GET,POST,PUT,DELETE,OPTIONS"
    CORS_ALLOWED_HEADERS: str = "Authorization,Content-Type,X-API-Key"
    
    
>>>>>>> 66c74c8
    @validator('MATTERMOST_PERSONAL_ACCESS_TOKEN', always=True)
    def set_mattermost_access_token(cls, v, values):
        """Set MATTERMOST_PERSONAL_ACCESS_TOKEN from MATTERMOST_PERSONAL_TOKEN if not provided"""
        if v is None and 'MATTERMOST_PERSONAL_TOKEN' in values:
            return values['MATTERMOST_PERSONAL_TOKEN']
        return v
    
    @validator('GENERATION_MODEL', always=True, pre=False)
    def set_generation_model(cls, v, values):
        """Set GENERATION_MODEL from env or fall back to CLAUDE_MODEL"""
        if v:
            return v
        # If GENERATION_MODEL not set, use CLAUDE_MODEL
        return values.get('CLAUDE_MODEL', 'claude-3-5-sonnet-20241022')
    
    @validator('GENERATION_MAX_TOKENS', always=True, pre=False)
    def set_generation_max_tokens(cls, v, values):
        """Set GENERATION_MAX_TOKENS from env or fall back to MAX_TOKENS"""
        if v:
            return v
        # If GENERATION_MAX_TOKENS not set, use MAX_TOKENS
        return values.get('MAX_TOKENS', 4096)
    
    @property
    def webhook_base_url(self) -> str:
        """Get the correct webhook base URL based on environment"""
        if self.ENVIRONMENT == "production":
            return f"http://{self.PRODUCTION_HOST}:8000"
        return f"http://localhost:{self.SERVER_PORT}"
    
    class Config:
        env_file = ".env.local"
        env_file_encoding = 'utf-8'
        extra = "ignore"  # Ignore extra fields in env file
        
    @classmethod
    def get_settings(cls, env: str = "development"):
        """Get environment-specific settings"""
        if env == "production":
            return cls(_env_file="/root/newmmrag/.env.production")
        return cls(_env_file=".env.local")


# Global settings instance
import os
env = os.getenv("ENVIRONMENT", "development")
if env == "production":
    settings = RAGSettings(_env_file="/root/newmmrag/.env.production")
else:
    settings = RAGSettings()