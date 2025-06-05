import asyncio
import uuid
<<<<<<< HEAD
from typing import Dict, Any, List, Optional, Union, Callable
=======
from typing import Dict, Any, List, Optional, Union, Callable, AsyncGenerator, Tuple
>>>>>>> 66c74c8
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import aiohttp
from urllib.parse import urlparse
from loguru import logger

from .channel_processor import ChannelMessageProcessor
from .parsers.document_parser import DocumentParser, ParsedDocument
from .chunking.text_chunker import TextChunker
from .format_detector import FormatDetector
from .error_recovery import ErrorRecoveryManager, ErrorRetryConfig
from .content_quality import ContentQualityManager, ContentQualityMetrics, DuplicateMatch
from ..storage.qdrant_client import QdrantManager
from ..retrieval.retrievers.vector_retriever import VectorRetriever
<<<<<<< HEAD
=======
from ..config.settings import settings
>>>>>>> 66c74c8
from ..monitoring.webhook_notifications import (
    webhook_service, notify_ingestion_started, notify_ingestion_progress,
    notify_ingestion_completed, notify_ingestion_failed, notify_ingestion_cancelled
)


class IngestionStatus(Enum):
    """Ingestion status enumeration"""
    PENDING = "pending"
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SourceType(Enum):
    """Source type enumeration"""
    MATTERMOST_CHANNEL = "mattermost_channel"
    LOCAL_FILE = "local_file"
    URL = "url"
    DIRECTORY = "directory"
    TEXT_CONTENT = "text_content"


@dataclass
class IngestionTask:
    """Represents an ingestion task"""
    task_id: str
    source_type: SourceType
    source_identifier: str  # channel_id, file_path, url, etc.
    status: IngestionStatus = IngestionStatus.PENDING
    progress_percentage: float = 0.0
    current_step: str = ""
    total_steps: int = 0
    completed_steps: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    quality_metrics: Optional[ContentQualityMetrics] = None
    duplicates_found: List[DuplicateMatch] = field(default_factory=list)
    quality_recommendations: List[str] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate task duration in seconds"""
        if self.start_time:
            end = self.end_time or datetime.now()
            return (end - self.start_time).total_seconds()
        return None
    
    def update_progress(self, step: str, completed_steps: int = None):
        """Update task progress"""
        self.current_step = step
        if completed_steps is not None:
            self.completed_steps = completed_steps
        
        if self.total_steps > 0:
            self.progress_percentage = (self.completed_steps / self.total_steps) * 100
        
        logger.info(f"Task {self.task_id}: {step} ({self.progress_percentage:.1f}%)")


@dataclass
class IngestionConfig:
    """Configuration for ingestion pipeline"""
    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 64
    parent_chunk_size: int = 1024
    
    # Processing settings
    max_concurrent_tasks: int = 3
    batch_size: int = 50
    retry_attempts: int = 3
    timeout_seconds: int = 300
    
    # Quality controls
    min_chunk_length: int = 50
    max_chunk_length: int = 8000
    content_quality_threshold: float = 0.4  # Lowered from 0.7 for better content acceptance
    enable_quality_assessment: bool = True
    enable_duplicate_detection: bool = True
    duplicate_similarity_threshold: float = 0.85
    skip_low_quality_content: bool = True
    skip_exact_duplicates: bool = True
    
    # Source-specific quality thresholds
    quality_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "mattermost_channel": 0.3,  # More lenient for conversations
        "url_webpage": 0.4,         # Gaming/reference sites may have lower scores
        "pdf_document": 0.6,        # Higher standard for documents
        "markdown_file": 0.5,       # Documentation standard
        "code_file": 0.3           # Code may score lower on readability
    })
    
    # Source-specific settings
    max_file_size_mb: int = 100
    supported_file_types: List[str] = field(default_factory=lambda: [
        '.pdf', '.docx', '.txt', '.md', '.csv', '.xlsx', '.json', '.html', '.py', '.js'
    ])
    
    # Web scraping settings
    user_agent: str = "RAG-Pipeline/1.0"
    request_timeout: int = 30
    max_redirects: int = 5


class ProgressMonitor:
    """Monitor and report ingestion progress"""
    
    def __init__(self):
        self.tasks: Dict[str, IngestionTask] = {}
        self.callbacks: List[Callable[[IngestionTask], None]] = []
    
    def register_callback(self, callback: Callable[[IngestionTask], None]):
        """Register a progress callback"""
        self.callbacks.append(callback)
    
    def update_task(self, task: IngestionTask):
        """Update task and notify callbacks"""
        self.tasks[task.task_id] = task
        
        for callback in self.callbacks:
            try:
                callback(task)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
        
        # Send webhook notifications for status changes
        self._send_webhook_notification(task)
    
    def get_task_status(self, task_id: str) -> Optional[IngestionTask]:
        """Get task status"""
        return self.tasks.get(task_id)
    
    def get_active_tasks(self) -> List[IngestionTask]:
        """Get all active tasks"""
        return [
            task for task in self.tasks.values()
            if task.status not in [IngestionStatus.COMPLETED, IngestionStatus.FAILED, IngestionStatus.CANCELLED]
        ]
    
    def get_completed_tasks(self) -> List[IngestionTask]:
        """Get completed tasks"""
        return [
            task for task in self.tasks.values()
            if task.status == IngestionStatus.COMPLETED
        ]
    
    def _send_webhook_notification(self, task: IngestionTask):
        """Send webhook notification for task status change"""
        try:
            # Create async task for webhook notification (fire and forget)
            if task.status == IngestionStatus.INITIALIZING:
                asyncio.create_task(notify_ingestion_started(task))
            elif task.status == IngestionStatus.PROCESSING and task.progress_percentage > 0:
                asyncio.create_task(notify_ingestion_progress(task))
            elif task.status == IngestionStatus.COMPLETED:
                asyncio.create_task(notify_ingestion_completed(task))
            elif task.status == IngestionStatus.FAILED:
                asyncio.create_task(notify_ingestion_failed(task))
            elif task.status == IngestionStatus.CANCELLED:
                asyncio.create_task(notify_ingestion_cancelled(task))
        except Exception as e:
            logger.warning(f"Failed to send webhook notification: {e}")


class IngestionPipeline:
    """Main ingestion pipeline orchestrator"""
    
    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig()
        self.progress_monitor = ProgressMonitor()
        
        # Initialize components
        self.channel_processor = ChannelMessageProcessor()
        self.document_parser = DocumentParser()
        self.text_chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.format_detector = FormatDetector()
        self.error_recovery = ErrorRecoveryManager()
        self.quality_manager = ContentQualityManager(
            similarity_threshold=self.config.duplicate_similarity_threshold
        )
        self.qdrant_manager = QdrantManager()
        self.vector_retriever = VectorRetriever()
        
        # Task management
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        
        # Initialize webhook service
        self._webhook_initialized = False
    
    async def _ensure_webhook_initialized(self):
        """Ensure webhook service is initialized"""
        if not self._webhook_initialized:
            await webhook_service.start()
            self._webhook_initialized = True
    
    async def ingest_mattermost_channel(
        self,
        channel_id: str,
        team_id: str,
        max_messages: int = 1000,
        task_id: Optional[str] = None
    ) -> str:
        """Ingest Mattermost channel content"""
        
        await self._ensure_webhook_initialized()
        task_id = task_id or f"channel_{channel_id}_{uuid.uuid4().hex[:8]}"
        
        task = IngestionTask(
            task_id=task_id,
            source_type=SourceType.MATTERMOST_CHANNEL,
            source_identifier=f"{team_id}/{channel_id}",
            total_steps=5,
            metadata={
                "channel_id": channel_id,
                "team_id": team_id,
                "max_messages": max_messages
            }
        )
        
        # Start async processing
        async_task = asyncio.create_task(
            self._process_channel_task(task, channel_id, team_id, max_messages)
        )
        self.active_tasks[task_id] = async_task
        
        return task_id
    
    async def ingest_file(
        self,
        file_path: Union[str, Path],
        task_id: Optional[str] = None
    ) -> str:
        """Ingest local file"""
        
        await self._ensure_webhook_initialized()
        file_path = Path(file_path)
        task_id = task_id or f"file_{file_path.stem}_{uuid.uuid4().hex[:8]}"
        
        task = IngestionTask(
            task_id=task_id,
            source_type=SourceType.LOCAL_FILE,
            source_identifier=str(file_path),
            total_steps=6,
            metadata={
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size if file_path.exists() else 0
            }
        )
        
        # Start async processing
        async_task = asyncio.create_task(
            self._process_file_task(task, file_path)
        )
        self.active_tasks[task_id] = async_task
        
        return task_id
    
    async def ingest_url(
        self,
        url: str,
        task_id: Optional[str] = None
    ) -> str:
        """Ingest content from URL"""
        
        await self._ensure_webhook_initialized()
        task_id = task_id or f"url_{urlparse(url).netloc}_{uuid.uuid4().hex[:8]}"
        
        task = IngestionTask(
            task_id=task_id,
            source_type=SourceType.URL,
            source_identifier=url,
            total_steps=7,
            metadata={
                "url": url,
                "domain": urlparse(url).netloc
            }
        )
        
        # Start async processing
        async_task = asyncio.create_task(
            self._process_url_task(task, url)
        )
        self.active_tasks[task_id] = async_task
        
        return task_id
    
    async def ingest_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        task_id: Optional[str] = None
    ) -> str:
        """Ingest all supported files in directory"""
        
        await self._ensure_webhook_initialized()
        directory_path = Path(directory_path)
        task_id = task_id or f"dir_{directory_path.name}_{uuid.uuid4().hex[:8]}"
        
        # Count files first
        file_pattern = "**/*" if recursive else "*"
        files = [
            f for f in directory_path.glob(file_pattern)
            if f.is_file() and f.suffix.lower() in self.config.supported_file_types
        ]
        
        task = IngestionTask(
            task_id=task_id,
            source_type=SourceType.DIRECTORY,
            source_identifier=str(directory_path),
            total_steps=len(files),
            metadata={
                "directory_path": str(directory_path),
                "file_count": len(files),
                "recursive": recursive
            }
        )
        
        # Start async processing
        async_task = asyncio.create_task(
            self._process_directory_task(task, directory_path, files)
        )
        self.active_tasks[task_id] = async_task
        
        return task_id
    
    async def ingest_text_content(
        self,
        content: str,
        source_name: str,
        content_type: str = "text",
        task_id: Optional[str] = None
    ) -> str:
        """Ingest raw text content"""
        
        await self._ensure_webhook_initialized()
        task_id = task_id or f"text_{source_name}_{uuid.uuid4().hex[:8]}"
        
        task = IngestionTask(
            task_id=task_id,
            source_type=SourceType.TEXT_CONTENT,
            source_identifier=source_name,
            total_steps=4,
            metadata={
                "source_name": source_name,
                "content_type": content_type,
                "content_length": len(content)
            }
        )
        
        # Start async processing
        async_task = asyncio.create_task(
            self._process_text_content_task(task, content, source_name, content_type)
        )
        self.active_tasks[task_id] = async_task
        
        return task_id
    
    # Task execution methods
    
    async def _process_channel_task(
        self,
        task: IngestionTask,
        channel_id: str,
        team_id: str,
        max_messages: int
    ):
        """Process Mattermost channel task"""
        
        async with self.semaphore:
            try:
                task.status = IngestionStatus.INITIALIZING
                task.start_time = datetime.now()
                task.update_progress("Initializing channel processing", 0)
                self.progress_monitor.update_task(task)
                
                # Use existing channel processor
                task.update_progress("Processing channel messages", 1)
                task.status = IngestionStatus.PROCESSING
                self.progress_monitor.update_task(task)
                
                result = await self.channel_processor.process_channel_complete(
                    channel_id=channel_id,
                    team_id=team_id,
                    max_messages=max_messages
                )
                
                if result.get("success"):
                    task.update_progress("Channel processing completed", 5)
                    task.status = IngestionStatus.COMPLETED
                    task.result = result
                else:
                    task.status = IngestionStatus.FAILED
                    task.error_message = result.get("error", "Unknown error")
                
                task.end_time = datetime.now()
                self.progress_monitor.update_task(task)
                
            except Exception as e:
                logger.error(f"Channel task failed: {e}")
                task.status = IngestionStatus.FAILED
                task.error_message = str(e)
                task.end_time = datetime.now()
                self.progress_monitor.update_task(task)
            finally:
                self.active_tasks.pop(task.task_id, None)
    
    async def _process_file_task(self, task: IngestionTask, file_path: Path):
        """Process file ingestion task with format detection and error recovery"""
        
        async with self.semaphore:
            try:
                task.status = IngestionStatus.INITIALIZING
                task.start_time = datetime.now()
                task.update_progress("Initializing file processing", 0)
                self.progress_monitor.update_task(task)
                
                # Format detection
                task.update_progress("Detecting file format", 1)
                self.progress_monitor.update_task(task)
                
                format_result = await self.error_recovery.execute_with_retry(
                    self.format_detector.detect_file_format,
                    f"format_detection_{task.task_id}",
                    {"file_path": str(file_path)},
                    file_path
                )
                
                logger.info(f"Detected format: {format_result.format_type} (confidence: {format_result.confidence:.2f})")
                
                # Validate file
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")
                
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                if file_size_mb > self.config.max_file_size_mb:
                    raise ValueError(f"File too large: {file_size_mb:.1f}MB > {self.config.max_file_size_mb}MB")
                
                # Check if format is supported
                if not self.format_detector.is_supported_format(format_result.format_type):
                    logger.warning(f"Unsupported format {format_result.format_type}, attempting text processing")
                
                # Parse document with retry
                task.update_progress("Parsing document", 2)
                task.status = IngestionStatus.PROCESSING
                self.progress_monitor.update_task(task)
                
                parsed_doc = await self.error_recovery.execute_with_retry(
                    self.document_parser.parse_file,
                    f"parse_document_{task.task_id}",
                    {
                        "file_path": str(file_path),
                        "format_type": format_result.format_type,
                        "file_size_mb": file_size_mb
                    },
                    file_path
                )
                
                if parsed_doc.error:
                    raise Exception(f"Document parsing failed: {parsed_doc.error}")
                
                # Add format detection metadata
                parsed_doc.metadata.update({
                    "detected_format": format_result.format_type,
                    "format_confidence": format_result.confidence,
                    "format_metadata": format_result.metadata,
                    "processing_strategy": format_result.metadata.get("processing_strategy")
                })
                
                # Process parsed content
                await self._process_parsed_document(task, parsed_doc)
                
            except Exception as e:
                logger.error(f"File task failed: {e}")
                task.status = IngestionStatus.FAILED
                task.error_message = str(e)
                task.end_time = datetime.now()
                self.progress_monitor.update_task(task)
            finally:
                self.active_tasks.pop(task.task_id, None)
    
    async def _process_url_task(self, task: IngestionTask, url: str):
        """Process URL ingestion task with format detection and error recovery"""
        
        async with self.semaphore:
            try:
                task.status = IngestionStatus.INITIALIZING
                task.start_time = datetime.now()
                task.update_progress("Fetching URL content", 0)
                self.progress_monitor.update_task(task)
                
                # Fetch URL content with retry
                task.update_progress("Downloading content", 1)
                self.progress_monitor.update_task(task)
                
                async def fetch_url():
<<<<<<< HEAD
=======
                    validated_url = url
                    logger.info(f"Fetching URL: {url}")
                    
>>>>>>> 66c74c8
                    async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=self.config.request_timeout),
                        headers={"User-Agent": self.config.user_agent}
                    ) as session:
                        async with session.get(url, max_redirects=self.config.max_redirects) as response:
                            if response.status != 200:
                                raise Exception(f"HTTP {response.status}: {response.reason}")
                            
                            content_type = response.content_type or "text/html"
                            
                            # Handle binary vs text content properly
                            if any(binary_type in content_type.lower() for binary_type in [
                                'application/pdf', 'application/octet-stream', 'application/msword',
                                'application/vnd.openxmlformats', 'application/vnd.ms-excel'
                            ]) or url.lower().endswith(('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx')):
                                # Binary content - return raw bytes
                                content = await response.read()
                                logger.info(f"Downloaded binary content: {len(content)} bytes, type: {content_type}")
                            else:
                                # Text content with encoding fallback
                                try:
                                    content = await response.text()
                                except UnicodeDecodeError:
                                    # Fallback for text with encoding issues
                                    raw_content = await response.read()
                                    content = raw_content.decode('latin-1', errors='ignore')
                                    logger.warning(f"Used latin-1 fallback for URL content")
                            
                            return content, content_type
                
                content, content_type = await self.error_recovery.execute_with_retry(
                    fetch_url,
                    f"fetch_url_{task.task_id}",
                    {"url": url, "user_agent": self.config.user_agent}
                )
                
                # Format detection for URL
                task.update_progress("Detecting content format", 2)
                self.progress_monitor.update_task(task)
                
                format_result = await self.error_recovery.execute_with_retry(
                    self.format_detector.detect_url_format,
                    f"url_format_detection_{task.task_id}",
                    {"url": url, "content_type": content_type},
                    url,
                    content_type
                )
                
                logger.info(f"URL format detected: {format_result.format_type} (confidence: {format_result.confidence:.2f})")
                
                # Parse URL content with retry
                task.update_progress("Parsing content", 3)
                task.status = IngestionStatus.PROCESSING
                self.progress_monitor.update_task(task)
                
                parsed_doc = await self.error_recovery.execute_with_retry(
                    self.document_parser.parse_url_content,
                    f"parse_url_content_{task.task_id}",
                    {
                        "url": url,
                        "content_type": content_type,
                        "format_type": format_result.format_type,
                        "content_length": len(content)
                    },
                    content,
                    url,
                    content_type
                )
                
                # Add format detection metadata
                parsed_doc.metadata.update({
                    "detected_format": format_result.format_type,
                    "format_confidence": format_result.confidence,
                    "format_metadata": format_result.metadata,
                    "processing_strategy": format_result.metadata.get("processing_strategy"),
                    "original_content_type": content_type
                })
                
                # Process parsed content
                await self._process_parsed_document(task, parsed_doc)
                
            except Exception as e:
                logger.error(f"URL task failed: {e}")
                task.status = IngestionStatus.FAILED
                task.error_message = str(e)
                task.end_time = datetime.now()
                self.progress_monitor.update_task(task)
            finally:
                self.active_tasks.pop(task.task_id, None)
    
    async def _process_directory_task(
        self,
        task: IngestionTask,
        directory_path: Path,
        files: List[Path]
    ):
        """Process directory ingestion task"""
        
        async with self.semaphore:
            try:
                task.status = IngestionStatus.INITIALIZING
                task.start_time = datetime.now()
                task.update_progress("Starting directory processing", 0)
                self.progress_monitor.update_task(task)
                
                successful_files = 0
                failed_files = 0
                
                # Process files in batches
                for i, file_path in enumerate(files):
                    try:
                        task.update_progress(f"Processing {file_path.name}", i)
                        task.status = IngestionStatus.PROCESSING
                        self.progress_monitor.update_task(task)
                        
                        # Create sub-task for file
                        sub_task_id = await self.ingest_file(file_path)
                        
                        # Wait for completion (with timeout)
                        sub_task = self.active_tasks.get(sub_task_id)
                        if sub_task:
                            await asyncio.wait_for(sub_task, timeout=self.config.timeout_seconds)
                        
                        # Check result
                        sub_task_status = self.progress_monitor.get_task_status(sub_task_id)
                        if sub_task_status and sub_task_status.status == IngestionStatus.COMPLETED:
                            successful_files += 1
                        else:
                            failed_files += 1
                            logger.warning(f"Failed to process {file_path}: {sub_task_status.error_message if sub_task_status else 'Unknown error'}")
                    
                    except Exception as e:
                        failed_files += 1
                        logger.error(f"Error processing {file_path}: {e}")
                
                task.update_progress("Directory processing completed", len(files))
                task.status = IngestionStatus.COMPLETED
                task.result = {
                    "total_files": len(files),
                    "successful_files": successful_files,
                    "failed_files": failed_files
                }
                task.end_time = datetime.now()
                self.progress_monitor.update_task(task)
                
            except Exception as e:
                logger.error(f"Directory task failed: {e}")
                task.status = IngestionStatus.FAILED
                task.error_message = str(e)
                task.end_time = datetime.now()
                self.progress_monitor.update_task(task)
            finally:
                self.active_tasks.pop(task.task_id, None)
    
    async def _process_text_content_task(
        self,
        task: IngestionTask,
        content: str,
        source_name: str,
        content_type: str
    ):
        """Process text content ingestion task"""
        
        async with self.semaphore:
            try:
                task.status = IngestionStatus.INITIALIZING
                task.start_time = datetime.now()
                task.update_progress("Processing text content", 0)
                self.progress_monitor.update_task(task)
                
                # Create parsed document
                parsed_doc = ParsedDocument(
                    content=content,
                    metadata={
                        "source_name": source_name,
                        "content_type": content_type,
                        "content_length": len(content),
                        "created_at": datetime.now().isoformat()
                    },
                    source_type=content_type,
                    source_path=source_name
                )
                
                # Process parsed content
                await self._process_parsed_document(task, parsed_doc)
                
            except Exception as e:
                logger.error(f"Text content task failed: {e}")
                task.status = IngestionStatus.FAILED
                task.error_message = str(e)
                task.end_time = datetime.now()
                self.progress_monitor.update_task(task)
            finally:
                self.active_tasks.pop(task.task_id, None)
    
    async def _process_parsed_document(self, task: IngestionTask, parsed_doc: ParsedDocument):
        """Common processing for parsed documents"""
        
        # Generate document ID first
        document_id = f"doc_{uuid.uuid4().hex}"
        
        # Comprehensive content quality assessment
        task.update_progress("Assessing content quality", task.completed_steps + 1)
        self.progress_monitor.update_task(task)
        
        # Perform quality assessment if enabled
        if self.config.enable_quality_assessment or self.config.enable_duplicate_detection:
            quality_results = await self._assess_content_quality(
                task, parsed_doc.content, parsed_doc.metadata, document_id
            )
            
            # Update task with quality information
            task.quality_metrics = quality_results["quality_metrics"]
            task.duplicates_found = quality_results["duplicates"]
            task.quality_recommendations = quality_results["recommendations"]
            
            # Apply quality-based filtering
            if self._should_skip_content(quality_results):
                task.status = IngestionStatus.COMPLETED
                task.result = {
                    "skipped": True,
                    "reason": "Content quality or duplicate filtering",
                    "quality_score": quality_results["quality_metrics"].overall_quality,
                    "recommendations": quality_results["recommendations"]
                }
                task.end_time = datetime.now()
                self.progress_monitor.update_task(task)
                return
        else:
            # Fall back to basic validation
            if not self._validate_content_quality(parsed_doc.content):
                raise ValueError("Content failed basic quality validation")
        
        # Chunking
        task.update_progress("Creating chunks", task.completed_steps + 1)
        task.status = IngestionStatus.CHUNKING
        self.progress_monitor.update_task(task)
        
        child_chunks, parent_chunks = self.text_chunker.hierarchical_chunk(
            text=parsed_doc.content,
            document_id=document_id
        )
        
        if not child_chunks:
            raise ValueError("No valid chunks created from content")
        
        # Generate embeddings
        task.update_progress("Generating embeddings", task.completed_steps + 1)
        task.status = IngestionStatus.EMBEDDING
        self.progress_monitor.update_task(task)
        
        # Prepare all chunks for embedding (children + parents)
        all_chunks = child_chunks + parent_chunks
        chunk_texts = [chunk.content for chunk in all_chunks]
        embeddings = await self.vector_retriever._get_embeddings(chunk_texts)
        
        # Store in vector database
        task.update_progress("Storing in vector database", task.completed_steps + 1)
        task.status = IngestionStatus.STORING
        self.progress_monitor.update_task(task)
        
        points = []
        
        # Store child chunks
        for i, chunk in enumerate(child_chunks):
            point = {
                "id": chunk.chunk_id,
                "vector": embeddings[i],
                "payload": {
                    "content": chunk.content,
                    "source": parsed_doc.source_path,
                    "document_id": document_id,
                    "chunk_type": "child",
                    "parent_id": chunk.parent_id,
                    "hierarchy_level": 1,
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                    "token_count": chunk.metadata.get("token_count", 0),
                    "char_count": len(chunk.content),
                    "source_type": parsed_doc.source_type,
                    "processing_timestamp": datetime.now().isoformat(),
                    # Quality metrics
                    "quality_score": task.quality_metrics.overall_quality if task.quality_metrics else None,
                    "authority_score": task.quality_metrics.authority_score if task.quality_metrics else None,
                    "relevance_score": task.quality_metrics.relevance_score if task.quality_metrics else None,
                    "source_authority": task.quality_metrics.source_authority if task.quality_metrics else "unknown",
                    **chunk.metadata,
                    **parsed_doc.metadata
                }
            }
            points.append(point)
        
        # Store parent chunks
        for i, parent_chunk in enumerate(parent_chunks):
            parent_point = {
                "id": parent_chunk.chunk_id,
                "vector": embeddings[len(child_chunks) + i],
                "payload": {
                    "content": parent_chunk.content,
                    "source": parsed_doc.source_path,
                    "document_id": document_id,
                    "chunk_type": "parent",
                    "parent_id": None,
                    "hierarchy_level": 0,
                    "start_index": parent_chunk.start_index,
                    "end_index": parent_chunk.end_index,
                    "token_count": parent_chunk.metadata.get("token_count", 0),
                    "char_count": len(parent_chunk.content),
                    "source_type": parsed_doc.source_type,
                    "processing_timestamp": datetime.now().isoformat(),
                    # Quality metrics
                    "quality_score": task.quality_metrics.overall_quality if task.quality_metrics else None,
                    "authority_score": task.quality_metrics.authority_score if task.quality_metrics else None,
                    "relevance_score": task.quality_metrics.relevance_score if task.quality_metrics else None,
                    "source_authority": task.quality_metrics.source_authority if task.quality_metrics else "unknown",
                    **parent_chunk.metadata,
                    **parsed_doc.metadata
                }
            }
            points.append(parent_point)
        
        await self.qdrant_manager.upsert_points(points)
        
        # Complete task
        task.update_progress("Processing completed", task.total_steps)
        task.status = IngestionStatus.COMPLETED
        task.result = {
            "document_id": document_id,
            "chunks_created": len(child_chunks),
            "parent_chunks_created": len(parent_chunks),
            "total_chunks": len(child_chunks) + len(parent_chunks),
            "source_type": parsed_doc.source_type,
            "content_length": len(parsed_doc.content),
            "quality_score": task.quality_metrics.overall_quality if task.quality_metrics else None,
            "authority_score": task.quality_metrics.authority_score if task.quality_metrics else None,
            "duplicates_detected": len(task.duplicates_found),
            "quality_recommendations": task.quality_recommendations
        }
        task.end_time = datetime.now()
        self.progress_monitor.update_task(task)
    
    def _validate_content_quality(self, content: str) -> bool:
        """Validate content quality"""
        
        if len(content.strip()) < self.config.min_chunk_length:
            return False
        
        # Check for reasonable text-to-whitespace ratio
        text_chars = len([c for c in content if c.isalnum()])
        total_chars = len(content)
        
        if total_chars > 0 and (text_chars / total_chars) < self.config.content_quality_threshold:
            return False
        
        return True
    
    async def _assess_content_quality(
        self,
        task: IngestionTask,
        content: str,
        metadata: Dict[str, Any],
        content_id: str
    ) -> Dict[str, Any]:
        """Comprehensive content quality assessment"""
        
        try:
            # Prepare metadata for quality assessment
            source_metadata = {
                "source_type": metadata.get("source_type", "unknown"),
                "source": metadata.get("source", ""),
                "domain": metadata.get("domain", ""),
                "created_at": metadata.get("created_at"),
                "modified_at": metadata.get("modified_at"),
                "is_official": metadata.get("is_official", False),
                "verification_status": metadata.get("verification_status"),
                **metadata
            }
            
            # Get existing documents for relationship analysis
            existing_documents = await self._get_existing_documents_for_analysis()
            
            # Perform comprehensive quality assessment
            quality_results = await self.quality_manager.process_content(
                content=content,
                content_id=content_id,
                source_metadata=source_metadata,
                existing_documents=existing_documents
            )
            
            # Add source type for quality threshold determination
            quality_results["source_type"] = "url_webpage"  # This will be properly detected in a full implementation
            
            logger.info(f"Quality assessment completed for {content_id}: "
                       f"quality={quality_results['quality_metrics'].overall_quality:.2f}, "
                       f"duplicates={len(quality_results['duplicates'])}")
            
            return quality_results
            
        except Exception as e:
            logger.error(f"Error in content quality assessment: {e}")
            # Return minimal results on error
            return {
                "quality_metrics": ContentQualityMetrics(),
                "duplicates": [],
                "relationships": {},
                "processing_status": "error",
                "recommendations": [f"Quality assessment failed: {str(e)}"]
            }
    
    async def _get_existing_documents_for_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Get existing documents for relationship analysis"""
        
        try:
            # This is a simplified version - in production you might want to limit this
            # to recent documents or documents from the same source type
            
            # For now, return empty dict - could be enhanced to query vector DB
            # for existing document metadata
            return {}
            
        except Exception as e:
            logger.warning(f"Could not retrieve existing documents for analysis: {e}")
            return {}
    
    def _should_skip_content(self, quality_results: Dict[str, Any]) -> bool:
        """Determine if content should be skipped based on quality assessment"""
        
        quality_metrics = quality_results["quality_metrics"]
        duplicates = quality_results["duplicates"]
        
        # Skip exact duplicates if configured
        if self.config.skip_exact_duplicates:
            exact_duplicates = [d for d in duplicates if d.match_type == "exact"]
            if exact_duplicates:
                logger.info(f"Skipping content due to exact duplicate: {exact_duplicates[0].content_id}")
                return True
        
        # Skip low quality content if configured
        if self.config.skip_low_quality_content:
            # Use source-specific threshold if available
            source_type = quality_results.get("source_type", "unknown")
            threshold = self.config.quality_thresholds.get(source_type, self.config.content_quality_threshold)
            
            if quality_metrics.overall_quality < threshold:
                logger.info(f"Skipping content due to low quality: {quality_metrics.overall_quality:.2f} < {threshold} (source: {source_type})")
                return True
        
        # Check for specific quality issues that should block ingestion
        blocking_issues = [
            "Contains placeholder content",
            "Content too short",
            "Insufficient word count"
        ]
        
        for issue in quality_metrics.quality_issues:
            if any(blocking_issue in issue for blocking_issue in blocking_issues):
                logger.info(f"Skipping content due to quality issue: {issue}")
                return True
        
        return False
    
    async def get_quality_summary(self) -> Dict[str, Any]:
        """Get overall quality summary from the quality manager"""
        return await self.quality_manager.get_quality_summary()
    
    # Status and monitoring methods
    
    def get_task_status(self, task_id: str) -> Optional[IngestionTask]:
        """Get task status"""
        return self.progress_monitor.get_task_status(task_id)
    
    def get_active_tasks(self) -> List[IngestionTask]:
        """Get all active tasks"""
        return self.progress_monitor.get_active_tasks()
    
    def get_completed_tasks(self) -> List[IngestionTask]:
        """Get completed tasks"""
        return self.progress_monitor.get_completed_tasks()
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel an active task"""
        
        if task_id in self.active_tasks:
            self.active_tasks[task_id].cancel()
            
            task = self.progress_monitor.get_task_status(task_id)
            if task:
                task.status = IngestionStatus.CANCELLED
                task.end_time = datetime.now()
                self.progress_monitor.update_task(task)
            
            return True
        
        return False
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> IngestionTask:
        """Wait for a task to complete"""
        
        if task_id in self.active_tasks:
            try:
                await asyncio.wait_for(self.active_tasks[task_id], timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Task {task_id} timed out")
        
        return self.progress_monitor.get_task_status(task_id)
    
    async def wait_for_all_tasks(self, timeout: Optional[float] = None) -> List[IngestionTask]:
        """Wait for all active tasks to complete"""
        
        if self.active_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.active_tasks.values(), return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning("Some tasks timed out")
        
        return self.progress_monitor.get_completed_tasks()
    
    def register_progress_callback(self, callback: Callable[[IngestionTask], None]):
        """Register a progress callback"""
        self.progress_monitor.register_callback(callback)
    
    async def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        
        # Get error recovery statistics
        recovery_stats = self.error_recovery.get_error_statistics()
        
        # Get task statistics
        pipeline_stats = await self.get_pipeline_statistics()
        
        # Get recent errors from error recovery
        recent_errors = self.error_recovery.get_recent_errors(limit=20)
        
        return {
            "pipeline_statistics": pipeline_stats,
            "error_recovery_statistics": recovery_stats,
            "recent_errors": recent_errors,
            "format_detection_statistics": self._get_format_detection_stats(),
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_format_detection_stats(self) -> Dict[str, Any]:
        """Get format detection statistics from completed tasks"""
        
        completed_tasks = self.get_completed_tasks()
        format_counts = {}
        confidence_scores = []
        
        for task in completed_tasks:
            if task.result and "metadata" in task.result:
                metadata = task.result["metadata"]
                detected_format = metadata.get("detected_format")
                format_confidence = metadata.get("format_confidence")
                
                if detected_format:
                    format_counts[detected_format] = format_counts.get(detected_format, 0) + 1
                
                if format_confidence is not None:
                    confidence_scores.append(format_confidence)
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return {
            "format_distribution": format_counts,
            "average_confidence": avg_confidence,
            "total_format_detections": len(confidence_scores),
            "supported_formats": len(self.format_detector.get_supported_formats())
        }
    
    async def retry_failed_task(self, task_id: str) -> str:
        """Retry a failed task with the same parameters"""
        
        original_task = self.get_task_status(task_id)
        if not original_task:
            raise ValueError(f"Task {task_id} not found")
        
        if original_task.status != IngestionStatus.FAILED:
            raise ValueError(f"Task {task_id} is not in failed state")
        
        # Create new task with same parameters
        new_task_id = f"retry_{task_id}_{uuid.uuid4().hex[:8]}"
        
        if original_task.source_type == SourceType.LOCAL_FILE:
            return await self.ingest_file(
                original_task.source_identifier,
                task_id=new_task_id
            )
        elif original_task.source_type == SourceType.URL:
            return await self.ingest_url(
                original_task.source_identifier,
                task_id=new_task_id
            )
        elif original_task.source_type == SourceType.MATTERMOST_CHANNEL:
            parts = original_task.source_identifier.split("/")
            if len(parts) == 2:
                team_id, channel_id = parts
                max_messages = original_task.metadata.get("max_messages", 1000)
                return await self.ingest_mattermost_channel(
                    channel_id, team_id, max_messages, task_id=new_task_id
                )
        elif original_task.source_type == SourceType.TEXT_CONTENT:
            content = original_task.metadata.get("content", "")
            content_type = original_task.metadata.get("content_type", "text")
            return await self.ingest_text_content(
                content, original_task.source_identifier, content_type, task_id=new_task_id
            )
        
        raise ValueError(f"Cannot retry task of type {original_task.source_type}")
    
    async def bulk_ingest_files(
        self,
        file_paths: List[Union[str, Path]],
        max_concurrent: Optional[int] = None
    ) -> List[str]:
        """Ingest multiple files concurrently"""
        
        max_concurrent = max_concurrent or self.config.max_concurrent_tasks
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def ingest_single_file(file_path):
            async with semaphore:
                return await self.ingest_file(file_path)
        
        # Create tasks for all files
        tasks = [ingest_single_file(fp) for fp in file_paths]
        
        # Execute concurrently
        task_ids = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return successful task IDs
        successful_task_ids = [
            task_id for task_id in task_ids
            if isinstance(task_id, str)
        ]
        
        logger.info(f"Bulk ingestion started: {len(successful_task_ids)}/{len(file_paths)} files")
        
        return successful_task_ids
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall pipeline health status"""
        
        active_tasks = self.get_active_tasks()
        error_stats = self.error_recovery.get_error_statistics()
        
        # Calculate health score
        total_tasks = len(self.progress_monitor.tasks)
        failed_tasks = len([t for t in self.progress_monitor.tasks.values() if t.status == IngestionStatus.FAILED])
        
        health_score = 1.0
        if total_tasks > 0:
            failure_rate = failed_tasks / total_tasks
            health_score = max(0.0, 1.0 - (failure_rate * 2))  # Penalize failures
        
        # Check component health
        component_health = {
            "format_detector": True,
            "error_recovery": error_stats.get("recent_errors", 0) < 10,
            "document_parser": True,
            "vector_retriever": True,
            "qdrant_manager": True
        }
        
        overall_health = all(component_health.values()) and health_score > 0.5
        
        return {
            "overall_healthy": overall_health,
            "health_score": health_score,
            "active_tasks_count": len(active_tasks),
            "failed_tasks_count": failed_tasks,
            "total_tasks_count": total_tasks,
            "recent_error_count": error_stats.get("recent_errors", 0),
            "component_health": component_health,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        
        all_tasks = list(self.progress_monitor.tasks.values())
        active_tasks = self.get_active_tasks()
        completed_tasks = self.get_completed_tasks()
        
        # Calculate statistics
        total_processing_time = sum(
            task.duration_seconds for task in completed_tasks
            if task.duration_seconds is not None
        )
        
        successful_tasks = [t for t in completed_tasks if t.status == IngestionStatus.COMPLETED]
        failed_tasks = [t for t in all_tasks if t.status == IngestionStatus.FAILED]
        
        # Source type distribution
        source_type_counts = {}
        for task in all_tasks:
            source_type = task.source_type.value
            source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
        
        return {
            "total_tasks": len(all_tasks),
            "active_tasks": len(active_tasks),
            "completed_tasks": len(completed_tasks),
            "successful_tasks": len(successful_tasks),
            "failed_tasks": len(failed_tasks),
            "total_processing_time_seconds": total_processing_time,
            "average_processing_time_seconds": total_processing_time / max(len(completed_tasks), 1),
            "source_type_distribution": source_type_counts,
            "success_rate": len(successful_tasks) / max(len(all_tasks), 1) * 100
        }