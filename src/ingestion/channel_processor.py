import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from loguru import logger

from .chunking.channel_chunker import ChannelMessageChunker
from ..mattermost.channel_client import MattermostClient
from ..storage.qdrant_client import QdrantManager
from ..config.settings import settings


@dataclass
class ChannelContext:
    """Rich context information for a channel"""
    channel_id: str
    channel_name: str
    display_name: str
    team_id: str
    team_name: str
    purpose: str
    header: str
    channel_type: str  # 'O' = public, 'P' = private, 'D' = direct, 'G' = group
    member_count: int
    created_at: datetime
    last_post_at: Optional[datetime]
    total_msg_count: int
    creator_id: str
    
    @classmethod
    def from_mattermost_data(
        cls,
        channel_data: Dict[str, Any],
        team_data: Dict[str, Any],
        stats_data: Dict[str, Any] = None
    ) -> 'ChannelContext':
        """Create ChannelContext from Mattermost API data"""
        
        stats = stats_data or {}
        
        return cls(
            channel_id=channel_data.get("id", ""),
            channel_name=channel_data.get("name", ""),
            display_name=channel_data.get("display_name", ""),
            team_id=team_data.get("id", ""),
            team_name=team_data.get("display_name", ""),
            purpose=channel_data.get("purpose", ""),
            header=channel_data.get("header", ""),
            channel_type=channel_data.get("type", "O"),
            member_count=stats.get("member_count", 0),
            created_at=datetime.fromtimestamp(channel_data.get("create_at", 0) / 1000),
            last_post_at=datetime.fromtimestamp(channel_data.get("last_post_at", 0) / 1000) if channel_data.get("last_post_at") else None,
            total_msg_count=stats.get("total_msg_count", 0),
            creator_id=channel_data.get("creator_id", "")
        )


@dataclass
class UserProfile:
    """User profile information"""
    user_id: str
    username: str
    email: str
    first_name: str
    last_name: str
    nickname: str
    position: str
    roles: str
    locale: str
    timezone: Optional[str] = None
    
    @property
    def display_name(self) -> str:
        """Get the best display name for the user"""
        if self.nickname:
            return self.nickname
        elif self.first_name or self.last_name:
            return f"{self.first_name} {self.last_name}".strip()
        else:
            return self.username
    
    @classmethod
    def from_mattermost_data(cls, user_data: Dict[str, Any]) -> 'UserProfile':
        """Create UserProfile from Mattermost API data"""
        
        return cls(
            user_id=user_data.get("id", ""),
            username=user_data.get("username", ""),
            email=user_data.get("email", ""),
            first_name=user_data.get("first_name", ""),
            last_name=user_data.get("last_name", ""),
            nickname=user_data.get("nickname", ""),
            position=user_data.get("position", ""),
            roles=user_data.get("roles", ""),
            locale=user_data.get("locale", "en"),
            timezone=user_data.get("timezone", {}).get("useAutomaticTimezone")
        )


@dataclass
class ProcessingMetrics:
    """Metrics from channel processing"""
    total_messages: int
    valid_messages: int
    conversation_groups: int
    threads_extracted: int
    chunks_created: int
    parent_chunks_created: int
    processing_time_seconds: float
    participants: Set[str]
    time_range: Optional[Tuple[datetime, datetime]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        result = asdict(self)
        result["participants"] = list(self.participants)
        if self.time_range:
            result["time_range"] = [dt.isoformat() for dt in self.time_range]
        return result


class ChannelMessageProcessor:
    """Advanced channel message processor with metadata integration"""
    
    def __init__(self):
        self.mattermost_client = MattermostClient()
        self.qdrant_manager = QdrantManager()
        self.chunker = ChannelMessageChunker(
            conversation_gap_minutes=settings.CONVERSATION_GAP_MINUTES,
            max_group_size=settings.MAX_GROUP_SIZE,
            preserve_threads=settings.PRESERVE_THREADS
        )
        self.user_cache: Dict[str, UserProfile] = {}
        
    async def process_channel_complete(
        self,
        channel_id: str,
        team_id: str,
        max_messages: int = 1000,
        include_user_profiles: bool = True
    ) -> Dict[str, Any]:
        """Complete channel processing with full metadata integration"""
        
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting complete channel processing for {channel_id}")
            
            # Step 1: Gather channel context
            channel_context = await self._build_channel_context(channel_id, team_id)
            if not channel_context:
                return {"success": False, "error": "Failed to get channel context"}
            
            # Step 2: Extract message history
            raw_messages = await self.mattermost_client.get_channel_history(
                channel_id=channel_id,
                max_messages=max_messages
            )
            
            if not raw_messages:
                return {"success": False, "error": "No messages found in channel"}
            
            # Step 3: Build user profiles if requested
            user_map = {}
            if include_user_profiles:
                user_map = await self._build_user_map(raw_messages)
            
            # Step 4: Process messages into chunks
            document_id = f"channel_{channel_id}_{int(datetime.now().timestamp())}"
            
            child_chunks, parent_chunks = self.chunker.chunk_channel_messages(
                raw_messages=raw_messages,
                channel_info=asdict(channel_context),
                team_info={"id": channel_context.team_id, "display_name": channel_context.team_name},
                document_id=document_id,
                user_map={uid: profile.display_name for uid, profile in user_map.items()}
            )
            
            if not child_chunks and not parent_chunks:
                return {"success": False, "error": "No chunks generated from messages"}
            
            # Step 5: Enhanced metadata integration
            enhanced_metadata = await self._create_enhanced_metadata(
                channel_context, user_map, raw_messages
            )
            
            # Step 6: Store in vector database
            await self._store_chunks_with_metadata(
                child_chunks, parent_chunks, channel_context, enhanced_metadata
            )
            
            # Step 7: Generate processing metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            participants = set(msg.get("user_id") for msg in raw_messages if msg.get("user_id"))
            
            if raw_messages:
                timestamps = [msg.get("create_at", 0) for msg in raw_messages]
                time_range = (
                    datetime.fromtimestamp(min(timestamps) / 1000),
                    datetime.fromtimestamp(max(timestamps) / 1000)
                )
            else:
                time_range = None
            
            metrics = ProcessingMetrics(
                total_messages=len(raw_messages),
                valid_messages=len([m for m in raw_messages if m.get("message", "").strip()]),
                conversation_groups=len(self.chunker.group_conversations(
                    self.chunker.parse_messages(raw_messages)
                )),
                threads_extracted=len(self.chunker.extract_conversation_threads(
                    self.chunker.parse_messages(raw_messages)
                )),
                chunks_created=len(child_chunks),
                parent_chunks_created=len(parent_chunks),
                processing_time_seconds=processing_time,
                participants=participants,
                time_range=time_range
            )
            
            logger.info(f"Channel processing completed successfully: {metrics.chunks_created} chunks created")
            
            return {
                "success": True,
                "document_id": document_id,
                "channel_context": asdict(channel_context),
                "metrics": metrics.to_dict(),
                "enhanced_metadata": enhanced_metadata
            }
            
        except Exception as e:
            logger.error(f"Channel processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _build_channel_context(self, channel_id: str, team_id: str) -> Optional[ChannelContext]:
        """Build comprehensive channel context"""
        
        try:
            # Get channel info
            channel_data = await self.mattermost_client.get_channel_info(channel_id)
            if not channel_data:
                return None
            
            # Get team info
            team_data = await self.mattermost_client.get_team_info(team_id)
            if not team_data:
                team_data = {"id": team_id, "display_name": "Unknown Team"}
            
            # Get channel stats
            stats_data = await self.mattermost_client.get_channel_stats(channel_id)
            
            return ChannelContext.from_mattermost_data(channel_data, team_data, stats_data)
            
        except Exception as e:
            logger.error(f"Failed to build channel context: {e}")
            return None
    
    async def _build_user_map(self, messages: List[Dict[str, Any]]) -> Dict[str, UserProfile]:
        """Build user profile map for message authors"""
        
        user_map = {}
        unique_users = set(msg.get("user_id") for msg in messages if msg.get("user_id"))
        
        logger.info(f"Building user profiles for {len(unique_users)} users")
        
        # Process users in batches to avoid overwhelming the API
        batch_size = 10
        user_batches = [list(unique_users)[i:i + batch_size] 
                       for i in range(0, len(unique_users), batch_size)]
        
        for batch in user_batches:
            # Process batch concurrently
            tasks = [self._get_user_profile(user_id) for user_id in batch]
            profiles = await asyncio.gather(*tasks, return_exceptions=True)
            
            for user_id, profile in zip(batch, profiles):
                if isinstance(profile, UserProfile):
                    user_map[user_id] = profile
                    self.user_cache[user_id] = profile
                else:
                    logger.warning(f"Failed to get profile for user {user_id}: {profile}")
        
        logger.info(f"Successfully built {len(user_map)} user profiles")
        return user_map
    
    async def _get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile with caching"""
        
        # Check cache first
        if user_id in self.user_cache:
            return self.user_cache[user_id]
        
        try:
            user_data = await self.mattermost_client.get_user_info(user_id)
            if user_data:
                profile = UserProfile.from_mattermost_data(user_data)
                self.user_cache[user_id] = profile
                return profile
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get user profile for {user_id}: {e}")
            return None
    
    async def _create_enhanced_metadata(
        self,
        channel_context: ChannelContext,
        user_map: Dict[str, UserProfile],
        raw_messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create enhanced metadata with analytics"""
        
        # Message analytics
        message_types = {}
        hashtag_frequency = {}
        user_activity = {}
        
        for msg in raw_messages:
            # Message type analysis
            msg_type = msg.get("type", "regular")
            message_types[msg_type] = message_types.get(msg_type, 0) + 1
            
            # Hashtag analysis
            hashtags = msg.get("hashtags", "").split()
            for hashtag in hashtags:
                if hashtag:
                    hashtag_frequency[hashtag] = hashtag_frequency.get(hashtag, 0) + 1
            
            # User activity analysis
            user_id = msg.get("user_id")
            if user_id:
                if user_id not in user_activity:
                    user_activity[user_id] = {"message_count": 0, "char_count": 0}
                user_activity[user_id]["message_count"] += 1
                user_activity[user_id]["char_count"] += len(msg.get("message", ""))
        
        # Channel activity patterns
        if raw_messages:
            timestamps = [msg.get("create_at", 0) for msg in raw_messages]
            activity_span_days = (max(timestamps) - min(timestamps)) / (1000 * 60 * 60 * 24)
            avg_messages_per_day = len(raw_messages) / max(activity_span_days, 1)
        else:
            activity_span_days = 0
            avg_messages_per_day = 0
        
        # Top participants
        top_participants = sorted(
            user_activity.items(),
            key=lambda x: x[1]["message_count"],
            reverse=True
        )[:10]
        
        # Enhanced metadata structure
        enhanced_metadata = {
            # Channel metadata
            "channel_context": asdict(channel_context),
            
            # User metadata  
            "user_profiles": {uid: asdict(profile) for uid, profile in user_map.items()},
            "participant_count": len(user_map),
            "top_participants": [
                {
                    "user_id": uid,
                    "username": user_map.get(uid, UserProfile("", uid, "", "", "", "", "", "", "")).username,
                    "message_count": stats["message_count"],
                    "char_count": stats["char_count"]
                }
                for uid, stats in top_participants
            ],
            
            # Content metadata
            "message_analytics": {
                "total_messages": len(raw_messages),
                "message_types": message_types,
                "hashtag_frequency": dict(sorted(hashtag_frequency.items(), key=lambda x: x[1], reverse=True)[:20]),
                "activity_span_days": activity_span_days,
                "avg_messages_per_day": avg_messages_per_day
            },
            
            # Processing metadata
            "processing_timestamp": datetime.now().isoformat(),
            "processor_version": "1.0",
            "chunking_strategy": "hierarchical_conversation_aware",
            "preserve_threads": self.chunker.preserve_threads,
            "conversation_gap_minutes": self.chunker.conversation_gap_minutes
        }
        
        return enhanced_metadata
    
    async def _store_chunks_with_metadata(
        self,
        child_chunks: List,
        parent_chunks: List,
        channel_context: ChannelContext,
        enhanced_metadata: Dict[str, Any]
    ):
        """Store chunks with comprehensive metadata"""
        
        try:
            # Prepare chunks for storage
            all_chunks = child_chunks + parent_chunks
            chunk_texts = [chunk.content for chunk in all_chunks]
            
            # Get embeddings
            from ..retrieval.retrievers.vector_retriever import VectorRetriever
            vector_retriever = VectorRetriever()
            embeddings = await vector_retriever._get_embeddings(chunk_texts)
            
            # Prepare points for Qdrant
            points = []
            
            # Common metadata for all chunks
            common_metadata = {
                "source_type": "mattermost_channel",
                "channel_id": channel_context.channel_id,
                "channel_name": channel_context.channel_name,
                "channel_display_name": channel_context.display_name,
                "team_id": channel_context.team_id,
                "team_name": channel_context.team_name,
                "channel_type": channel_context.channel_type,
                "channel_purpose": channel_context.purpose,
                "processing_timestamp": datetime.now().isoformat(),
                "enhanced_metadata": enhanced_metadata
            }
            
            # Store child chunks
            for i, chunk in enumerate(child_chunks):
                point = {
                    "id": chunk.chunk_id,
                    "vector": embeddings[i],
                    "payload": {
                        "content": chunk.content,
                        "source": f"mattermost://{channel_context.team_name}/{channel_context.channel_name}",
                        "document_id": chunk.metadata.get("document_id"),
                        "chunk_type": "child",
                        "parent_id": chunk.parent_id,
                        "hierarchy_level": 1,
                        "start_index": chunk.start_index,
                        "end_index": chunk.end_index,
                        "token_count": chunk.metadata.get("token_count", 0),
                        "char_count": len(chunk.content),
                        **common_metadata,
                        **chunk.metadata
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
                        "source": f"mattermost://{channel_context.team_name}/{channel_context.channel_name}",
                        "document_id": parent_chunk.metadata.get("document_id"),
                        "chunk_type": "parent",
                        "parent_id": None,
                        "hierarchy_level": 0,
                        "start_index": parent_chunk.start_index,
                        "end_index": parent_chunk.end_index,
                        "token_count": parent_chunk.metadata.get("token_count", 0),
                        "char_count": len(parent_chunk.content),
                        **common_metadata,
                        **parent_chunk.metadata
                    }
                }
                points.append(parent_point)
            
            # Store in Qdrant
            await self.qdrant_manager.upsert_points(points)
            
            logger.info(f"Stored {len(points)} chunks with enhanced metadata in vector database")
            
        except Exception as e:
            logger.error(f"Failed to store chunks with metadata: {e}")
            raise
    
    async def process_channel_incremental(
        self,
        channel_id: str,
        team_id: str,
        since_timestamp: Optional[int] = None,
        max_messages: int = 500
    ) -> Dict[str, Any]:
        """Process only new messages since last ingestion"""
        
        try:
            logger.info(f"Starting incremental channel processing for {channel_id}")
            
            # Get messages since timestamp
            raw_messages = await self.mattermost_client.get_channel_history(
                channel_id=channel_id,
                max_messages=max_messages,
                after=str(since_timestamp) if since_timestamp else None
            )
            
            if not raw_messages:
                return {"success": True, "message": "No new messages to process"}
            
            # Use the complete processing pipeline for new messages
            return await self.process_channel_complete(
                channel_id=channel_id,
                team_id=team_id,
                max_messages=len(raw_messages)
            )
            
        except Exception as e:
            logger.error(f"Incremental channel processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_channel_processing_status(self, channel_id: str) -> Dict[str, Any]:
        """Get processing status for a channel"""
        
        try:
            # Search for existing chunks from this channel
            from ..retrieval.retrievers.vector_retriever import VectorRetriever
            vector_retriever = VectorRetriever()
            
            results = await vector_retriever.similarity_search(
                query="status",  # Dummy query
                top_k=1,
                filters={"channel_id": channel_id}
            )
            
            if results:
                latest_chunk = results[0]
                metadata = latest_chunk.metadata
                
                return {
                    "processed": True,
                    "last_processing_timestamp": metadata.get("processing_timestamp"),
                    "chunk_count": len(results),
                    "channel_name": metadata.get("channel_display_name"),
                    "team_name": metadata.get("team_name")
                }
            else:
                return {"processed": False, "message": "Channel not yet processed"}
                
        except Exception as e:
            logger.error(f"Failed to get channel processing status: {e}")
            return {"processed": False, "error": str(e)}