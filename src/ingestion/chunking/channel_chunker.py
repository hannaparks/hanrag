import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from loguru import logger
from .text_chunker import TextChunk, TextChunker


@dataclass
class ChannelMessage:
    """Represents a Mattermost channel message with metadata"""
    id: str
    message: str
    user_id: str
    create_at: int
    update_at: Optional[int] = None
    channel_id: Optional[str] = None
    type: str = ""
    props: Dict[str, Any] = None
    hashtags: str = ""
    reply_count: int = 0
    root_id: str = ""
    parent_id: str = ""
    
    def __post_init__(self):
        if self.props is None:
            self.props = {}
        if self.update_at is None:
            self.update_at = self.create_at
    
    @property
    def datetime(self) -> datetime:
        """Convert timestamp to datetime object"""
        return datetime.fromtimestamp(self.create_at / 1000)
    
    @property
    def is_thread_reply(self) -> bool:
        """Check if this is a thread reply"""
        return bool(self.root_id and self.root_id != self.id)
    
    @property
    def is_system_message(self) -> bool:
        """Check if this is a system message"""
        return self.type.startswith("system_") or not self.message.strip()


@dataclass
class ConversationThread:
    """Represents a conversation thread with metadata"""
    thread_id: str
    root_message: ChannelMessage
    replies: List[ChannelMessage]
    participants: List[str]
    created_at: datetime
    last_activity: datetime
    topic_keywords: List[str]
    
    @property
    def all_messages(self) -> List[ChannelMessage]:
        """Get all messages in thread (root + replies)"""
        return [self.root_message] + self.replies
    
    @property
    def message_count(self) -> int:
        """Total number of messages in thread"""
        return len(self.all_messages)
    
    @property
    def duration_minutes(self) -> float:
        """Duration of conversation in minutes"""
        return (self.last_activity - self.created_at).total_seconds() / 60


@dataclass  
class ConversationGroup:
    """Represents a grouped conversation for optimal chunking"""
    group_id: str
    messages: List[ChannelMessage]
    threads: List[ConversationThread]
    time_range: Tuple[datetime, datetime]
    participants: List[str]
    dominant_topics: List[str]
    
    @property
    def total_messages(self) -> int:
        """Total number of messages in group"""
        return len(self.messages)
    
    @property
    def duration_hours(self) -> float:
        """Duration of conversation group in hours"""
        start, end = self.time_range
        return (end - start).total_seconds() / 3600


class ChannelMessageChunker(TextChunker):
    """Specialized chunker for Mattermost channel messages"""
    
    def __init__(
        self,
        chunk_size: int = None,
        parent_chunk_size: int = None,
        chunk_overlap: int = None,
        conversation_gap_minutes: int = 30,
        max_group_size: int = 25,
        preserve_threads: bool = True
    ):
        super().__init__(chunk_size, parent_chunk_size, chunk_overlap)
        self.conversation_gap_minutes = conversation_gap_minutes
        self.max_group_size = max_group_size
        self.preserve_threads = preserve_threads
    
    def parse_messages(self, raw_messages: List[Dict[str, Any]]) -> List[ChannelMessage]:
        """Parse raw message dictionaries into ChannelMessage objects"""
        
        messages = []
        for msg_data in raw_messages:
            try:
                message = ChannelMessage(
                    id=msg_data.get("id", str(uuid.uuid4())),
                    message=msg_data.get("message", "").strip(),
                    user_id=msg_data.get("user_id", "unknown"),
                    create_at=msg_data.get("create_at", 0),
                    update_at=msg_data.get("update_at"),
                    channel_id=msg_data.get("channel_id"),
                    type=msg_data.get("type", ""),
                    props=msg_data.get("props", {}),
                    hashtags=msg_data.get("hashtags", ""),
                    reply_count=msg_data.get("reply_count", 0),
                    root_id=msg_data.get("root_id", ""),
                    parent_id=msg_data.get("parent_id", "")
                )
                
                # Skip empty or system messages unless they're important
                if message.message and not message.is_system_message:
                    messages.append(message)
                elif message.type in ["channel_header", "channel_purpose", "add_to_channel"]:
                    # Keep important system messages
                    messages.append(message)
                    
            except Exception as e:
                logger.warning(f"Failed to parse message: {e}")
                continue
        
        return messages
    
    def extract_conversation_threads(self, messages: List[ChannelMessage]) -> List[ConversationThread]:
        """Extract conversation threads from messages"""
        
        threads = []
        thread_map = {}
        
        # Group messages by thread
        for message in messages:
            if message.is_thread_reply and message.root_id:
                # This is a reply to a thread
                if message.root_id not in thread_map:
                    thread_map[message.root_id] = {
                        "root": None,
                        "replies": []
                    }
                thread_map[message.root_id]["replies"].append(message)
            else:
                # This could be a root message
                if message.id not in thread_map:
                    thread_map[message.id] = {
                        "root": message,
                        "replies": []
                    }
                else:
                    thread_map[message.id]["root"] = message
        
        # Create ConversationThread objects
        for thread_id, thread_data in thread_map.items():
            root_message = thread_data["root"]
            replies = thread_data["replies"]
            
            if root_message and (replies or not root_message.is_thread_reply):
                # Only create threads with actual content
                all_msgs = [root_message] + replies
                participants = list(set(msg.user_id for msg in all_msgs))
                
                # Extract topic keywords from thread content
                combined_text = " ".join(msg.message for msg in all_msgs)
                topic_keywords = self._extract_keywords(combined_text)
                
                thread = ConversationThread(
                    thread_id=thread_id,
                    root_message=root_message,
                    replies=sorted(replies, key=lambda x: x.create_at),
                    participants=participants,
                    created_at=root_message.datetime,
                    last_activity=max(msg.datetime for msg in all_msgs),
                    topic_keywords=topic_keywords
                )
                threads.append(thread)
        
        return sorted(threads, key=lambda x: x.created_at)
    
    def group_conversations(
        self, 
        messages: List[ChannelMessage],
        threads: List[ConversationThread] = None
    ) -> List[ConversationGroup]:
        """Group conversations for optimal chunking"""
        
        if threads is None:
            threads = self.extract_conversation_threads(messages)
        
        groups = []
        current_messages = []
        current_threads = []
        group_start_time = None
        last_activity = None
        
        # Sort all messages chronologically
        sorted_messages = sorted(messages, key=lambda x: x.create_at)
        
        for message in sorted_messages:
            message_time = message.datetime
            
            # Determine if we should start a new group
            should_start_new_group = (
                current_messages and (
                    # Time gap exceeded
                    (message_time - last_activity).total_seconds() / 60 > self.conversation_gap_minutes or
                    # Group size exceeded
                    len(current_messages) >= self.max_group_size
                )
            )
            
            if should_start_new_group:
                # Finalize current group
                if current_messages:
                    group = self._create_conversation_group(
                        current_messages, 
                        current_threads,
                        group_start_time,
                        last_activity
                    )
                    groups.append(group)
                
                # Start new group
                current_messages = []
                current_threads = []
                group_start_time = None
            
            # Add message to current group
            current_messages.append(message)
            if group_start_time is None:
                group_start_time = message_time
            last_activity = message_time
            
            # Add relevant threads
            for thread in threads:
                if (thread.root_message.id == message.id or 
                    any(reply.id == message.id for reply in thread.replies)):
                    if thread not in current_threads:
                        current_threads.append(thread)
        
        # Add the last group
        if current_messages:
            group = self._create_conversation_group(
                current_messages,
                current_threads, 
                group_start_time,
                last_activity
            )
            groups.append(group)
        
        logger.info(f"Created {len(groups)} conversation groups from {len(messages)} messages")
        return groups
    
    def _create_conversation_group(
        self,
        messages: List[ChannelMessage],
        threads: List[ConversationThread],
        start_time: datetime,
        end_time: datetime
    ) -> ConversationGroup:
        """Create a conversation group with metadata"""
        
        group_id = f"group_{uuid.uuid4().hex[:8]}"
        participants = list(set(msg.user_id for msg in messages))
        
        # Extract dominant topics
        all_text = " ".join(msg.message for msg in messages)
        dominant_topics = self._extract_keywords(all_text, top_k=5)
        
        return ConversationGroup(
            group_id=group_id,
            messages=messages,
            threads=threads,
            time_range=(start_time, end_time),
            participants=participants,
            dominant_topics=dominant_topics
        )
    
    def _extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract keywords from text using simple frequency analysis"""
        
        # Basic keyword extraction (can be enhanced with NLP libraries)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'are', 'was', 'will', 'been', 'have', 'had', 'were', 
            'said', 'each', 'which', 'their', 'time', 'would', 'there', 'could',
            'other', 'after', 'first', 'with', 'this', 'that', 'they', 'from',
            'you', 'your', 'yours', 'his', 'her', 'him', 'can', 'about', 'into'
        }
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k] if freq > 1]
    
    def format_conversation_group(
        self,
        group: ConversationGroup,
        channel_info: Dict[str, Any],
        team_info: Dict[str, Any],
        user_map: Dict[str, str] = None
    ) -> str:
        """Format a conversation group as readable text for embedding"""
        
        lines = []
        
        # Add group header with context
        channel_name = channel_info.get("display_name", "Unknown Channel")
        team_name = team_info.get("display_name", "Unknown Team")
        
        start_time, end_time = group.time_range
        duration = group.duration_hours
        
        lines.append(f"## Conversation in #{channel_name} ({team_name})")
        lines.append(f"**Time Range:** {start_time.strftime('%Y-%m-%d %H:%M')} - {end_time.strftime('%H:%M')} ({duration:.1f}h)")
        lines.append(f"**Participants:** {len(group.participants)} users")
        
        if group.dominant_topics:
            lines.append(f"**Topics:** {', '.join(group.dominant_topics)}")
        
        # Add channel context if available
        if channel_info.get("purpose"):
            lines.append(f"**Channel Purpose:** {channel_info['purpose']}")
        
        lines.append("")
        
        # Format threads if preserve_threads is enabled
        if self.preserve_threads and group.threads:
            lines.append("### Conversation Threads:")
            lines.append("")
            
            # Group messages by thread
            threaded_messages = set()
            
            for thread in group.threads:
                if thread.message_count > 1:  # Only show actual threads
                    lines.append(f"**Thread: {thread.topic_keywords[0] if thread.topic_keywords else 'Discussion'}**")
                    
                    # Format root message
                    root = thread.root_message
                    username = user_map.get(root.user_id, root.user_id) if user_map else root.user_id
                    time_str = root.datetime.strftime("%H:%M")
                    lines.append(f"**{username}** ({time_str}): {root.message}")
                    threaded_messages.add(root.id)
                    
                    # Format replies
                    for reply in thread.replies:
                        reply_username = user_map.get(reply.user_id, reply.user_id) if user_map else reply.user_id
                        reply_time = reply.datetime.strftime("%H:%M")
                        lines.append(f"  ↳ **{reply_username}** ({reply_time}): {reply.message}")
                        threaded_messages.add(reply.id)
                    
                    lines.append("")
            
            # Add non-threaded messages
            non_threaded = [msg for msg in group.messages if msg.id not in threaded_messages]
            if non_threaded:
                lines.append("### Other Messages:")
                lines.append("")
                
                for message in non_threaded:
                    username = user_map.get(message.user_id, message.user_id) if user_map else message.user_id
                    time_str = message.datetime.strftime("%H:%M")
                    lines.append(f"**{username}** ({time_str}): {message.message}")
                    
                    if message.hashtags:
                        lines.append(f"  Tags: {message.hashtags}")
                    lines.append("")
        else:
            # Simple chronological format
            for message in sorted(group.messages, key=lambda x: x.create_at):
                username = user_map.get(message.user_id, message.user_id) if user_map else message.user_id
                time_str = message.datetime.strftime("%H:%M")
                
                if message.is_thread_reply:
                    lines.append(f"  ↳ **{username}** ({time_str}): {message.message}")
                else:
                    lines.append(f"**{username}** ({time_str}): {message.message}")
                
                if message.hashtags:
                    lines.append(f"  Tags: {message.hashtags}")
                lines.append("")
        
        return "\n".join(lines)
    
    def chunk_channel_messages(
        self,
        raw_messages: List[Dict[str, Any]],
        channel_info: Dict[str, Any],
        team_info: Dict[str, Any],
        document_id: str,
        user_map: Dict[str, str] = None
    ) -> Tuple[List[TextChunk], List[TextChunk]]:
        """Complete pipeline for chunking channel messages"""
        
        try:
            # Parse messages
            messages = self.parse_messages(raw_messages)
            if not messages:
                logger.warning("No valid messages to chunk")
                return [], []
            
            # Extract conversation threads
            threads = self.extract_conversation_threads(messages)
            logger.info(f"Extracted {len(threads)} conversation threads")
            
            # Group conversations
            groups = self.group_conversations(messages, threads)
            logger.info(f"Created {len(groups)} conversation groups")
            
            # Process each group into chunks
            all_child_chunks = []
            all_parent_chunks = []
            
            for group_idx, group in enumerate(groups):
                try:
                    # Format group as text
                    group_text = self.format_conversation_group(
                        group, channel_info, team_info, user_map
                    )
                    
                    # Skip very short groups
                    if len(group_text.strip()) < 100:
                        continue
                    
                    # Create hierarchical chunks for this group
                    group_doc_id = f"{document_id}_group_{group_idx}"
                    child_chunks, parent_chunks = self.hierarchical_chunk(
                        text=group_text,
                        document_id=group_doc_id
                    )
                    
                    # Enhance metadata with channel-specific information
                    for chunk in child_chunks + parent_chunks:
                        chunk.metadata.update({
                            "conversation_group_id": group.group_id,
                            "group_index": group_idx,
                            "message_count": group.total_messages,
                            "thread_count": len(group.threads),
                            "participants": group.participants,
                            "dominant_topics": group.dominant_topics,
                            "conversation_duration_hours": group.duration_hours,
                            "time_range_start": group.time_range[0].isoformat(),
                            "time_range_end": group.time_range[1].isoformat(),
                            "channel_id": channel_info.get("id"),
                            "channel_name": channel_info.get("display_name"),
                            "team_id": team_info.get("id"),
                            "team_name": team_info.get("display_name"),
                            "source_type": "mattermost_channel_conversation"
                        })
                    
                    all_child_chunks.extend(child_chunks)
                    all_parent_chunks.extend(parent_chunks)
                    
                except Exception as e:
                    logger.error(f"Failed to process conversation group {group_idx}: {e}")
                    continue
            
            logger.info(f"Generated {len(all_child_chunks)} child chunks and {len(all_parent_chunks)} parent chunks")
            return all_child_chunks, all_parent_chunks
            
        except Exception as e:
            logger.error(f"Channel message chunking failed: {e}")
            return [], []
    
    def chunk_single_conversation(
        self,
        messages: List[ChannelMessage],
        conversation_id: str,
        metadata: Dict[str, Any] = None
    ) -> List[TextChunk]:
        """Chunk a single conversation thread"""
        
        try:
            # Format conversation as text
            lines = []
            
            if metadata:
                lines.append(f"## Conversation: {metadata.get('title', 'Discussion')}")
                lines.append("")
            
            for message in sorted(messages, key=lambda x: x.create_at):
                time_str = message.datetime.strftime("%H:%M")
                
                if message.is_thread_reply:
                    lines.append(f"  ↳ **{message.user_id}** ({time_str}): {message.message}")
                else:
                    lines.append(f"**{message.user_id}** ({time_str}): {message.message}")
                
                if message.hashtags:
                    lines.append(f"  Tags: {message.hashtags}")
                lines.append("")
            
            conversation_text = "\n".join(lines)
            
            # Create chunks
            chunks = self.chunk_by_tokens(conversation_text, self.chunk_size, self.chunk_overlap)
            
            # Create TextChunk objects
            text_chunks = []
            for i, chunk_text in enumerate(chunks):
                chunk = TextChunk(
                    content=chunk_text,
                    start_index=0,  # Simplified for conversation chunks
                    end_index=len(chunk_text),
                    chunk_id=f"{conversation_id}_{i}",
                    metadata={
                        "chunk_type": "conversation",
                        "conversation_id": conversation_id,
                        "chunk_index": i,
                        "message_count": len(messages),
                        "participants": list(set(msg.user_id for msg in messages)),
                        **(metadata or {})
                    }
                )
                text_chunks.append(chunk)
            
            return text_chunks
            
        except Exception as e:
            logger.error(f"Failed to chunk conversation {conversation_id}: {e}")
            return []