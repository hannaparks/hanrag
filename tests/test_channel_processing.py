import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from src.ingestion.chunking.channel_chunker import (
    ChannelMessageChunker, 
    ChannelMessage, 
    ConversationThread,
    ConversationGroup
)
from src.ingestion.channel_processor import (
    ChannelMessageProcessor,
    ChannelContext,
    UserProfile,
    ProcessingMetrics
)


@pytest.fixture
def sample_messages() -> List[Dict[str, Any]]:
    """Sample Mattermost messages for testing"""
    base_time = int(datetime.now().timestamp() * 1000)
    
    return [
        {
            "id": "msg1",
            "message": "Hello everyone, how is the project going?",
            "user_id": "user1",
            "create_at": base_time,
            "channel_id": "channel1",
            "type": "",
            "props": {},
            "hashtags": "",
            "reply_count": 2,
            "root_id": "",
            "parent_id": ""
        },
        {
            "id": "msg2", 
            "message": "The project is going well, we just finished the authentication module",
            "user_id": "user2",
            "create_at": base_time + 60000,  # 1 minute later
            "channel_id": "channel1",
            "type": "",
            "props": {},
            "hashtags": "#progress #auth",
            "reply_count": 0,
            "root_id": "msg1",
            "parent_id": "msg1"
        },
        {
            "id": "msg3",
            "message": "Great! What's next on the roadmap?",
            "user_id": "user3", 
            "create_at": base_time + 120000,  # 2 minutes later
            "channel_id": "channel1",
            "type": "",
            "props": {},
            "hashtags": "",
            "reply_count": 0,
            "root_id": "msg1",
            "parent_id": "msg1"
        },
        {
            "id": "msg4",
            "message": "Let's discuss the database design tomorrow",
            "user_id": "user1",
            "create_at": base_time + 1800000,  # 30 minutes later (new conversation)
            "channel_id": "channel1",
            "type": "",
            "props": {},
            "hashtags": "#database #planning",
            "reply_count": 0,
            "root_id": "",
            "parent_id": ""
        }
    ]


@pytest.fixture
def sample_channel_info() -> Dict[str, Any]:
    """Sample channel information"""
    return {
        "id": "channel1",
        "name": "general",
        "display_name": "General",
        "type": "O",
        "purpose": "General discussion for the team",
        "header": "Welcome to the team!",
        "create_at": int(datetime.now().timestamp() * 1000) - 86400000,
        "creator_id": "admin1"
    }


@pytest.fixture
def sample_team_info() -> Dict[str, Any]:
    """Sample team information"""
    return {
        "id": "team1",
        "name": "testteam",
        "display_name": "Test Team"
    }


@pytest.fixture
def sample_user_data() -> List[Dict[str, Any]]:
    """Sample user data"""
    return [
        {
            "id": "user1",
            "username": "alice",
            "email": "alice@example.com",
            "first_name": "Alice",
            "last_name": "Smith",
            "nickname": "Al",
            "position": "Project Manager",
            "roles": "team_user",
            "locale": "en"
        },
        {
            "id": "user2", 
            "username": "bob",
            "email": "bob@example.com",
            "first_name": "Bob",
            "last_name": "Johnson",
            "nickname": "",
            "position": "Developer",
            "roles": "team_user",
            "locale": "en"
        },
        {
            "id": "user3",
            "username": "charlie",
            "email": "charlie@example.com",
            "first_name": "Charlie",
            "last_name": "Brown",
            "nickname": "CB",
            "position": "Designer",
            "roles": "team_user",
            "locale": "en"
        }
    ]


class TestChannelMessageChunker:
    """Test cases for ChannelMessageChunker"""
    
    def test_parse_messages(self, sample_messages):
        """Test message parsing from raw data"""
        chunker = ChannelMessageChunker()
        messages = chunker.parse_messages(sample_messages)
        
        assert len(messages) == 4
        assert all(isinstance(msg, ChannelMessage) for msg in messages)
        assert messages[0].message == "Hello everyone, how is the project going?"
        assert messages[1].is_thread_reply == True
        assert messages[3].is_thread_reply == False
    
    def test_extract_conversation_threads(self, sample_messages):
        """Test conversation thread extraction"""
        chunker = ChannelMessageChunker()
        messages = chunker.parse_messages(sample_messages)
        threads = chunker.extract_conversation_threads(messages)
        
        assert len(threads) >= 1  # Should have at least one thread
        
        # Find the main thread
        main_thread = None
        for thread in threads:
            if thread.root_message.id == "msg1":
                main_thread = thread
                break
        
        assert main_thread is not None
        assert len(main_thread.replies) == 2
        assert main_thread.participants == ["user1", "user2", "user3"]
    
    def test_group_conversations(self, sample_messages):
        """Test conversation grouping logic"""
        chunker = ChannelMessageChunker(conversation_gap_minutes=20)
        messages = chunker.parse_messages(sample_messages)
        groups = chunker.group_conversations(messages)
        
        # Should create 2 groups due to 30-minute gap
        assert len(groups) == 2
        
        # First group should have 3 messages (the thread)
        assert groups[0].total_messages == 3
        
        # Second group should have 1 message
        assert groups[1].total_messages == 1
    
    def test_format_conversation_group(self, sample_messages, sample_channel_info, sample_team_info):
        """Test conversation group formatting"""
        chunker = ChannelMessageChunker()
        messages = chunker.parse_messages(sample_messages)
        groups = chunker.group_conversations(messages)
        
        user_map = {
            "user1": "Alice Smith",
            "user2": "Bob Johnson", 
            "user3": "Charlie Brown"
        }
        
        formatted_text = chunker.format_conversation_group(
            groups[0], sample_channel_info, sample_team_info, user_map
        )
        
        assert "## Conversation in #General (Test Team)" in formatted_text
        assert "Alice Smith" in formatted_text
        assert "authentication module" in formatted_text
        assert "Thread:" in formatted_text  # Should preserve thread structure
    
    def test_chunk_channel_messages(self, sample_messages, sample_channel_info, sample_team_info):
        """Test complete channel message chunking pipeline"""
        chunker = ChannelMessageChunker()
        
        child_chunks, parent_chunks = chunker.chunk_channel_messages(
            raw_messages=sample_messages,
            channel_info=sample_channel_info,
            team_info=sample_team_info,
            document_id="test_doc"
        )
        
        assert len(child_chunks) > 0
        assert len(parent_chunks) > 0
        
        # Check metadata
        for chunk in child_chunks:
            assert chunk.metadata["source_type"] == "mattermost_channel_conversation"
            assert chunk.metadata["channel_id"] == "channel1"
            assert chunk.metadata["team_name"] == "Test Team"
    
    def test_keyword_extraction(self):
        """Test keyword extraction from text"""
        chunker = ChannelMessageChunker()
        
        text = "The authentication module project is going well with database design"
        keywords = chunker._extract_keywords(text)
        
        assert "authentication" in keywords
        assert "database" in keywords
        assert "project" in keywords
        # Should filter out common words
        assert "the" not in keywords
        assert "is" not in keywords


class TestChannelContext:
    """Test cases for ChannelContext"""
    
    def test_from_mattermost_data(self, sample_channel_info, sample_team_info):
        """Test ChannelContext creation from Mattermost data"""
        
        stats_data = {
            "member_count": 15,
            "total_msg_count": 245
        }
        
        context = ChannelContext.from_mattermost_data(
            sample_channel_info, sample_team_info, stats_data
        )
        
        assert context.channel_id == "channel1"
        assert context.channel_name == "general"
        assert context.display_name == "General"
        assert context.team_name == "Test Team"
        assert context.member_count == 15
        assert context.total_msg_count == 245
        assert context.channel_type == "O"


class TestUserProfile:
    """Test cases for UserProfile"""
    
    def test_from_mattermost_data(self, sample_user_data):
        """Test UserProfile creation from Mattermost data"""
        
        profile = UserProfile.from_mattermost_data(sample_user_data[0])
        
        assert profile.user_id == "user1"
        assert profile.username == "alice"
        assert profile.display_name == "Al"  # Should use nickname
        
        # Test without nickname
        profile2 = UserProfile.from_mattermost_data(sample_user_data[1])
        assert profile2.display_name == "Bob Johnson"  # Should use first + last
    
    def test_display_name_fallback(self):
        """Test display name fallback logic"""
        
        # With nickname
        profile1 = UserProfile("1", "user1", "test@example.com", "John", "Doe", "JD", "", "", "en")
        assert profile1.display_name == "JD"
        
        # Without nickname, with names
        profile2 = UserProfile("2", "user2", "test@example.com", "Jane", "Smith", "", "", "", "en")
        assert profile2.display_name == "Jane Smith"
        
        # Only username
        profile3 = UserProfile("3", "user3", "test@example.com", "", "", "", "", "", "en")
        assert profile3.display_name == "user3"


class TestChannelMessageProcessor:
    """Test cases for ChannelMessageProcessor"""
    
    @pytest.fixture
    def mock_processor(self):
        """Create a mock ChannelMessageProcessor"""
        processor = ChannelMessageProcessor()
        
        # Mock the clients
        processor.mattermost_client = AsyncMock()
        processor.qdrant_manager = AsyncMock()
        
        return processor
    
    @pytest.mark.asyncio
    async def test_build_channel_context(self, mock_processor, sample_channel_info, sample_team_info):
        """Test channel context building"""
        
        # Mock API responses
        mock_processor.mattermost_client.get_channel_info.return_value = sample_channel_info
        mock_processor.mattermost_client.get_team_info.return_value = sample_team_info
        mock_processor.mattermost_client.get_channel_stats.return_value = {
            "member_count": 10,
            "total_msg_count": 100
        }
        
        context = await mock_processor._build_channel_context("channel1", "team1")
        
        assert context is not None
        assert context.channel_id == "channel1"
        assert context.team_name == "Test Team"
        assert context.member_count == 10
    
    @pytest.mark.asyncio
    async def test_build_user_map(self, mock_processor, sample_user_data):
        """Test user profile map building"""
        
        sample_messages = [
            {"user_id": "user1", "message": "test1"},
            {"user_id": "user2", "message": "test2"}
        ]
        
        # Mock user info responses
        mock_processor.mattermost_client.get_user_info.side_effect = [
            sample_user_data[0],  # user1
            sample_user_data[1]   # user2
        ]
        
        user_map = await mock_processor._build_user_map(sample_messages)
        
        assert len(user_map) == 2
        assert "user1" in user_map
        assert "user2" in user_map
        assert user_map["user1"].display_name == "Al"
        assert user_map["user2"].display_name == "Bob Johnson"
    
    @pytest.mark.asyncio 
    async def test_create_enhanced_metadata(self, mock_processor, sample_messages, sample_channel_info, sample_team_info):
        """Test enhanced metadata creation"""
        
        # Create channel context
        context = ChannelContext.from_mattermost_data(sample_channel_info, sample_team_info)
        
        # Create user map
        user_map = {
            "user1": UserProfile("user1", "alice", "alice@example.com", "Alice", "Smith", "Al", "PM", "user", "en"),
            "user2": UserProfile("user2", "bob", "bob@example.com", "Bob", "Johnson", "", "Dev", "user", "en")
        }
        
        metadata = await mock_processor._create_enhanced_metadata(
            context, user_map, sample_messages
        )
        
        assert "channel_context" in metadata
        assert "user_profiles" in metadata
        assert "message_analytics" in metadata
        assert "processing_timestamp" in metadata
        
        # Check message analytics
        analytics = metadata["message_analytics"]
        assert analytics["total_messages"] == 4
        assert "hashtag_frequency" in analytics
        assert "#progress" in str(analytics["hashtag_frequency"])
    
    @pytest.mark.asyncio
    async def test_process_channel_complete_integration(self, mock_processor, sample_messages, sample_channel_info, sample_team_info, sample_user_data):
        """Test complete channel processing integration"""
        
        # Mock all the API calls
        mock_processor.mattermost_client.get_channel_info.return_value = sample_channel_info
        mock_processor.mattermost_client.get_team_info.return_value = sample_team_info
        mock_processor.mattermost_client.get_channel_stats.return_value = {"member_count": 10}
        mock_processor.mattermost_client.get_channel_history.return_value = sample_messages
        mock_processor.mattermost_client.get_user_info.side_effect = sample_user_data
        
        # Mock embedding and storage
        with patch('src.retrieval.retrievers.vector_retriever.VectorRetriever') as mock_retriever:
            mock_retriever.return_value._get_embeddings.return_value = [[0.1] * 1536] * 10
            mock_processor.qdrant_manager.upsert_points.return_value = None
            
            result = await mock_processor.process_channel_complete(
                channel_id="channel1",
                team_id="team1",
                max_messages=1000,
                include_user_profiles=True
            )
        
        assert result["success"] == True
        assert "document_id" in result
        assert "metrics" in result
        assert "channel_context" in result
        
        # Check metrics
        metrics = result["metrics"]
        assert metrics["total_messages"] == 4
        assert metrics["chunks_created"] > 0


class TestProcessingMetrics:
    """Test cases for ProcessingMetrics"""
    
    def test_to_dict(self):
        """Test metrics conversion to dictionary"""
        
        participants = {"user1", "user2", "user3"}
        time_range = (datetime.now(), datetime.now())
        
        metrics = ProcessingMetrics(
            total_messages=100,
            valid_messages=95,
            conversation_groups=5,
            threads_extracted=3,
            chunks_created=50,
            parent_chunks_created=25,
            processing_time_seconds=45.5,
            participants=participants,
            time_range=time_range
        )
        
        result_dict = metrics.to_dict()
        
        assert result_dict["total_messages"] == 100
        assert result_dict["participants"] == ["user1", "user2", "user3"]
        assert len(result_dict["time_range"]) == 2
        assert isinstance(result_dict["time_range"][0], str)  # ISO format


@pytest.mark.asyncio
async def test_end_to_end_channel_processing(sample_messages, sample_channel_info, sample_team_info):
    """End-to-end test of channel processing pipeline"""
    
    # Create chunker and process messages
    chunker = ChannelMessageChunker()
    
    child_chunks, parent_chunks = chunker.chunk_channel_messages(
        raw_messages=sample_messages,
        channel_info=sample_channel_info,
        team_info=sample_team_info,
        document_id="e2e_test"
    )
    
    # Verify results
    assert len(child_chunks) > 0
    assert len(parent_chunks) > 0
    
    # Check that chunks have proper metadata
    for chunk in child_chunks:
        assert chunk.chunk_id is not None
        assert chunk.content is not None
        assert chunk.metadata["source_type"] == "mattermost_channel_conversation"
        assert "conversation_group_id" in chunk.metadata
        assert "participants" in chunk.metadata
    
    # Check parent-child relationships
    child_parent_ids = {chunk.parent_id for chunk in child_chunks if chunk.parent_id}
    parent_ids = {chunk.chunk_id for chunk in parent_chunks}
    
    # All child parent_ids should exist in parent chunks
    assert child_parent_ids.issubset(parent_ids)
    
    # Check that conversation content is preserved
    all_content = " ".join(chunk.content for chunk in child_chunks)
    assert "authentication module" in all_content
    assert "database design" in all_content
    assert "General" in all_content  # Channel name should be included


if __name__ == "__main__":
    pytest.main([__file__, "-v"])