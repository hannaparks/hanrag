import asyncio
import os
from typing import Dict, Any, List
from dataclasses import dataclass
from loguru import logger
from fastapi import HTTPException

from .channel_client import MattermostClient
from .response_formatter import ResponseFormatter
from ..config.settings import settings


@dataclass
class SlashCommandRequest:
    """Represents a slash command request from Mattermost"""
    token: str
    team_id: str
    team_domain: str
    channel_id: str
    channel_name: str
    user_id: str
    user_name: str
    command: str
    text: str
    response_url: str
    trigger_id: str


class SlashCommandHandler:
    """Handles Mattermost slash commands for RAG operations"""
    
    def __init__(self, rag_pipeline=None):
        self.inject_token = settings.MATTERMOST_INJECT_TOKEN
        self.ask_token = settings.MATTERMOST_ASK_TOKEN
        self.mattermost_client = MattermostClient()
        self.response_formatter = ResponseFormatter()
        self.rag_pipeline = rag_pipeline  # Will be injected
        
    def _validate_token(self, token: str, command_type: str) -> bool:
        """Validate slash command tokens"""
        if command_type == "inject":
            return token == self.inject_token
        elif command_type == "ask":
            return token == self.ask_token
        return False
    
    async def handle_inject_command(
        self,
        token: str,
        team_id: str,
        channel_id: str,
        user_id: str,
        text: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle /inject slash command"""
        
        # Validate inject token
        if not self._validate_token(token, "inject"):
            raise HTTPException(status_code=401, detail="Invalid inject token")
        
        try:
            text_parts = text.strip().split() if text.strip() else []
            
            # Check for --config flag
            if "--config" in text_parts:
                # Determine the API server URL
                if settings.ENVIRONMENT == "production":
                    # In production, use the VPS IP (nginx proxy on port 80)
                    api_host = settings.PRODUCTION_HOST or "168.231.68.82"
                    config_url = f"http://{api_host}/config"
                else:
                    # In development, use localhost
                    config_url = f"http://localhost:{settings.SERVER_PORT}/config"
                
                return {
                    "response_type": "ephemeral",
                    "text": f"**RAG System Configuration**\n\nOpen the configuration page to manage your system settings:\n\n[{config_url}]({config_url})\n\n_Configure API keys, models, and RAG parameters._",
                    "username": "RAG Assistant",
                    "icon_emoji": ":gear:"
                }
            
            # Check for --purge flag
            elif "--purge" in text_parts:
                return await self._handle_purge_command(channel_id, team_id, user_id, text_parts)
            
            elif text.strip():
                # URL ingestion: /inject <url>
                url = text.strip()
                logger.info(f"User {user_id} requested URL ingestion: {url}")
                
                # Start ingestion process asynchronously
                asyncio.create_task(self._ingest_url(url, channel_id, team_id, user_id))
                
                return {
                    "response_type": "ephemeral",
                    "text": f"**Ingesting content from URL**\n`{url}`\n\n_Processing... This may take a few moments._",
                    "username": "RAG Assistant",
                    "icon_emoji": ":arrows_counterclockwise:"
                }
            else:
                # Channel ingestion: /inject
                logger.info(f"User {user_id} requested channel history ingestion for channel {channel_id}")
                
                # Start enhanced ingestion process asynchronously
                asyncio.create_task(self._ingest_channel_enhanced(channel_id, team_id, user_id))
                
                return {
                    "response_type": "ephemeral",
                    "text": "**Ingesting channel message history**\n\n_Processing recent messages and conversations. This may take a few moments._",
                    "username": "RAG Assistant",
                    "icon_emoji": ":arrows_counterclockwise:"
                }
                
        except Exception as e:
            logger.error(f"Inject command failed: {e}")
            return {
                "response_type": "ephemeral",
                "text": f"**Error during ingestion**\n```\n{str(e)}\n```",
                "username": "RAG Assistant",
                "icon_emoji": ":warning:"
            }
    
    async def handle_ask_command(
        self,
        token: str,
        team_id: str,
        channel_id: str,
        user_id: str,
        text: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle /ask slash command with immediate feedback"""
        
        # Validate ask token
        if not self._validate_token(token, "ask"):
            raise HTTPException(status_code=401, detail="Invalid ask token")
        
        if not text.strip():
            help_text = self.response_formatter.format_help_response("ask")
            return {
                "response_type": "ephemeral",
                "text": f"**Missing Question**\n\n{help_text}",
                "username": "RAG Assistant",
                "icon_emoji": ":question:"
            }
        
        try:
            question = text.strip()
            logger.info(f"User {user_id} asked question in channel {channel_id}: {question}")
            
            if not self.rag_pipeline:
                error_text = self.response_formatter.format_error_response(
                    "The knowledge base system is currently unavailable. Please try again later.",
                    "api_error"
                )
                return {
                    "response_type": "ephemeral",
                    "text": error_text,
                    "username": "RAG Assistant",
                    "icon_emoji": ":warning:"
                }
            
            # Start async processing and provide immediate feedback
            asyncio.create_task(
                self._process_question_async(
                    question, channel_id, team_id, user_id
                )
            )
            
            # Return immediate feedback to clear the input
            return {
                "response_type": "in_channel",
                "text": f"**Processing your question:** \n> {question}\n\n_Searching the knowledge base..._",
                "username": "RAG Assistant",
                "icon_emoji": ":mag:"
            }
            
        except Exception as e:
            logger.error(f"Ask command failed: {e}")
            # Enhanced error formatting
            error_text = self.response_formatter.format_error_response(
                str(e),
                "api_error",
                user_id
            )
            
            return {
                "response_type": "ephemeral",
                "text": error_text,
                "username": "RAG Assistant",
                "icon_emoji": ":warning:"
            }
    
    async def _ingest_url(self, url: str, channel_id: str, team_id: str, user_id: str):
        """Ingest content from URL (async background task)"""
        
        try:
            if not self.rag_pipeline:
                logger.error("RAG pipeline not available for URL ingestion")
                return
            
            logger.info(f"Starting URL ingestion: {url}")
            
            # Use the ingestion pipeline to process the URL
            result = await self.rag_pipeline.ingest_url(
                url=url,
                metadata={
                    "channel_id": channel_id,
                    "team_id": team_id,
                    "ingested_by": user_id,
                    "source_type": "url"
                }
            )
            
            if result.get("success"):
                logger.info(f"URL ingestion completed: {url}")
                # Could send a follow-up message here if needed
            else:
                logger.error(f"URL ingestion failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"URL ingestion background task failed: {e}")
    
    async def _ingest_channel_history(self, channel_id: str, team_id: str, user_id: str):
        """Ingest channel message history (async background task)"""
        
        try:
            if not self.rag_pipeline:
                logger.error("RAG pipeline not available for channel ingestion")
                return
            
            logger.info(f"Starting channel history ingestion: {channel_id}")
            
            # Get channel messages
            messages = await self.mattermost_client.get_channel_history(
                channel_id=channel_id,
                max_messages=1000
            )
            
            if not messages:
                logger.warning(f"No messages found in channel {channel_id}")
                return
            
            # Get channel info for context
            channel_info = await self.mattermost_client.get_channel_info(channel_id)
            team_info = await self.mattermost_client.get_team_info(team_id)
            
            # Use the ingestion pipeline to process the messages
            result = await self.rag_pipeline.ingest_channel_messages(
                messages=messages,
                channel_info=channel_info,
                team_info=team_info,
                metadata={
                    "channel_id": channel_id,
                    "team_id": team_id,
                    "ingested_by": user_id,
                    "source_type": "channel_history"
                }
            )
            
            if result.get("success"):
                logger.info(f"Channel ingestion completed: {channel_id}, processed {len(messages)} messages")
            else:
                logger.error(f"Channel ingestion failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Channel ingestion background task failed: {e}")
    
    async def _ingest_channel_enhanced(self, channel_id: str, team_id: str, user_id: str):
        """Enhanced channel ingestion with conversation analysis (async background task)"""
        
        try:
            if not self.rag_pipeline:
                logger.error("RAG pipeline not available for enhanced channel ingestion")
                return
            
            logger.info(f"Starting enhanced channel history ingestion: {channel_id}")
            
            # Use the enhanced channel ingestion pipeline
            result = await self.rag_pipeline.ingest_channel_enhanced(
                channel_id=channel_id,
                team_id=team_id,
                max_messages=1000,
                include_user_profiles=True,
                incremental=False
            )
            
            if result.get("success"):
                metrics = result.get("metrics", {})
                channel_context = result.get("channel_context", {})
                
                logger.info(f"Enhanced channel ingestion completed: {channel_id}")
                logger.info(f"Processed {metrics.get('conversation_groups', 0)} conversation groups, "
                           f"{metrics.get('chunks_created', 0)} chunks created")
                
                # Could send a follow-up message with detailed results
                channel_name = channel_context.get("display_name", "Unknown")
                logger.info(f"Enhanced ingestion results for #{channel_name}: "
                           f"{metrics.get('total_messages', 0)} messages, "
                           f"{metrics.get('threads_extracted', 0)} threads, "
                           f"{len(metrics.get('participants', []))} participants")
                
            else:
                logger.error(f"Enhanced channel ingestion failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Enhanced channel ingestion background task failed: {e}")
    
    async def _handle_purge_command(
        self, 
        channel_id: str, 
        team_id: str, 
        user_id: str, 
        text_parts: List[str]
    ) -> Dict[str, Any]:
        """Handle database purge commands with safety checks"""
        
        try:
            if not self.rag_pipeline:
                return {
                    "response_type": "ephemeral",
                    "text": "**RAG pipeline not available**\n\nPurge operation cannot be performed.",
                    "username": "RAG Assistant",
                    "icon_emoji": ":warning:"
                }
            
            # Parse purge options
            is_preview = "--preview" in text_parts
            is_confirmed = "--confirm" in text_parts
            is_channel_only = "--channel" in text_parts
            is_team_only = "--team" in text_parts
            
            # Build filters
            filters = {}
            if is_channel_only:
                filters["channel_id"] = channel_id
            elif is_team_only:
                filters["team_id"] = team_id
            
            # Get filter description for display
            filter_desc = ""
            if is_channel_only:
                filter_desc = f" from **this channel**"
            elif is_team_only:
                filter_desc = f" from **this team**"
            else:
                filter_desc = " from **entire database**"
            
            if is_preview or not is_confirmed:
                # Show preview of what would be purged
                logger.info(f"User {user_id} requested purge preview{filter_desc}")
                
                preview = await self.rag_pipeline.purge_database(
                    filters=filters,
                    preview_only=True
                )
                
                if not preview.get("success"):
                    return {
                        "response_type": "ephemeral",
                        "text": f"**Purge Preview Failed**\n```\n{preview.get('error', 'Unknown error')}\n```",
                        "username": "RAG Assistant",
                        "icon_emoji": ":warning:"
                    }
                
                points_text = f"{preview['points_to_delete']} chunks" if preview['points_to_delete'] != 1 else "1 chunk"
                
                confirm_text = ""
                if preview["points_to_delete"] > 0 and not is_preview:
                    confirm_text = f"\n\n**To confirm this purge, use:**\n`/inject --purge --confirm{' --channel' if is_channel_only else ' --team' if is_team_only else ''}`"
                
                return {
                    "response_type": "ephemeral",
                    "text": f"**Purge Preview{filter_desc.title()}**\n\n" +
                           f"**Current:** {preview['total_points']} total chunks in database\n" +
                           f"**Will Delete:** {points_text}\n" +
                           f"**Operation:** {preview['operation']}\n\n" +
                           f"_Warning: {preview.get('warning', 'This operation cannot be undone!')}_" +
                           confirm_text,
                    "username": "RAG Assistant",
                    "icon_emoji": ":warning:"
                }
            
            else:
                # Perform actual purge
                logger.warning(f"User {user_id} confirmed database purge{filter_desc}")
                
                # Double-check with actual purge (requires confirmation)
                result = await self.rag_pipeline.purge_database(
                    filters=filters,
                    confirm_purge=True
                )
                
                if result.get("success"):
                    deleted_text = f"{result['deleted_count']} chunks" if result['deleted_count'] != 1 else "1 chunk"
                    
                    return {
                        "response_type": "ephemeral",
                        "text": f"**Database Purged Successfully{filter_desc.title()}**\n\n" +
                               f"**Deleted:** {deleted_text}\n" +
                               f"**Operation:** {result.get('operation', 'purge')}\n\n" +
                               f"_The knowledge base has been cleaned up._",
                        "username": "RAG Assistant",
                        "icon_emoji": ":white_check_mark:"
                    }
                else:
                    return {
                        "response_type": "ephemeral",
                        "text": f"**Purge Failed**\n```\n{result.get('error', 'Unknown error')}\n```",
                        "username": "RAG Assistant",
                        "icon_emoji": ":warning:"
                    }
                    
        except Exception as e:
            logger.error(f"Purge command failed: {e}")
            return {
                "response_type": "ephemeral",
                "text": f"**Purge Command Error**\n```\n{str(e)}\n```",
                "username": "RAG Assistant",
                "icon_emoji": ":warning:"
            }
    
    async def get_help_response(self, command_type: str = "general") -> Dict[str, Any]:
        """Get help response for commands"""
        
        if command_type == "inject":
            help_text = """**📥 Inject Command Help**

**Usage:**
• `/inject` - Ingest current channel's message history  
• `/inject <url>` - Ingest content from a URL
• `/inject --config` - Open the configuration page
• `/inject --purge [options]` - Purge (delete) data from knowledge base

**Examples:**
• `/inject` - Add recent channel messages to knowledge base
• `/inject https://docs.example.com/api` - Add documentation from URL
• `/inject https://github.com/user/repo/blob/main/README.md` - Add README content
• `/inject --config` - Configure API keys, models, and RAG parameters

**Purge Commands:**
• `/inject --purge` - Preview what would be deleted from entire database
• `/inject --purge --channel` - Preview deletion from current channel only
• `/inject --purge --team` - Preview deletion from current team only
• `/inject --purge --confirm` - **Actually delete** all data (⚠️ permanent!)
• `/inject --purge --channel --confirm` - Delete current channel data only

**Supported Content:**
• Web pages (HTML)
• Documentation sites
• GitHub files
• PDF documents (via URL)
• Text files
• Markdown files

**⚠️ Warning:** Purge operations cannot be undone! Always use preview first.
**Note:** Content ingestion may take a few moments depending on size."""

        elif command_type == "ask":
            help_text = """**❓ Ask Command Help**

**Usage:**
• `/ask <your question>` - Ask questions about ingested content

**Examples:**
• `/ask What are the main features discussed in this channel?`
• `/ask How do I configure the authentication system?`
• `/ask What issues were discussed yesterday?`
• `/ask Compare the different deployment options`

**Tips:**
• Be specific in your questions for better results
• Ask about topics that have been discussed in the channel
• Use follow-up questions to get more detailed information
• Questions can reference ingested documentation or URLs"""

        else:
            help_text = """**🤖 RAG Assistant Help**

**Available Commands:**

**📥 /inject** - Add content to knowledge base
• `/inject` - Add channel message history
• `/inject <url>` - Add content from URL
• `/inject --config` - Open configuration page

**❓ /ask** - Query the knowledge base  
• `/ask <question>` - Ask about ingested content

**Getting Started:**
1. Use `/inject` to add channel messages to the knowledge base
2. Use `/inject <url>` to add external documentation
3. Use `/ask <question>` to query the knowledge base
4. Use `/inject --config` to configure system settings

**Need specific help?**
• `/inject help` - Help with ingestion
• `/ask help` - Help with questions"""

        return {
            "response_type": "ephemeral",
            "text": help_text,
            "username": "RAG Assistant",
            "icon_emoji": ":information_source:"
        }
    
    async def _process_question_async(
        self,
        question: str,
        channel_id: str,
        team_id: str,
        user_id: str
    ):
        """Process the question asynchronously and post the result"""
        
        try:
            logger.info(f"Starting async processing for question: {question}")
            
            # Query across all channels (no filtering)
            response = await self.rag_pipeline.query(question)
            
            # Format response for Mattermost
            formatted_response = self.response_formatter.format_rag_response(
                response,
                question,
                user_id
            )
            
            # Post the answer to the channel
            await self.mattermost_client.post_message(
                channel_id=channel_id,
                message=formatted_response
            )
            
            logger.info(f"Posted answer for question: {question[:50]}...")
            
        except Exception as e:
            logger.error(f"Async question processing failed: {e}")
            
            # Post error message to channel
            error_text = self.response_formatter.format_error_response(
                str(e),
                "processing_error",
                user_id
            )
            
            try:
                await self.mattermost_client.post_message(
                    channel_id=channel_id,
                    message=f"**Error processing your question:**\n{error_text}"
                )
            except Exception as post_error:
                logger.error(f"Failed to post error message: {post_error}")
    
    def set_rag_pipeline(self, rag_pipeline):
        """Set the RAG pipeline instance"""
        self.rag_pipeline = rag_pipeline