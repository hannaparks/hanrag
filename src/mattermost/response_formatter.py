from typing import Dict, Any, List, Optional
import re
from loguru import logger


class ResponseFormatter:
    """Formats RAG responses for Mattermost display"""
    
    def __init__(self):
        self.max_response_length = 16000  # Mattermost supports up to 16383 chars
        self.max_context_display = 3  # Max sources to show
    
    def format_rag_response(
        self,
        rag_result: Dict[str, Any],
        original_question: str,
        user_id: Optional[str] = None
    ) -> str:
        """Format RAG response for Mattermost with enhanced visual design"""
        
        try:
            response_text = rag_result.get("response", "")
            sources = rag_result.get("sources", [])
            context_count = rag_result.get("context_count", 0)
            query_type = rag_result.get("query_type", "general")
            processing_time = rag_result.get("processing_time", 0)
            generation_model = rag_result.get("generation_model", None)
            
            # Debug logging
            logger.debug(f"RAG result contains generation_model: {generation_model}")
            
            # Start with a beautiful header
            formatted = self._format_response_header(user_id, query_type)
            
            # Add the main response with better formatting
            formatted += self._format_main_response(response_text)
            
            # Skip source display for now (sources still processed internally for quality)
            # if sources and len(sources) > 0:
            #     formatted += "\n\n" + self._format_enhanced_sources(sources)
            
            # Add enhanced metadata footer
            formatted += self._format_enhanced_metadata(context_count, processing_time, generation_model)
            
            # Truncate if too long with prettier truncation
            if len(formatted) > self.max_response_length:
                formatted = self._prettier_truncate(formatted)
            
            return formatted
            
        except Exception as e:
            logger.error(f"Failed to format RAG response: {e}")
            return f"I encountered an error while formatting the response: {str(e)}"
    
    def _format_response_header(self, user_id: Optional[str], query_type: str) -> str:
        """Create a beautiful response header"""
        header = ""
        
        # User mention with style
        if user_id:
            header += f"<@{user_id}> "
        
        # Simple professional greeting
        header += "**Here's what I found:**\n\n"
        
        return header
    
    def _format_main_response(self, text: str) -> str:
        """Enhanced response text formatting"""
        
        if not text:
            return "I couldn't find relevant information to answer your question.\n\n**Try:**\n• Using different keywords\n• Being more specific\n• Asking about topics from ingested content"
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = text.strip()
        
        # Enhanced formatting
        # Make headers more prominent without emoji
        text = re.sub(r'^(#{1,3})\s+(.+)$', r'**\2**', text, flags=re.MULTILINE)
        
        # Clean bullet points
        text = re.sub(r'^[\-\*]\s+', '• ', text, flags=re.MULTILINE)
        
        # Number lists
        text = re.sub(r'^(\d+)\s*[\.\)]\s+', r'**\1.** ', text, flags=re.MULTILINE)
        
        # Code blocks - simple label
        text = re.sub(r'```([^`]+)```', r'**Code:**\n```\1```', text)
        
        # Inline code with subtle styling
        text = re.sub(r'`([^`]+)`', r'`\1`', text)
        
        return text
    
    def _format_enhanced_sources(self, sources: List[Any]) -> str:
        """Format source citations with beautiful visual design"""
        
        if not sources:
            return ""
        
        # Beautiful header with section divider
        source_text = "\n---\n\n📖 **Sources & References**\n\n"
        
        # Handle different source formats with enhanced styling
        displayed_sources = 0
        for i, source in enumerate(sources):
            if displayed_sources >= self.max_context_display:
                remaining = len(sources) - displayed_sources
                source_text += f"\n🔗 _**+{remaining} additional sources** - ask for more details if needed_\n"
                break
            
            if isinstance(source, str):
                # Simple string source with emoji
                source_text += f"📄 **{source}**\n"
            elif isinstance(source, dict):
                # Enhanced dictionary source formatting
                source_name = source.get('title', source.get('source', f'Document {i+1}'))
                source_type = source.get('type', 'document')
                citation_key = source.get('citation_key', f'#{i+1}')
                
                # Choose emoji based on source type
                emoji = self._get_source_emoji(source_type)
                
                # Format with citation key for easy reference
                if source.get('url'):
                    source_text += f"{emoji} **[{source_name}]({source['url']})** `[{citation_key}]`\n"
                else:
                    source_text += f"{emoji} **{source_name}** `[{citation_key}]`\n"
                
                # Add snippet if available
                if source.get('snippet'):
                    snippet = source['snippet'][:100] + "..." if len(source['snippet']) > 100 else source['snippet']
                    source_text += f"   💬 _{snippet}_\n"
                
                # Add enhanced relevance and authority scores
                relevance_score = source.get('relevance_score', 0)
                authority_score = source.get('authority_score', 0)
                
                if relevance_score > 0.8 or authority_score > 0.8:
                    quality_indicators = []
                    if relevance_score > 0.8:
                        quality_indicators.append("Highly relevant")
                    if authority_score > 0.8:
                        quality_indicators.append("Authoritative")
                    source_text += f"   ⭐ _{' • '.join(quality_indicators)}_\n"
                
                # Add timestamp if recent
                if source.get('timestamp'):
                    from datetime import datetime
                    try:
                        timestamp = datetime.fromisoformat(source['timestamp'].replace('Z', '+00:00'))
                        if (datetime.now() - timestamp).days < 7:
                            source_text += f"   🕒 _Recent ({timestamp.strftime('%b %d')})_\n"
                    except:
                        pass
                
            else:
                source_text += f"📋 **Reference {i+1}**\n"
            
            displayed_sources += 1
            if displayed_sources < len(sources) and displayed_sources < self.max_context_display:
                source_text += "\n"
        
        return source_text.rstrip()
    
    def _get_source_emoji(self, source_type: str) -> str:
        """Get appropriate emoji for source type"""
        emoji_map = {
            'mattermost': '💬',
            'web': '🌐',
            'document': '📋',
            'file': '📁',
            'unknown': '📖',
            'url': '🌐',
            'pdf': '📄',
            'markdown': '📝',
            'code': '💻',
            'channel': '💬',
            'message': '💭',
            'api': '⚡',
            'database': '🗃️'
        }
        return emoji_map.get(source_type.lower(), '📖')
    
    def _format_model_name(self, model_name: str) -> str:
        """Format model name for display"""
        if not model_name:
            return "Claude 3.5 Sonnet"
        
        # Model name mappings for cleaner display
        model_display_names = {
            "claude-opus-4-20250514": "Claude Opus 4",
            "claude-sonnet-4-20250514": "Claude Sonnet 4",
            "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
            "claude-3-opus-20240229": "Claude 3 Opus",
            "claude-3-sonnet-20240229": "Claude 3 Sonnet",
            "claude-3-haiku-20240307": "Claude 3 Haiku",
            "gpt-4": "GPT-4",
            "gpt-4-turbo": "GPT-4 Turbo",
            "gpt-3.5-turbo": "GPT-3.5 Turbo",
            "gpt-4o": "GPT-4o",
            "gpt-4o-mini": "GPT-4o Mini",
        }
        
        # Return mapped name or cleaned version of original
        if model_name in model_display_names:
            return model_display_names[model_name]
        
        # Clean up the raw model name
        # Remove version dates and clean formatting
        cleaned = model_name
        cleaned = re.sub(r'-\d{8}$', '', cleaned)  # Remove date suffix
        cleaned = cleaned.replace('-', ' ').title()
        
        return cleaned
    
    def _format_enhanced_metadata(self, context_count: int, processing_time: float, generation_model: Optional[str] = None) -> str:
        """Format enhanced metadata footer with beautiful styling"""
        
        footer = "\n\n---\n"
        
        # Format model name for display
        model_display = self._format_model_name(generation_model) if generation_model else "Claude 3.5 Sonnet"
        
        # System info
        footer += f"\n**Powered by** LlamaIndex + {model_display}"
        
        return footer
    
    def _prettier_truncate(self, text: str) -> str:
        """Prettier truncation with better UX"""
        
        # Find a good breaking point
        truncated = text[:self.max_response_length - 200]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        last_separator = truncated.rfind('---')
        
        break_point = max(last_period, last_newline, last_separator)
        if break_point > self.max_response_length * 0.7:  # Good break point found
            truncated = text[:break_point + 1]
        else:
            truncated = text[:self.max_response_length - 200]
        
        # Beautiful truncation notice
        truncation_notice = "\n\n---\n\n⚠️ **Response truncated**\n"
        truncation_notice += "_The full answer was too long for a single message._\n\n"
        truncation_notice += "💡 **To get the complete answer:**\n"
        truncation_notice += "▫️ Ask for specific parts: 'Tell me more about X'\n"
        truncation_notice += "▫️ Break down your question into smaller parts\n"
        truncation_notice += "▫️ Ask follow-up questions for details"
        
        return truncated + truncation_notice
    
    def format_error_response(
        self,
        error_message: str,
        error_type: str = "general",
        user_id: Optional[str] = None
    ) -> str:
        """Format beautiful error responses with helpful guidance"""
        
        formatted = ""
        
        if user_id:
            formatted += f"<@{user_id}> "
        
        error_messages = {
            "no_context": self._format_no_context_error(),
            "insufficient_context": self._format_insufficient_context_error(),
            "api_error": self._format_api_error(),
            "rate_limit": self._format_rate_limit_error(),
            "parsing_error": self._format_parsing_error(),
            "general": self._format_general_error(error_message)
        }
        
        formatted += error_messages.get(error_type, error_messages["general"])
        
        return formatted
    
    def _format_no_context_error(self) -> str:
        """Format no context found error with helpful suggestions"""
        return """🔍 **No matching information found**

I searched through all available content but couldn't find information relevant to your question.

**💡 Try these approaches:**
▫️ **Rephrase your question** - use different keywords
▫️ **Be more specific** - add context or details
▫️ **Check if content exists** - has this topic been discussed?
▫️ **Ask simpler questions** - break complex queries into parts

**📁 To expand the knowledge base:**
• Use `/inject` to add channel history
• Use `/inject <url>` to add documentation
• Share relevant documents in channels before ingesting

_Remember: I can only find information that has been ingested into the knowledge base._"""
    
    def _format_insufficient_context_error(self) -> str:
        """Format insufficient context error"""
        return """⚠️ **Limited information available**

📊 I found some potentially relevant content, but it may not fully address your question.

**💡 To get better results:**
▫️ **Ask more specific questions** about particular aspects
▫️ **Request clarification** on specific points
▫️ **Try different keywords** or phrases
▫️ **Break your question** into smaller, focused parts

_The information below represents the best match I could find:_"""
    
    def _format_api_error(self) -> str:
        """Format API error with friendly message"""
        return """🔧 **Service temporarily unavailable**

🔄 I'm experiencing technical difficulties right now.

**📋 What's happening:**
The AI service is temporarily unavailable or overloaded.

**⏰ Please try:**
▫️ **Wait 30-60 seconds** and ask again
▫️ **Simplify your question** if it was complex
▫️ **Check system status** if issues persist

🤖 _I'll be back to full service shortly!_"""
    
    def _format_rate_limit_error(self) -> str:
        """Format rate limit error with timing guidance"""
        return """⏱️ **I'm quite busy right now!**

📊 I'm currently processing many requests and need a moment to catch up.

**⏰ Please wait:**
▫️ **30-60 seconds** before trying again
▫️ **Queue complex questions** for a few minutes
▫️ **Break large questions** into smaller ones

**💡 Pro tip:** I work best with one question at a time!

_Thank you for your patience! 🙏_"""
    
    def _format_parsing_error(self) -> str:
        """Format parsing error with helpful examples"""
        return """🤔 **I need some clarification**

💬 I had trouble understanding exactly what you're asking.

**💡 Try rephrasing with:**
▫️ **Clear, specific questions**: "How do I configure X?"
▫️ **Concrete examples**: "What's the difference between A and B?"
▫️ **Step-by-step requests**: "What are the steps to do Y?"
▫️ **Focused topics**: Avoid multiple questions in one message

**✨ Example good questions:**
• `/ask How do I set up authentication?`
• `/ask What are the deployment options?`
• `/ask Explain the database configuration`

_I work best with clear, focused questions!_"""
    
    def _format_general_error(self, error_message: str) -> str:
        """Format general error with helpful context"""
        return f"""⚠️ **Something went wrong**

📊 **Technical details:**
```
{error_message}
```

**🔄 What you can try:**
▫️ **Rephrase your question** and try again
▫️ **Simplify the request** if it was complex
▫️ **Wait a moment** and retry
▫️ **Check if content exists** in the knowledge base

🤖 _If this keeps happening, there might be a technical issue I need help with._"""
    
    def format_ingestion_status(
        self,
        status: str,
        details: Dict[str, Any],
        source_type: str = "content"
    ) -> str:
        """Format beautiful ingestion status messages"""
        
        if status == "started":
            return self._format_ingestion_started(source_type, details)
        elif status == "completed":
            return self._format_ingestion_completed(details)
        elif status == "failed":
            return self._format_ingestion_failed(details)
        else:
            return f"📊 **Status Update:** {status.title()}"
    
    def _format_ingestion_started(self, source_type: str, details: Dict[str, Any]) -> str:
        """Format ingestion started message with progress indication"""
        
        if source_type == "url":
            url = details.get('url', 'Unknown URL')
            return f"""🌐 **Processing Web Content**

🔗 **Source:** `{url}`

🔄 **Progress:**
▫️ Fetching content from URL
▫️ Analyzing and parsing content
▫️ Creating searchable segments
▫️ Adding to knowledge base

⏰ _This typically takes 30-60 seconds..._
💬 _I'll notify you when it's ready for questions!_"""
            
        elif source_type == "channel":
            message_count = details.get("message_count", 0)
            return f"""💬 **Processing Channel History**

📊 **Scope:** {message_count:,} messages from this channel

🔄 **Progress:**
▫️ Extracting message content
▫️ Filtering relevant information
▫️ Creating searchable segments
▫️ Indexing conversations

⏰ _Processing channel history..._
🤖 _Your team's knowledge is being organized!_"""
        else:
            return f"""📁 **Processing Content**

🔄 **Status:** Analyzing and indexing content

▫️ Parsing document structure
▫️ Extracting key information
▫️ Creating searchable format
▫️ Adding to knowledge base

⏰ _This may take a few moments..._"""
    
    def _format_ingestion_completed(self, details: Dict[str, Any]) -> str:
        """Format successful ingestion completion"""
        
        chunks_processed = details.get("chunks_processed", 0)
        processing_time = details.get("processing_time", 0)
        source_name = details.get("source_name", "content")
        
        success_msg = f"""✨ **Content Successfully Added!**

🎉 **Great news!** Your content is now searchable across all channels.

📊 **Processing Summary:**
▫️ **Content segments:** {chunks_processed:,} pieces
▫️ **Source:** {source_name}"""
        
        if processing_time > 0:
            success_msg += f"\n▫️ **Processing time:** {processing_time:.1f} seconds"
        
        success_msg += f"""

🚀 **What's Next:**
▫️ Use `/ask <question>` to search this content
▫️ Questions work from **any channel** now
▫️ Try asking about specific topics or details

💡 **Example:** `/ask What are the main topics in this content?`

✅ _Ready for your questions!_"""
        
        return success_msg
    
    def _format_ingestion_failed(self, details: Dict[str, Any]) -> str:
        """Format ingestion failure with helpful guidance"""
        
        error = details.get("error", "Unknown error occurred")
        source_name = details.get("source_name", "content")
        
        return f"""⚠️ **Ingestion Failed**

📁 **Source:** {source_name}

📊 **What happened:**
```
{error}
```

🔧 **Common solutions:**
▫️ **Check URL accessibility** - is the link working?
▫️ **Verify permissions** - can the content be accessed?
▫️ **Try smaller content** - break large sources into parts
▫️ **Check format support** - is this a supported file type?

🔄 **Next steps:**
▫️ Fix the issue and try `/inject` again
▫️ Contact support if the problem persists
▫️ Try alternative content sources

🤖 _I'm here to help once the issue is resolved!_"""
    
    def format_help_response(self, help_type: str = "general") -> str:
        """Format beautiful help responses with enhanced guidance"""
        
        if help_type == "inject":
            return self._format_inject_help()
        elif help_type == "ask":
            return self._format_ask_help()
        else:
            return self._format_general_help()
    
    def _format_inject_help(self) -> str:
        """Format inject command help with visual examples"""
        return """📥 **Content Ingestion Guide**

**🎆 What is content ingestion?**
I help you build a searchable knowledge base from your conversations and documents!

---

**💬 Add Channel History:**
```
/inject
```
▫️ Analyzes recent messages from this channel
▫️ Extracts key information and discussions
▫️ Makes conversations searchable from any channel

**🌐 Add Web Content:**
```
/inject https://docs.example.com/api
```
▫️ Fetches content from URLs
▫️ Processes documentation and articles
▫️ Creates searchable knowledge base

---

**📚 Supported Content Types:**
🌐 Web pages and documentation sites
📋 GitHub repositories and files
📄 PDF documents and guides
📝 Markdown and text files
💬 Mattermost channel discussions
📧 API documentation and wikis

**✨ Pro Tips:**
▫️ **Start with key documentation** your team uses most
▫️ **Ingest channel history** from important discussions
▫️ **Add new content regularly** to keep knowledge fresh
▫️ **Content works across all channels** once ingested

**⏰ Processing Time:** Usually 30-90 seconds depending on content size

💡 _Once ingested, anyone can search this content from any channel!_"""
    
    def _format_ask_help(self) -> str:
        """Format ask command help with practical examples"""
        return """❓ **Smart Question Guide**

**🎆 What can I help you find?**
I search through all ingested content to answer your questions with relevant information and sources!

---

**💬 Basic Usage:**
```
/ask How do I configure authentication?
```

**🎯 Question Types That Work Great:**

📈 **Factual Questions:**
• "What is X?" or "How does Y work?"
• "What are the features of Z?"

🛠️ **How-To Questions:**
• "How do I set up X?"
• "What are the steps to do Y?"

🔄 **Comparison Questions:**
• "What's the difference between A and B?"
• "Which option is better for X?"

🔍 **Troubleshooting:**
• "Why does X fail?"
• "How do I fix Y error?"

---

**✨ Example Questions:**
▫️ `/ask What are the deployment options available?`
▫️ `/ask How do I troubleshoot connection issues?`
▫️ `/ask Compare the authentication methods`
▫️ `/ask What was discussed about the API changes?`
▫️ `/ask Show me the configuration examples`

**💡 Tips for Better Results:**
▫️ **Be specific** - include relevant keywords
▫️ **Ask one thing at a time** - break complex questions apart
▫️ **Use natural language** - ask like you're talking to a colleague
▫️ **Follow up** - ask for clarification or more details

🌐 _I search across ALL channels and ingested content to find your answers!_"""
    
    def _format_general_help(self) -> str:
        """Format general help with comprehensive overview"""
        return """🤖 **RAG Assistant - Your AI Knowledge Helper**

**🎆 What I Do:**
I help your team build and search a comprehensive knowledge base from conversations, documentation, and web content!

---

**🚀 Key Features:**

📥 **Smart Content Ingestion**
▫️ Add channel history and discussions
▫️ Import web documentation and guides
▫️ Process multiple content formats

🔍 **Intelligent Search**
▫️ Ask questions in natural language
▫️ Get answers with source citations
▫️ Search across all channels and content

🌐 **Cross-Channel Access**
▫️ Ingest in any channel, search from anywhere
▫️ Team-wide knowledge sharing
▫️ Consistent information access

---

**📝 Quick Start Guide:**

**1. 📥 Add Content:**
• `/inject` - Add current channel history
• `/inject <url>` - Add web documentation

**2. 🔍 Search & Ask:**
• `/ask <question>` - Get intelligent answers
• Follow up with more specific questions

**3. 🔄 Keep It Fresh:**
• Regularly add new discussions
• Update documentation sources
• Share knowledge across teams

---

**📚 Learn More:**
▫️ `/inject help` - Content ingestion guide
▫️ `/ask help` - Question asking tips

**🔒 Privacy & Security:**
▫️ Content stays within your team context
▫️ No external data sharing
▫️ Secure processing and storage

🎉 _Ready to make your team's knowledge instantly searchable? Start with `/inject` to add some content!_"""
    
    def format_stats_response(self, stats: Dict[str, Any]) -> str:
        """Format beautiful statistics response with insights"""
        
        total_docs = stats.get("total_documents", 0)
        total_chunks = stats.get("total_chunks", 0)
        channels = stats.get("channels", [])
        teams = stats.get("teams", [])
        last_updated = stats.get("last_updated", "Unknown")
        storage_size = stats.get("storage_size", 0)
        query_count = stats.get("query_count", 0)
        
        response = f"""📊 **Knowledge Base Analytics**

🎆 **Your Team's Knowledge at a Glance**

---

**📚 Content Overview:**
▫️ **Documents:** {total_docs:,} sources
▫️ **Searchable segments:** {total_chunks:,} pieces
▫️ **Active channels:** {len(channels)} with content"""
        
        if len(teams) > 0:
            response += f"\n▫️ **Teams using system:** {len(teams)}"
        
        if storage_size > 0:
            size_mb = storage_size / (1024 * 1024)
            if size_mb > 1024:
                size_str = f"{size_mb/1024:.1f} GB"
            else:
                size_str = f"{size_mb:.1f} MB"
            response += f"\n▫️ **Storage used:** {size_str}"
        
        response += f"\n\n**🕰️ Activity:**"
        response += f"\n▫️ **Last updated:** {last_updated}"
        
        if query_count > 0:
            response += f"\n▫️ **Questions answered:** {query_count:,}"
        
        # Add insights based on the data
        response += "\n\n**💡 Insights:**\n"
        
        if total_chunks > 10000:
            response += "✨ **Rich knowledge base** - lots of searchable content!\n"
        elif total_chunks > 1000:
            response += "📈 **Growing knowledge base** - good foundation for Q&A\n"
        else:
            response += "🌱 **Getting started** - add more content with `/inject`\n"
        
        if len(channels) > 5:
            response += "🌐 **Wide coverage** - content from multiple channels\n"
        elif len(channels) > 1:
            response += "🔗 **Multi-channel** - knowledge shared across teams\n"
        
        if query_count > 100:
            response += "💯 **Actively used** - team is getting great value!\n"
        
        response += "\n**🚀 Next Steps:**\n"
        response += "▫️ Use `/inject` to add more content\n"
        response += "▫️ Ask questions with `/ask` to test knowledge\n"
        response += "▫️ Share this system with more team members"
        
        return response
    
    def truncate_text(self, text: str, max_length: int = None) -> str:
        """Beautifully truncate text with helpful guidance"""
        
        max_length = max_length or self.max_response_length
        
        if len(text) <= max_length:
            return text
        
        # Find intelligent breaking points
        search_area = text[:max_length - 300]  # Leave room for truncation message
        
        # Look for good break points in order of preference
        break_points = [
            search_area.rfind('\n\n---\n'),  # Section dividers
            search_area.rfind('\n\n**'),    # Headers
            search_area.rfind('. '),        # Sentence ends
            search_area.rfind('\n'),        # Line breaks
        ]
        
        break_point = -1
        for bp in break_points:
            if bp > max_length * 0.6:  # Ensure we keep a good portion
                break_point = bp
                break
        
        if break_point == -1:
            break_point = max_length - 300
        
        truncated = text[:break_point]
        
        # Beautiful truncation notice
        truncation_msg = "\n\n---\n\n✂️ **Content Truncated**\n\n"
        truncation_msg += "💡 **The full response was too long for one message.**\n\n"
        truncation_msg += "**🔄 To get the complete answer:**\n"
        truncation_msg += "▫️ Ask for specific sections: _'Tell me more about X'_\n"
        truncation_msg += "▫️ Break your question into parts\n"
        truncation_msg += "▫️ Use follow-up questions for details\n\n"
        truncation_msg += "🤖 _I'm happy to provide more specific information!_"
        
        return truncated + truncation_msg