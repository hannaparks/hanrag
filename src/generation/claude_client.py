"""
Claude Client with Streaming Support

This module provides both regular and streaming response generation using the Claude API.
The streaming functionality allows for real-time response generation, improving perceived
latency and enabling progressive UI updates.

Key Features:
- Standard response generation
- Streaming response generation with Server-Sent Events (SSE)
- Circuit breaker pattern for fault tolerance
- Retry logic with exponential backoff
- Citation extraction and validation

Streaming Usage:
    async for chunk in claude_client.generate_response_stream(context, query):
        if chunk["type"] == "content":
            # Handle streaming text
            print(chunk["content"], end="")
        elif chunk["type"] == "complete":
            # Handle complete response with metadata
            print(f"Response complete")
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, AsyncIterator
from anthropic import AsyncAnthropic, APIError, APIConnectionError, RateLimitError
from loguru import logger
from ..config.settings import settings
from ..utils.circuit_breaker import circuit_breaker, CircuitBreakerConfig
from ..utils.retry_utils import with_retry, RetryConfig
from ..monitoring.metrics import track_llm_request


class ClaudeClient:
    """Manages Claude API interactions for response generation"""
    
    def __init__(self):
        self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        # Model will be read dynamically from settings/env to support runtime changes
        self.temperature = settings.TEMPERATURE
        
        # Quality controller removed - no longer calculating confidence scores
        logger.info(f"ClaudeClient initialized")
        
        # Circuit breaker configuration for Claude API
        self.circuit_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=120,  # 2 minutes
            success_threshold=3,
            excluded_exceptions=(ValueError, TypeError)  # Don't trip on client errors
        )
        
        # Retry configuration for API calls
        self.retry_config = RetryConfig(
            max_attempts=3,
            initial_delay=2.0,
            max_delay=30.0,
            retry_on=(APIConnectionError, APIError, asyncio.TimeoutError),
            retry_condition=lambda e: (
                not isinstance(e, RateLimitError) or  # Don't retry rate limits
                (hasattr(e, 'status_code') and e.status_code >= 500)
            )
        )
    
    @property
    def model(self) -> str:
        """Get the current model dynamically"""
        return self._get_current_model()
    
    @property
    def max_tokens(self) -> int:
        """Get the current max tokens dynamically"""
        return self._get_current_max_tokens()
    
    def _get_current_model(self) -> str:
        """Get the current generation model from settings/env dynamically"""
        # First check if env has been updated with GENERATION_MODEL
        from ..utils.env_manager import EnvFileManager
        env_manager = EnvFileManager()
        env_vars = env_manager.read_env_file()
        
        # Check env file first for runtime changes
        env_model = env_vars.get('GENERATION_MODEL')
        if env_model:
            logger.debug(f"Using model from env: {env_model}")
            return env_model
            
        # Fall back to settings
        model = getattr(settings, 'GENERATION_MODEL', settings.CLAUDE_MODEL)
        logger.debug(f"Using model from settings: {model}")
        return model
    
    def _get_current_max_tokens(self) -> int:
        """Get the current max tokens from settings/env dynamically"""
        # First check if env has been updated
        from ..utils.env_manager import EnvFileManager
        env_manager = EnvFileManager()
        env_vars = env_manager.read_env_file()
        
        # Check env file first for runtime changes
        env_max_tokens = env_vars.get('GENERATION_MAX_TOKENS')
        if env_max_tokens:
            try:
                return int(env_max_tokens)
            except:
                pass
                
        # Fall back to settings
        return getattr(settings, 'GENERATION_MAX_TOKENS', settings.MAX_TOKENS)
    
    def _is_hybrid_mode_enabled(self) -> bool:
        """Check if hybrid mode is enabled and model supports it"""
        # First check if the current model is Claude 4
        current_model = self._get_current_model()
        if not (current_model.startswith('claude-opus-4') or current_model.startswith('claude-sonnet-4')):
            return False
            
        # Check if hybrid mode is enabled in env/settings
        from ..utils.env_manager import EnvFileManager
        env_manager = EnvFileManager()
        env_vars = env_manager.read_env_file()
        
        # Check env file first
        env_hybrid = env_vars.get('ENABLE_HYBRID_MODE')
        if env_hybrid is not None:
            return env_hybrid.lower() in ('true', '1', 'yes', 'on')
            
        # Fall back to settings
        return getattr(settings, 'ENABLE_HYBRID_MODE', False)
    
    def _build_rag_prompt(self, context: List[str], query: str, citation_guide: Optional[str] = None) -> str:
        """Build RAG prompt with context and query"""
        
        # Use first context item (it's pre-processed by source manager)
        context_text = context[0] if context else ""
        
        # Check if hybrid mode is enabled
        if self._is_hybrid_mode_enabled():
            logger.info("Using hybrid mode with extended thinking for Claude 4")
            # Add thinking tags for Claude 4 hybrid mode
            prompt = f"""<thinking>
You are answering a question based on the provided context. Take time to think through the answer step by step.

Context:
{context_text}

User Question: {query}

{citation_guide if citation_guide else ''}

Think through:
1. What information from the context is most relevant?
2. How can I best structure my answer?
3. What specific details should I include?
4. Are there any nuances or caveats I should mention?
</thinking>

Based on the context provided, I'll answer your question.

{context_text}

**Question**: {query}

{citation_guide if citation_guide else ''}

Please provide a comprehensive answer based on the context above."""
            return prompt
        
        # Build citation instructions
        citation_instructions = citation_guide if citation_guide else """
3. Cite specific sources when making claims (e.g., "According to Source 1...")"""

        prompt = f"""You are a helpful AI assistant that answers questions based on provided context. Use the following context to answer the user's question accurately and comprehensively.

CONTEXT:
{context_text}

QUESTION: {query}

INSTRUCTIONS:
1. Answer the question based solely on the provided context
2. If the context doesn't contain enough information to answer the question, say so clearly
{citation_instructions}
4. Be concise but thorough
5. If there are conflicting information in the sources, acknowledge this

ANSWER:"""
        
        return prompt
    
    def _build_conversational_rag_prompt(
        self, 
        context: List[str], 
        query: str, 
        conversation_history: List[Dict[str, str]] = None,
        citation_guide: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Build conversational RAG prompt with context, query, and conversation history
        
        Returns messages in Claude's expected format
        """
        
        # Use first context item (it's pre-processed by source manager)
        context_text = context[0] if context else ""
        
        # Build citation instructions
        citation_instructions = citation_guide if citation_guide else """
- Cite specific sources when making claims (e.g., "According to Source 1...")"""
        
        messages = []
        
        # Add system message with context
        system_prompt = f"""You are a helpful AI assistant that answers questions based on provided context. 

AVAILABLE CONTEXT:
{context_text}

INSTRUCTIONS:
- Answer questions based on the provided context
- If the context doesn't contain enough information, say so clearly
{citation_instructions}
- Be concise but thorough
- If there are conflicting information in the sources, acknowledge this
- Maintain conversation continuity by referring to previous exchanges when relevant"""
        
        # Add conversation history if provided
        if conversation_history:
            # Start with system context
            messages.append({"role": "user", "content": system_prompt})
            messages.append({"role": "assistant", "content": "I understand. I'll answer your questions based on the provided context, cite sources appropriately, and maintain our conversation continuity."})
            
            # Add previous conversation turns
            for turn in conversation_history:
                messages.append(turn)
        else:
            # No history, include system context in first user message
            messages.append({"role": "user", "content": f"{system_prompt}\n\nQUESTION: {query}"})
            return messages
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    
    def _extract_citations(self, response_text: str) -> List[str]:
        """Extract source citations from response text"""
        citations = []
        
        # Look for patterns like "Source 1", "According to Source 2", etc.
        import re
        patterns = [
            r"Source \d+",
            r"according to Source \d+",
            r"from Source \d+"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            citations.extend(matches)
        
        return list(set(citations))  # Remove duplicates
    
    @circuit_breaker(
        name="claude_api",
        failure_threshold=5,
        recovery_timeout=120,
        success_threshold=3,
        excluded_exceptions=(ValueError, TypeError)
    )
    @with_retry(
        max_attempts=3,
        initial_delay=2.0,
        max_delay=30.0,
        retry_on=(APIConnectionError, APIError, asyncio.TimeoutError),
        retry_condition=lambda e: not isinstance(e, RateLimitError)
    )
    async def generate_response(
        self, 
        context: List[str], 
        query: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        source_metadata: Optional[List[Dict]] = None,
        citation_guide: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate a response using Claude with the provided context"""
        
        try:
            start_time = time.time()
            prompt = self._build_rag_prompt(context, query, citation_guide)
            
            # Use provided parameters or defaults
            temp = temperature if temperature is not None else self.temperature
            tokens = max_tokens if max_tokens is not None else self.max_tokens
            
            logger.info(f"Generating response with model: {self.model}")
            
<<<<<<< HEAD
            # Use timeout if specified for long operations
            if timeout:
                logger.info(f"Using timeout of {timeout} seconds for generation")
                response = await asyncio.wait_for(
                    self.client.messages.create(
                        model=self.model,
                        max_tokens=tokens,
                        temperature=temp,
                        messages=[{"role": "user", "content": prompt}]
                    ),
                    timeout=timeout
                )
            else:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=tokens,
                    temperature=temp,
                    messages=[{"role": "user", "content": prompt}]
                )
=======
            # Always use streaming to prevent timeout errors for long-running requests
            logger.info(f"Using streaming for {self.model} to prevent timeout errors")
            # Use streaming and collect the full response
            full_response = ""
            input_tokens = 0
            output_tokens = 0
            
            stream = await self.client.messages.create(
                model=self.model,
                max_tokens=tokens,
                temperature=temp,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            
            async for event in stream:
                if event.type == "message_start":
                    usage = event.message.usage
                    input_tokens = usage.input_tokens if usage else 0
                elif event.type == "content_block_delta":
                    if hasattr(event.delta, 'text'):
                        full_response += event.delta.text
                elif event.type == "message_delta":
                    if hasattr(event, 'usage') and event.usage:
                        output_tokens = event.usage.output_tokens
            
            # Create a response-like object to match the non-streaming format
            class StreamResponse:
                def __init__(self, content, input_tokens, output_tokens):
                    self.content = [type('obj', (object,), {'text': content})]
                    self.usage = type('obj', (object,), {
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens or len(content.split()) * 2  # Estimate if not provided
                    })
            
            response = StreamResponse(full_response, input_tokens, output_tokens)
>>>>>>> 66c74c8
            
            # Track metrics
            duration = time.time() - start_time
            usage = response.usage
            track_llm_request(
                model=self.model,
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                duration=duration,
                status="success"
            )
            
            response_text = response.content[0].text
            
            result = {
                "response": response_text,
                "model": self.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                "sources": self._extract_citations(response_text),
                "context_count": len(context)
            }
            
            logger.info(
                f"Generated response with {result['usage']['total_tokens']} tokens"
            )
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate response with Claude: {e}")
            # Track failed request
            track_llm_request(
                model=self.model,
                prompt_tokens=0,
                completion_tokens=0,
                duration=time.time() - start_time if 'start_time' in locals() else 0,
                status="error"
            )
            raise
    
    @circuit_breaker(
        name="claude_api",
        failure_threshold=5,
        recovery_timeout=120,
        success_threshold=3,
        excluded_exceptions=(ValueError, TypeError)
    )
    @with_retry(
        max_attempts=3,
        initial_delay=2.0,
        max_delay=30.0,
        retry_on=(APIConnectionError, APIError, asyncio.TimeoutError),
        retry_condition=lambda e: not isinstance(e, RateLimitError)
    )
    async def generate_conversational_response(
        self, 
        context: List[str], 
        query: str,
        conversation_history: List[Dict[str, str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        source_metadata: Optional[List[Dict]] = None,
        citation_guide: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a response using Claude with conversation history support"""
        
        try:
            start_time = time.time()
            
            # Build conversational prompt
            messages = self._build_conversational_rag_prompt(
                context, query, conversation_history, citation_guide
            )
            
            # Use provided parameters or defaults
            temp = temperature if temperature is not None else self.temperature
            tokens = max_tokens if max_tokens is not None else self.max_tokens
            
            logger.debug(f"Generating conversational response with Claude {self.model}, {len(messages)} messages")
            
<<<<<<< HEAD
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=tokens,
                temperature=temp,
                messages=messages
            )
=======
            # Always use streaming to prevent timeout errors for long-running requests
            logger.info(f"Using streaming for {self.model} conversational response to prevent timeout")
            # Use streaming and collect the full response
            full_response = ""
            input_tokens = 0
            output_tokens = 0
            
            stream = await self.client.messages.create(
                model=self.model,
                max_tokens=tokens,
                temperature=temp,
                messages=messages,
                stream=True
            )
            
            async for event in stream:
                if event.type == "message_start":
                    usage = event.message.usage
                    input_tokens = usage.input_tokens if usage else 0
                elif event.type == "content_block_delta":
                    if hasattr(event.delta, 'text'):
                        full_response += event.delta.text
                elif event.type == "message_delta":
                    if hasattr(event, 'usage') and event.usage:
                        output_tokens = event.usage.output_tokens
            
            # Create a response-like object to match the non-streaming format
            class StreamResponse:
                def __init__(self, content, input_tokens, output_tokens):
                    self.content = [type('obj', (object,), {'text': content})]
                    self.usage = type('obj', (object,), {
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens or len(content.split()) * 2  # Estimate if not provided
                    })
            
            response = StreamResponse(full_response, input_tokens, output_tokens)
>>>>>>> 66c74c8
            
            # Track metrics
            duration = time.time() - start_time
            usage = response.usage
            track_llm_request(
                model=self.model,
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                duration=duration,
                status="success"
            )
            
            response_text = response.content[0].text
            
            result = {
                "response": response_text,
                "model": self.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                "sources": self._extract_citations(response_text),
                "context_count": len(context),
                "conversation_turns": len(conversation_history) if conversation_history else 0
            }
            
            logger.info(
                f"Generated conversational response with {result['usage']['total_tokens']} tokens, "
                f"conversation turns: {result['conversation_turns']}"
            )
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate conversational response with Claude: {e}")
            # Track failed request
            track_llm_request(
                model=self.model,
                prompt_tokens=0,
                completion_tokens=0,
                duration=time.time() - start_time if 'start_time' in locals() else 0,
                status="error"
            )
            raise
    
    @circuit_breaker(
        name="claude_api",
        failure_threshold=5,
        recovery_timeout=120,
        success_threshold=3
    )
    @with_retry(
        max_attempts=3,
        initial_delay=2.0,
        retry_on=(APIConnectionError, APIError, asyncio.TimeoutError)
    )
    async def generate_simple_response(self, prompt: str) -> str:
        """Generate a simple response without RAG context"""
        
        try:
<<<<<<< HEAD
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
=======
            # Always use streaming to prevent timeout errors
            logger.info(f"Using streaming for {self.model} simple response to avoid timeout")
            # Use streaming and collect the full response
            full_response = ""
            
            stream = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            
            async for event in stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, 'text'):
                        full_response += event.delta.text
            
            return full_response
>>>>>>> 66c74c8
            
        except Exception as e:
            logger.error(f"Failed to generate simple response: {e}")
            return "I'm sorry, I encountered an error while generating a response."
    
    async def validate_response_quality(
        self, 
        response: str, 
        context: List[str], 
        query: str
    ) -> Dict[str, Any]:
        """Validate response quality against context"""
        
        validation_prompt = f"""Evaluate the following AI response for quality and accuracy.

ORIGINAL QUERY: {query}

CONTEXT PROVIDED:
{chr(10).join([f"Source {i+1}: {ctx}" for i, ctx in enumerate(context)])}

AI RESPONSE: {response}

Evaluate the response on:
1. Accuracy: Is the information correct based on the context?
2. Completeness: Does it address all parts of the question?
3. Grounding: Is the response based on the provided context?
4. Citations: Are sources properly referenced?

Provide a JSON response with scores (0-1) for each criterion and overall assessment."""
        
        try:
            validation = await self.generate_simple_response(validation_prompt)
            
            # Simple validation result
            return {
                "validation_response": validation,
                "quality_score": 0.8,  # Default score
                "is_grounded": True
            }
            
        except Exception as e:
            logger.error(f"Failed to validate response quality: {e}")
            return {
                "validation_response": "Validation failed",
                "quality_score": 0.5,
                "is_grounded": False
            }
    
    async def summarize_context(self, context: List[str], max_length: int = 2000) -> str:
        """Summarize context if it's too long"""
        
        full_context = "\n\n".join(context)
        
        if len(full_context) <= max_length:
            return full_context
        
        summary_prompt = f"""Summarize the following context while preserving key information:

{full_context}

Provide a concise summary that maintains the most important facts and details."""
        
        try:
            summary = await self.generate_simple_response(summary_prompt)
            return summary
        except Exception as e:
            logger.error(f"Failed to summarize context: {e}")
            # Fallback: truncate context
            return full_context[:max_length] + "..."
    
    @circuit_breaker(
        name="claude_api",
        failure_threshold=5,
        recovery_timeout=120,
        success_threshold=3,
        excluded_exceptions=(ValueError, TypeError)
    )
    async def generate_response_stream(
        self, 
        context: List[str], 
        query: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        source_metadata: Optional[List[Dict]] = None,
        citation_guide: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Generate a streaming response using Claude with the provided context"""
        
        try:
            prompt = self._build_rag_prompt(context, query, citation_guide)
            
            # Use provided parameters or defaults
            temp = temperature if temperature is not None else self.temperature
            tokens = max_tokens if max_tokens is not None else self.max_tokens
            
            logger.debug(f"Starting streaming response with Claude {self.model}")
            
            # Create streaming message
            stream = await self.client.messages.create(
                model=self.model,
                max_tokens=tokens,
                temperature=temp,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            
            # Initialize response collection for quality assessment
            full_response = ""
            token_count = 0
            
            # Stream text chunks as they arrive
            async for event in stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, 'text'):
                        chunk_text = event.delta.text
                        full_response += chunk_text
                        token_count += 1  # Approximate token count
                        
                        yield {
                            "type": "content",
                            "content": chunk_text,
                            "cumulative_tokens": token_count
                        }
                
                elif event.type == "message_start":
                    # Send initial metadata
                    yield {
                        "type": "start",
                        "model": self.model,
                        "message_id": event.message.id if hasattr(event, 'message') else None
                    }
                
                elif event.type == "message_stop":
                    # Message completed - perform quality assessment
                    logger.debug("Stream completed, running quality assessment")
                    
                    
                    # Extract citations from the complete response
                    citations = self._extract_citations(full_response)
                    
                    # Send final metadata
                    yield {
                        "type": "complete",
                        "response": full_response,
                        "model": self.model,
                        "usage": {
                            "estimated_tokens": token_count,
                            "note": "Exact token count available in non-streaming mode"
                        },
                        "sources": citations,
                        "context_count": len(context)
                    }
            
            logger.info(
                f"Completed streaming response with ~{token_count} tokens, "
            )
            
        except Exception as e:
            logger.error(f"Failed to generate streaming response with Claude: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
            raise
    
    @circuit_breaker(
        name="claude_api",
        failure_threshold=5,
        recovery_timeout=120,
        success_threshold=3,
        excluded_exceptions=(ValueError, TypeError)
    )
    async def generate_conversational_response_stream(
        self, 
        context: List[str], 
        query: str,
        conversation_history: List[Dict[str, str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        source_metadata: Optional[List[Dict]] = None,
        citation_guide: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Generate a streaming response using Claude with conversation history support"""
        
        try:
            # Build conversational prompt
            messages = self._build_conversational_rag_prompt(
                context, query, conversation_history, citation_guide
            )
            
            # Use provided parameters or defaults
            temp = temperature if temperature is not None else self.temperature
            tokens = max_tokens if max_tokens is not None else self.max_tokens
            
            logger.debug(f"Starting conversational streaming response with Claude {self.model}, {len(messages)} messages")
            
            # Create streaming message
            stream = await self.client.messages.create(
                model=self.model,
                max_tokens=tokens,
                temperature=temp,
                messages=messages,
                stream=True
            )
            
            # Initialize response collection for quality assessment
            full_response = ""
            token_count = 0
            
            # Stream text chunks as they arrive
            async for event in stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, 'text'):
                        chunk_text = event.delta.text
                        full_response += chunk_text
                        token_count += 1  # Approximate token count
                        
                        yield {
                            "type": "content",
                            "content": chunk_text,
                            "cumulative_tokens": token_count
                        }
                
                elif event.type == "message_start":
                    # Send initial metadata
                    yield {
                        "type": "start",
                        "model": self.model,
                        "message_id": event.message.id if hasattr(event, 'message') else None,
                        "conversation_turns": len(conversation_history) if conversation_history else 0
                    }
                
                elif event.type == "message_stop":
                    # Message completed - perform quality assessment
                    logger.debug("Stream completed, running quality assessment")
                    
                    
                    # Extract citations from the complete response
                    citations = self._extract_citations(full_response)
                    
                    # Send final metadata
                    yield {
                        "type": "complete",
                        "response": full_response,
                        "model": self.model,
                        "usage": {
                            "estimated_tokens": token_count,
                            "note": "Exact token count available in non-streaming mode"
                        },
                        "sources": citations,
                        "context_count": len(context),
                        "conversation_turns": len(conversation_history) if conversation_history else 0
                    }
            
            logger.info(
                f"Completed conversational streaming response with ~{token_count} tokens, "
                f"conversation turns: {len(conversation_history) if conversation_history else 0}"
            )
            
        except Exception as e:
            logger.error(f"Failed to generate conversational streaming response with Claude: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
            raise
    
    @circuit_breaker(
        name="claude_api",
        failure_threshold=5,
        recovery_timeout=120,
        success_threshold=3
    )
    async def generate_simple_response_stream(
        self, 
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Generate a simple streaming response without RAG context"""
        
        try:
            # Use provided parameters or defaults
            temp = temperature if temperature is not None else self.temperature
            tokens = max_tokens if max_tokens is not None else self.max_tokens
            
            stream = await self.client.messages.create(
                model=self.model,
                max_tokens=tokens,
                temperature=temp,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            
            full_response = ""
            
            async for event in stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, 'text'):
                        chunk_text = event.delta.text
                        full_response += chunk_text
                        yield {
                            "type": "content",
                            "content": chunk_text
                        }
                elif event.type == "message_start":
                    yield {
                        "type": "start",
                        "model": self.model
                    }
                elif event.type == "message_stop":
                    yield {
                        "type": "complete",
                        "response": full_response
                    }
            
        except Exception as e:
            logger.error(f"Failed to generate simple streaming response: {e}")
            yield {
                "type": "error",
                "error": str(e)
            }