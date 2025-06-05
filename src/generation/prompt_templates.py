from typing import List, Dict, Any, Optional


class PromptTemplates:
    """Collection of prompt templates for different RAG scenarios"""
    
    @staticmethod
    def rag_system_prompt() -> str:
        """Base system prompt for RAG responses"""
        return """You are a helpful AI assistant that answers questions based on provided context. 
You have access to a knowledge base and should use the provided context to give accurate, helpful responses.

Guidelines:
- Always base your answers on the provided context
- If the context doesn't contain enough information, acknowledge this limitation
- Cite sources when making specific claims
- Be concise but comprehensive
- If there are multiple perspectives in the sources, present them fairly
- Acknowledge uncertainty when appropriate"""
    
    @staticmethod
    def build_rag_prompt(
        context: List[str], 
        query: str, 
        context_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Build a comprehensive RAG prompt"""
        
        # Build context section with metadata if available
        context_sections = []
        for i, chunk in enumerate(context):
            section = f"## Source {i+1}"
            
            # Add metadata if available
            if context_metadata and i < len(context_metadata):
                metadata = context_metadata[i]
                if metadata.get('source'):
                    section += f" ({metadata['source']})"
                if metadata.get('timestamp'):
                    section += f" - {metadata['timestamp']}"
            
            section += f"\n{chunk}"
            context_sections.append(section)
        
        context_text = "\n\n".join(context_sections)
        
        return f"""{PromptTemplates.rag_system_prompt()}

# KNOWLEDGE BASE CONTEXT

{context_text}

# USER QUESTION
{query}

# YOUR RESPONSE
Please provide a helpful answer based on the context above. If you reference specific information, cite the relevant source(s)."""
    
    @staticmethod
    def channel_context_prompt(
        context: List[str], 
        query: str, 
        channel_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Prompt for Mattermost channel-specific queries"""
        
        channel_context = ""
        if channel_info:
            channel_context = f"""
# CHANNEL CONTEXT
Channel: {channel_info.get('display_name', 'Unknown')}
Team: {channel_info.get('team_name', 'Unknown')}
"""
        
        context_text = "\n\n".join([f"Message {i+1}: {chunk}" for i, chunk in enumerate(context)])
        
        return f"""{PromptTemplates.rag_system_prompt()}

You are helping answer questions about conversations and information from a Mattermost channel.
{channel_context}

# CHANNEL MESSAGES & CONTENT

{context_text}

# QUESTION
{query}

# RESPONSE
Please answer based on the channel content above. Reference specific messages when helpful."""
    
    @staticmethod
    def web_content_prompt(context: List[str], query: str, url: Optional[str] = None) -> str:
        """Prompt for web content queries"""
        
        url_info = f"\nSource URL: {url}" if url else ""
        context_text = "\n\n".join([f"Section {i+1}:\n{chunk}" for i, chunk in enumerate(context)])
        
        return f"""{PromptTemplates.rag_system_prompt()}

You are answering questions about web content that was ingested into the knowledge base.{url_info}

# WEB CONTENT

{context_text}

# QUESTION
{query}

# RESPONSE
Please answer based on the web content above."""
    
    @staticmethod
    def document_analysis_prompt(context: List[str], query: str) -> str:
        """Prompt for document analysis tasks"""
        
        context_text = "\n\n".join([f"Document Section {i+1}:\n{chunk}" for i, chunk in enumerate(context)])
        
        return f"""{PromptTemplates.rag_system_prompt()}

You are analyzing documents to answer questions. Focus on providing detailed, accurate information.

# DOCUMENT CONTENT

{context_text}

# ANALYSIS REQUEST
{query}

# ANALYSIS
Please provide a thorough analysis based on the document content above."""
    
    @staticmethod
    def multi_hop_reasoning_prompt(
        context: List[str], 
        query: str, 
        reasoning_steps: Optional[List[str]] = None
    ) -> str:
        """Prompt for multi-hop reasoning tasks"""
        
        context_text = "\n\n".join([f"Source {i+1}:\n{chunk}" for i, chunk in enumerate(context)])
        
        reasoning_context = ""
        if reasoning_steps:
            reasoning_context = f"""
# REASONING STEPS
{chr(10).join([f"{i+1}. {step}" for i, step in enumerate(reasoning_steps)])}
"""
        
        return f"""{PromptTemplates.rag_system_prompt()}

You need to perform multi-step reasoning to answer this question. Break down the problem and use the provided sources systematically.
{reasoning_context}

# AVAILABLE SOURCES

{context_text}

# COMPLEX QUESTION
{query}

# REASONING & ANSWER
Please work through this step-by-step, citing relevant sources for each step of your reasoning."""
    
    @staticmethod
    def error_response_prompt(error_type: str, context: Optional[str] = None) -> str:
        """Generate appropriate error response prompts"""
        
        error_templates = {
            "no_context": "I don't have any relevant information in my knowledge base to answer your question. Could you try rephrasing your question or provide more specific details?",
            
            "insufficient_context": "I found some related information, but it doesn't provide enough detail to fully answer your question. Here's what I can tell you based on the available context:",
            
            "api_error": "I encountered a technical issue while processing your request. Please try again in a moment.",
            
            "rate_limit": "I'm currently experiencing high usage. Please wait a moment and try your question again.",
            
            "parsing_error": "I had trouble understanding your question. Could you please rephrase it more clearly?"
        }
        
        base_response = error_templates.get(error_type, "I encountered an unexpected issue while processing your request.")
        
        if context and error_type == "insufficient_context":
            return f"{base_response}\n\n{context}"
        
        return base_response
    
    @staticmethod
    def confidence_calibration_prompt(response: str, context: List[str]) -> str:
        """Prompt for calibrating response confidence"""
        
        context_text = "\n".join([f"Source {i+1}: {chunk[:200]}..." for i, chunk in enumerate(context)])
        
        return f"""Evaluate the confidence level of this AI response:

RESPONSE: {response}

CONTEXT USED:
{context_text}

Rate the confidence (0-1) based on:
1. How well the context supports the response
2. Completeness of information
3. Clarity of sources
4. Potential for alternative interpretations

Provide just a number between 0 and 1."""