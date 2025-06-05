import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from ...config.settings import settings
from .base import TextChunk


class TextChunker:
    """Advanced text chunking with hierarchical and format-aware strategies"""
    
    def __init__(
        self,
        chunk_size: int = None,
        parent_chunk_size: int = None,
        chunk_overlap: int = None
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.parent_chunk_size = parent_chunk_size or settings.PARENT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ~= 4 characters)"""
        return len(text) // 4
    
    def chunk_by_tokens(self, text: str, max_tokens: int, overlap_tokens: int = 0) -> List[str]:
        """Split text into chunks by estimated token count"""
        
        # Convert tokens to characters (rough estimation)
        max_chars = max_tokens * 4
        overlap_chars = overlap_tokens * 4
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chars
            
            # Don't split in the middle of words
            if end < len(text):
                # Find the last space before the limit
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap_chars
            if start <= 0:
                start = end
        
        return chunks
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """Split text into sentence-based chunks"""
        
        # Simple sentence splitting (can be improved with spaCy/NLTK)
        sentence_endings = r'[.!?]+(?:\s|$)'
        sentences = re.split(sentence_endings, text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            # Check if adding this sentence exceeds chunk size
            if self.estimate_tokens(potential_chunk) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = potential_chunk
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_by_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraph-based chunks"""
        
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            # Check if adding this paragraph exceeds chunk size
            if self.estimate_tokens(potential_chunk) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk = potential_chunk
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def hierarchical_chunk(self, text: str, document_id: str) -> Tuple[List[TextChunk], List[TextChunk]]:
        """Create hierarchical chunks with improved parent-child relationships"""
        
        # First, create parent chunks with proper overlapping
        parent_chunk_texts = self.chunk_by_tokens(
            text, 
            self.parent_chunk_size, 
            self.chunk_overlap
        )
        parent_chunks = []
        current_position = 0
        
        for i, parent_text in enumerate(parent_chunk_texts):
            # Find actual position in text more accurately
            start_pos = text.find(parent_text, current_position)
            if start_pos == -1:
                start_pos = current_position
            
            parent_id = str(uuid.uuid4())
            parent_chunk = TextChunk(
                content=parent_text,
                start_index=start_pos,
                end_index=start_pos + len(parent_text),
                chunk_id=parent_id,
                metadata={
                    "chunk_type": "parent",
                    "document_id": document_id,
                    "parent_index": i,
                    "token_count": self.estimate_tokens(parent_text),
                    "char_count": len(parent_text),
                    "hierarchy_level": 0  # Parent level
                }
            )
            parent_chunks.append(parent_chunk)
            current_position = start_pos + len(parent_text) - self.chunk_overlap * 4  # Adjust for next search
        
        # Create smaller chunks within each parent with better tracking
        child_chunks = []
        global_child_index = 0
        
        for parent_idx, parent_chunk in enumerate(parent_chunks):
            small_chunk_texts = self.chunk_by_tokens(
                parent_chunk.content, 
                self.chunk_size, 
                self.chunk_overlap
            )
            
            parent_child_index = 0
            for small_text in small_chunk_texts:
                # Find position within parent chunk
                local_start = parent_chunk.content.find(small_text)
                if local_start == -1:
                    local_start = 0
                
                child_id = str(uuid.uuid4())
                child_chunk = TextChunk(
                    content=small_text,
                    start_index=parent_chunk.start_index + local_start,
                    end_index=parent_chunk.start_index + local_start + len(small_text),
                    chunk_id=child_id,
                    parent_id=parent_chunk.chunk_id,
                    metadata={
                        "chunk_type": "child",
                        "document_id": document_id,
                        "parent_id": parent_chunk.chunk_id,
                        "parent_index": parent_idx,
                        "child_index_global": global_child_index,
                        "child_index_in_parent": parent_child_index,
                        "token_count": self.estimate_tokens(small_text),
                        "char_count": len(small_text),
                        "hierarchy_level": 1,  # Child level
                        "context_window": {
                            "parent_content_preview": parent_chunk.content[:200] + "..." if len(parent_chunk.content) > 200 else parent_chunk.content,
                            "position_in_parent": f"{parent_child_index + 1}/{len(small_chunk_texts)}"
                        }
                    }
                )
                child_chunks.append(child_chunk)
                global_child_index += 1
                parent_child_index += 1
        
        logger.info(f"Created hierarchical chunks: {len(parent_chunks)} parents, {len(child_chunks)} children")
        return child_chunks, parent_chunks
    
    def chunk_markdown(self, text: str, document_id: str) -> List[TextChunk]:
        """Markdown-aware chunking that respects headers and structure"""
        
        chunks = []
        lines = text.split('\n')
        current_chunk = ""
        current_header = ""
        chunk_index = 0
        
        for line in lines:
            # Check if this is a header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if header_match:
                # Save current chunk if it exists
                if current_chunk.strip():
                    chunk_id = f"{document_id}_md_{chunk_index}"
                    chunk = TextChunk(
                        content=current_chunk.strip(),
                        start_index=0,  # Would need more complex tracking for exact positions
                        end_index=len(current_chunk.strip()),
                        chunk_id=chunk_id,
                        metadata={
                            "chunk_type": "markdown_section",
                            "document_id": document_id,
                            "header": current_header,
                            "header_level": len(header_match.group(1)) if header_match else 0
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with header
                current_header = header_match.group(2)
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
                
                # Check if chunk is getting too large
                if self.estimate_tokens(current_chunk) > self.chunk_size:
                    chunk_id = f"{document_id}_md_{chunk_index}"
                    chunk = TextChunk(
                        content=current_chunk.strip(),
                        start_index=0,
                        end_index=len(current_chunk.strip()),
                        chunk_id=chunk_id,
                        metadata={
                            "chunk_type": "markdown_section",
                            "document_id": document_id,
                            "header": current_header,
                            "partial": True
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk = ""
        
        # Add the last chunk
        if current_chunk.strip():
            chunk_id = f"{document_id}_md_{chunk_index}"
            chunk = TextChunk(
                content=current_chunk.strip(),
                start_index=0,
                end_index=len(current_chunk.strip()),
                chunk_id=chunk_id,
                metadata={
                    "chunk_type": "markdown_section",
                    "document_id": document_id,
                    "header": current_header
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_code(self, text: str, document_id: str, language: str = "unknown") -> List[TextChunk]:
        """Code-aware chunking that respects function and class boundaries"""
        
        chunks = []
        lines = text.split('\n')
        current_chunk = ""
        current_function = ""
        chunk_index = 0
        
        # Simple function detection patterns
        function_patterns = {
            "python": r'^\s*def\s+(\w+)',
            "javascript": r'^\s*function\s+(\w+)',
            "java": r'^\s*(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(',
            "cpp": r'^\s*(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*\{?$'
        }
        
        pattern = function_patterns.get(language.lower(), r'^\s*(?:def|function|class)\s+(\w+)')
        
        for line in lines:
            function_match = re.match(pattern, line)
            
            if function_match:
                # Save current chunk if it exists
                if current_chunk.strip():
                    chunk_id = f"{document_id}_code_{chunk_index}"
                    chunk = TextChunk(
                        content=current_chunk.strip(),
                        start_index=0,
                        end_index=len(current_chunk.strip()),
                        chunk_id=chunk_id,
                        metadata={
                            "chunk_type": "code_block",
                            "document_id": document_id,
                            "function": current_function,
                            "language": language
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with function
                current_function = function_match.group(1)
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
                
                # Check if chunk is getting too large
                if self.estimate_tokens(current_chunk) > self.chunk_size:
                    chunk_id = f"{document_id}_code_{chunk_index}"
                    chunk = TextChunk(
                        content=current_chunk.strip(),
                        start_index=0,
                        end_index=len(current_chunk.strip()),
                        chunk_id=chunk_id,
                        metadata={
                            "chunk_type": "code_block",
                            "document_id": document_id,
                            "function": current_function,
                            "language": language,
                            "partial": True
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk = ""
        
        # Add the last chunk
        if current_chunk.strip():
            chunk_id = f"{document_id}_code_{chunk_index}"
            chunk = TextChunk(
                content=current_chunk.strip(),
                start_index=0,
                end_index=len(current_chunk.strip()),
                chunk_id=chunk_id,
                metadata={
                    "chunk_type": "code_block",
                    "document_id": document_id,
                    "function": current_function,
                    "language": language
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def get_expanded_context(self, child_chunk: TextChunk, parent_chunks: List[TextChunk]) -> str:
        """Get expanded context by finding and assembling the parent chunk content"""
        
        if not child_chunk.parent_id:
            return child_chunk.content
        
        # Find the parent chunk
        parent_chunk = None
        for chunk in parent_chunks:
            if chunk.chunk_id == child_chunk.parent_id:
                parent_chunk = chunk
                break
        
        if not parent_chunk:
            return child_chunk.content
        
        # Return parent content with child highlighted
        parent_content = parent_chunk.content
        child_content = child_chunk.content
        
        # Try to find child content in parent and add context markers
        if child_content in parent_content:
            highlighted = parent_content.replace(
                child_content, 
                f"**[FOCUS: {child_content}]**"
            )
            return highlighted
        else:
            # Fallback: return parent + child
            return f"{parent_content}\n\n--- FOCUS SECTION ---\n{child_content}"
    
    def chunk_csv_data(self, text: str, document_id: str, headers: List[str] = None) -> List[TextChunk]:
        """CSV-aware chunking that preserves row context"""
        
        lines = text.strip().split('\n')
        if not lines:
            return []
        
        # Extract headers if not provided
        if headers is None:
            headers = lines[0].split(',')
            data_lines = lines[1:]
        else:
            data_lines = lines
        
        chunks = []
        current_rows = []
        chunk_index = 0
        
        for line in data_lines:
            current_rows.append(line)
            
            # Create chunk with header context
            chunk_content = ','.join(headers) + '\n' + '\n'.join(current_rows)
            
            # Check if chunk is getting too large
            if self.estimate_tokens(chunk_content) > self.chunk_size:
                if len(current_rows) > 1:
                    # Save chunk with all but the last row
                    final_content = ','.join(headers) + '\n' + '\n'.join(current_rows[:-1])
                    chunk_id = f"{document_id}_csv_{chunk_index}"
                    chunk = TextChunk(
                        content=final_content,
                        start_index=0,
                        end_index=len(final_content),
                        chunk_id=chunk_id,
                        metadata={
                            "chunk_type": "csv_data",
                            "document_id": document_id,
                            "headers": headers,
                            "row_count": len(current_rows) - 1
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Start new chunk with the last row
                    current_rows = [current_rows[-1]]
                else:
                    # Single row is too large, keep it anyway
                    chunk_id = f"{document_id}_csv_{chunk_index}"
                    chunk = TextChunk(
                        content=chunk_content,
                        start_index=0,
                        end_index=len(chunk_content),
                        chunk_id=chunk_id,
                        metadata={
                            "chunk_type": "csv_data",
                            "document_id": document_id,
                            "headers": headers,
                            "row_count": 1,
                            "oversized": True
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_rows = []
        
        # Add the last chunk
        if current_rows:
            final_content = ','.join(headers) + '\n' + '\n'.join(current_rows)
            chunk_id = f"{document_id}_csv_{chunk_index}"
            chunk = TextChunk(
                content=final_content,
                start_index=0,
                end_index=len(final_content),
                chunk_id=chunk_id,
                metadata={
                    "chunk_type": "csv_data",
                    "document_id": document_id,
                    "headers": headers,
                    "row_count": len(current_rows)
                }
            )
            chunks.append(chunk)
        
        return chunks