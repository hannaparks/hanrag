"""Base classes for chunking"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    content: str
    start_index: int
    end_index: int
    chunk_id: str
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}