try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False
    
import mimetypes
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from urllib.parse import urlparse
import re
from dataclasses import dataclass
from loguru import logger


@dataclass
class FormatDetectionResult:
    """Result of format detection"""
    format_type: str
    confidence: float
    mime_type: Optional[str] = None
    encoding: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class FormatDetector:
    """Advanced format detection and routing system"""
    
    def __init__(self):
        # Check for optional dependencies
        if not HAS_MAGIC:
            logger.warning(
                "python-magic library not available. MIME type detection will use fallback methods. "
                "Install with: pip install python-magic"
            )
        
        # File extension mappings
        self.extension_map = {
            # Documents
            '.pdf': 'pdf',
            '.docx': 'docx',
            '.doc': 'doc',
            '.odt': 'odt',
            '.rtf': 'rtf',
            
            # Text formats
            '.txt': 'text',
            '.md': 'markdown',
            '.rst': 'restructuredtext',
            '.tex': 'latex',
            
            # Data formats
            '.csv': 'csv',
            '.tsv': 'tsv',
            '.xlsx': 'excel',
            '.xls': 'excel_legacy',
            '.ods': 'ods',
            
            # Structured data
            '.json': 'json',
            '.jsonl': 'jsonlines',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.ini': 'ini',
            
            # Web formats
            '.html': 'html',
            '.htm': 'html',
            '.xhtml': 'xhtml',
            '.css': 'css',
            
            # Code formats
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'jsx',
            '.tsx': 'tsx',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c_header',
            '.hpp': 'cpp_header',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'golang',
            '.rs': 'rust',
            '.sql': 'sql',
            '.sh': 'shell',
            '.bash': 'bash',
            '.ps1': 'powershell',
            
            # Configuration
            '.conf': 'config',
            '.cfg': 'config',
            '.properties': 'properties',
            '.env': 'environment',
            
            # Archives (for reference)
            '.zip': 'archive_zip',
            '.tar': 'archive_tar',
            '.gz': 'archive_gzip',
            '.7z': 'archive_7z',
            
            # Images (for OCR potential)
            '.png': 'image_png',
            '.jpg': 'image_jpeg',
            '.jpeg': 'image_jpeg',
            '.gif': 'image_gif',
            '.bmp': 'image_bmp',
            '.tiff': 'image_tiff',
            '.svg': 'image_svg',
        }
        
        # MIME type mappings
        self.mime_map = {
            'application/pdf': 'pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
            'application/msword': 'doc',
            'text/plain': 'text',
            'text/markdown': 'markdown',
            'text/csv': 'csv',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'excel',
            'application/vnd.ms-excel': 'excel_legacy',
            'application/json': 'json',
            'application/xml': 'xml',
            'text/xml': 'xml',
            'text/html': 'html',
            'application/xhtml+xml': 'xhtml',
            'text/css': 'css',
            'application/javascript': 'javascript',
            'text/javascript': 'javascript',
            'application/x-python': 'python',
            'text/x-python': 'python',
        }
        
        # Content pattern detection
        self.content_patterns = {
            'json': [
                re.compile(r'^\s*[\{\[].*[\}\]]\s*$', re.DOTALL),
                re.compile(r'^\s*\{.*".*":.*\}.*$', re.DOTALL)
            ],
            'xml': [
                re.compile(r'^\s*<\?xml.*\?>', re.MULTILINE),
                re.compile(r'^\s*<[^>]+>.*</[^>]+>\s*$', re.DOTALL)
            ],
            'html': [
                re.compile(r'<!DOCTYPE\s+html', re.IGNORECASE),
                re.compile(r'<html.*?>.*</html>', re.DOTALL | re.IGNORECASE),
                re.compile(r'<head.*?>.*</head>', re.DOTALL | re.IGNORECASE)
            ],
            'csv': [
                re.compile(r'^[^,\n]+,[^,\n]+', re.MULTILINE),
                re.compile(r'("[^"]*",\s*){2,}')
            ],
            'markdown': [
                re.compile(r'^#{1,6}\s+.+$', re.MULTILINE),
                re.compile(r'^\*{1,2}.+\*{1,2}$', re.MULTILINE),
                re.compile(r'^\[.+\]\(.+\)$', re.MULTILINE)
            ],
            'python': [
                re.compile(r'^(import|from)\s+\w+', re.MULTILINE),
                re.compile(r'^def\s+\w+\(.*\):', re.MULTILINE),
                re.compile(r'^class\s+\w+.*:', re.MULTILINE)
            ],
            'javascript': [
                re.compile(r'^(var|let|const)\s+\w+', re.MULTILINE),
                re.compile(r'^function\s+\w+\(.*\)', re.MULTILINE),
                re.compile(r'=>', re.MULTILINE)
            ],
            'sql': [
                re.compile(r'^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\s+', re.IGNORECASE | re.MULTILINE),
                re.compile(r'\b(FROM|WHERE|JOIN|GROUP BY|ORDER BY)\b', re.IGNORECASE)
            ]
        }
        
        # Processing strategy mappings
        self.processing_strategies = {
            # Document formats
            'pdf': 'document_parser',
            'docx': 'document_parser',
            'doc': 'document_parser',
            'odt': 'document_parser',
            
            # Text formats
            'text': 'text_processor',
            'markdown': 'text_processor',
            'restructuredtext': 'text_processor',
            
            # Data formats
            'csv': 'structured_data_processor',
            'tsv': 'structured_data_processor',
            'excel': 'structured_data_processor',
            'excel_legacy': 'structured_data_processor',
            
            # Structured formats
            'json': 'structured_data_processor',
            'jsonlines': 'structured_data_processor',
            'xml': 'structured_data_processor',
            'yaml': 'structured_data_processor',
            
            # Web formats
            'html': 'web_content_processor',
            'xhtml': 'web_content_processor',
            'css': 'code_processor',
            
            # Code formats
            'python': 'code_processor',
            'javascript': 'code_processor',
            'typescript': 'code_processor',
            'java': 'code_processor',
            'cpp': 'code_processor',
            'c': 'code_processor',
            'sql': 'code_processor',
            
            # Default fallback
            'unknown': 'text_processor'
        }
    
    def detect_file_format(self, file_path: Union[str, Path]) -> FormatDetectionResult:
        """Detect format of a local file"""
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            return FormatDetectionResult(
                format_type='unknown',
                confidence=0.0,
                metadata={'error': 'File not found'}
            )
        
        # Start with extension-based detection
        extension_result = self._detect_by_extension(file_path)
        
        # Enhance with MIME type detection
        mime_result = self._detect_by_mime_type(file_path)
        
        # Content-based detection for ambiguous cases
        content_result = None
        if extension_result.confidence < 0.8 or extension_result.format_type == 'unknown':
            content_result = self._detect_by_content(file_path)
        
        # Combine results
        final_result = self._combine_detection_results(
            extension_result, mime_result, content_result
        )
        
        # Add file metadata
        final_result.metadata.update({
            'file_name': file_path.name,
            'file_size': file_path.stat().st_size,
            'file_extension': file_path.suffix.lower(),
            'processing_strategy': self.processing_strategies.get(
                final_result.format_type, 'text_processor'
            )
        })
        
        return final_result
    
    def detect_url_format(self, url: str, content_type: Optional[str] = None) -> FormatDetectionResult:
        """Detect format of URL content"""
        
        parsed_url = urlparse(url)
        path = Path(parsed_url.path)
        
        # Start with URL path extension
        extension_result = self._detect_by_extension(path) if path.suffix else FormatDetectionResult('unknown', 0.0)
        
        # Use provided content type
        mime_result = FormatDetectionResult('unknown', 0.0)
        if content_type:
            format_type = self.mime_map.get(content_type, 'unknown')
            mime_result = FormatDetectionResult(
                format_type=format_type,
                confidence=0.9 if format_type != 'unknown' else 0.1,
                mime_type=content_type
            )
        
        # Combine results
        final_result = self._combine_detection_results(extension_result, mime_result)
        
        # Add URL metadata
        final_result.metadata.update({
            'url': url,
            'domain': parsed_url.netloc,
            'path': parsed_url.path,
            'content_type': content_type,
            'processing_strategy': self.processing_strategies.get(
                final_result.format_type, 'web_content_processor'
            )
        })
        
        return final_result
    
    def detect_content_format(self, content: str, hint: Optional[str] = None) -> FormatDetectionResult:
        """Detect format of raw content"""
        
        # Start with hint if provided
        hint_result = FormatDetectionResult('unknown', 0.0)
        if hint and hint in self.extension_map.values():
            hint_result = FormatDetectionResult(format_type=hint, confidence=0.7)
        
        # Content pattern detection
        content_result = self._detect_by_content_string(content)
        
        # Combine results
        final_result = self._combine_detection_results(hint_result, content_result)
        
        # Add content metadata
        final_result.metadata.update({
            'content_length': len(content),
            'line_count': len(content.split('\n')),
            'processing_strategy': self.processing_strategies.get(
                final_result.format_type, 'text_processor'
            )
        })
        
        return final_result
    
    def _detect_by_extension(self, file_path: Path) -> FormatDetectionResult:
        """Detect format by file extension"""
        
        extension = file_path.suffix.lower()
        
        if extension in self.extension_map:
            return FormatDetectionResult(
                format_type=self.extension_map[extension],
                confidence=0.8,
                metadata={'detection_method': 'extension'}
            )
        
        return FormatDetectionResult(
            format_type='unknown',
            confidence=0.0,
            metadata={'detection_method': 'extension', 'unknown_extension': extension}
        )
    
    def _detect_by_mime_type(self, file_path: Path) -> FormatDetectionResult:
        """Detect format by MIME type"""
        
        try:
            # Try python-magic first (more accurate) if available
            mime_type = None
            if HAS_MAGIC:
                try:
                    mime_type = magic.from_file(str(file_path), mime=True)
                except Exception as e:
                    logger.debug(f"Magic library failed for {file_path}: {e}")
            
            # Fallback to mimetypes module
            if not mime_type:
                mime_type, _ = mimetypes.guess_type(str(file_path))
            
            if mime_type and mime_type in self.mime_map:
                return FormatDetectionResult(
                    format_type=self.mime_map[mime_type],
                    confidence=0.9,
                    mime_type=mime_type,
                    metadata={'detection_method': 'mime_type'}
                )
            
            return FormatDetectionResult(
                format_type='unknown',
                confidence=0.0,
                mime_type=mime_type,
                metadata={'detection_method': 'mime_type', 'unknown_mime_type': mime_type}
            )
            
        except Exception as e:
            logger.warning(f"MIME type detection failed for {file_path}: {e}")
            return FormatDetectionResult(
                format_type='unknown',
                confidence=0.0,
                metadata={'detection_method': 'mime_type', 'error': str(e)}
            )
    
    def _detect_by_content(self, file_path: Path) -> FormatDetectionResult:
        """Detect format by content analysis"""
        
        try:
            # Read file content (first few KB for efficiency)
            max_bytes = 8192
            with open(file_path, 'rb') as f:
                raw_content = f.read(max_bytes)
            
            # Try to decode as text
            content = None
            for encoding in ['utf-8', 'utf-16', 'latin-1']:
                try:
                    content = raw_content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                return FormatDetectionResult(
                    format_type='binary',
                    confidence=0.8,
                    metadata={'detection_method': 'content', 'reason': 'non_text_content'}
                )
            
            return self._detect_by_content_string(content)
            
        except Exception as e:
            logger.warning(f"Content detection failed for {file_path}: {e}")
            return FormatDetectionResult(
                format_type='unknown',
                confidence=0.0,
                metadata={'detection_method': 'content', 'error': str(e)}
            )
    
    def _detect_by_content_string(self, content: str) -> FormatDetectionResult:
        """Detect format by analyzing content string"""
        
        # Limit content for pattern matching (first 2KB)
        sample_content = content[:2048]
        
        # Test each pattern
        best_match = None
        best_confidence = 0.0
        
        for format_type, patterns in self.content_patterns.items():
            confidence = 0.0
            matches = 0
            
            for pattern in patterns:
                if pattern.search(sample_content):
                    matches += 1
            
            # Calculate confidence based on pattern matches
            if matches > 0:
                confidence = min(0.9, 0.3 + (matches * 0.3))
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = format_type
        
        if best_match:
            return FormatDetectionResult(
                format_type=best_match,
                confidence=best_confidence,
                metadata={
                    'detection_method': 'content_pattern',
                    'pattern_matches': best_match
                }
            )
        
        # Default to text if no patterns match but content is readable
        return FormatDetectionResult(
            format_type='text',
            confidence=0.5,
            metadata={'detection_method': 'content_pattern', 'reason': 'default_text'}
        )
    
    def _combine_detection_results(
        self,
        *results: FormatDetectionResult
    ) -> FormatDetectionResult:
        """Combine multiple detection results"""
        
        valid_results = [r for r in results if r is not None and r.confidence > 0.0]
        
        if not valid_results:
            return FormatDetectionResult(format_type='unknown', confidence=0.0)
        
        # Find result with highest confidence
        best_result = max(valid_results, key=lambda r: r.confidence)
        
        # If results agree, boost confidence
        format_types = [r.format_type for r in valid_results]
        if len(set(format_types)) == 1:
            best_result.confidence = min(0.95, best_result.confidence + 0.1)
        
        # Combine metadata
        combined_metadata = {}
        for result in valid_results:
            combined_metadata.update(result.metadata or {})
        
        best_result.metadata = combined_metadata
        
        return best_result
    
    def get_processor_for_format(self, format_type: str) -> str:
        """Get the appropriate processor for a format"""
        return self.processing_strategies.get(format_type, 'text_processor')
    
    def is_supported_format(self, format_type: str) -> bool:
        """Check if format is supported for processing"""
        return format_type in self.processing_strategies
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get all supported formats grouped by processor"""
        
        grouped = {}
        for format_type, processor in self.processing_strategies.items():
            if processor not in grouped:
                grouped[processor] = []
            grouped[processor].append(format_type)
        
        return grouped
    
    def get_format_info(self, format_type: str) -> Dict[str, Any]:
        """Get detailed information about a format"""
        
        # Find extensions for this format
        extensions = [ext for ext, fmt in self.extension_map.items() if fmt == format_type]
        
        # Find MIME types for this format
        mime_types = [mime for mime, fmt in self.mime_map.items() if fmt == format_type]
        
        return {
            'format_type': format_type,
            'extensions': extensions,
            'mime_types': mime_types,
            'processor': self.processing_strategies.get(format_type, 'unknown'),
            'supported': self.is_supported_format(format_type)
        }