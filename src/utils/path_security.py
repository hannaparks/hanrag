"""
Path security utilities for preventing path traversal attacks.
"""
from pathlib import Path
from typing import Union, Optional
from fastapi import HTTPException


class PathSecurityError(Exception):
    """Raised when a path security violation is detected."""


def validate_file_path(
    file_path: Union[str, Path], 
    allowed_base_dir: Union[str, Path],
    allow_symlinks: bool = False
) -> Path:
    """
    Validate that a file path is safe and within the allowed base directory.
    
    Args:
        file_path: The file path to validate
        allowed_base_dir: The base directory that files must be within
        allow_symlinks: Whether to allow symbolic links (default: False)
        
    Returns:
        Path: The resolved, validated path
        
    Raises:
        PathSecurityError: If the path is unsafe or outside allowed directory
    """
    try:
        # Convert to Path objects and resolve to absolute paths
        file_path = Path(file_path).resolve()
        allowed_base_dir = Path(allowed_base_dir).resolve()
        
        # Check if the resolved path is within the allowed base directory
        if not _is_path_within_directory(file_path, allowed_base_dir):
            raise PathSecurityError(
                f"Path traversal detected: {file_path} is outside allowed directory {allowed_base_dir}"
            )
        
        # Check for symbolic links if not allowed
        if not allow_symlinks and file_path.is_symlink():
            raise PathSecurityError(f"Symbolic links not allowed: {file_path}")
        
        return file_path
        
    except (OSError, ValueError) as e:
        raise PathSecurityError(f"Invalid path: {e}")


def validate_filename(filename: str, allowed_extensions: Optional[set[str]] = None) -> str:
    """
    Validate a filename for security issues.
    
    Args:
        filename: The filename to validate
        allowed_extensions: Set of allowed file extensions (with dots, e.g., {'.txt', '.md'})
        
    Returns:
        str: The validated filename
        
    Raises:
        PathSecurityError: If the filename is unsafe
    """
    if not filename or not isinstance(filename, str):
        raise PathSecurityError("Filename must be a non-empty string")
    
    # Check for path traversal attempts
    if ".." in filename or "/" in filename or "\\" in filename:
        raise PathSecurityError(f"Invalid characters in filename: {filename}")
    
    # Check for null bytes
    if "\x00" in filename:
        raise PathSecurityError("Null bytes not allowed in filename")
    
    # Check for control characters
    if any(ord(c) < 32 for c in filename if c != '\t'):
        raise PathSecurityError("Control characters not allowed in filename")
    
    # Check file extension if allowed_extensions is specified
    if allowed_extensions is not None:
        file_ext = Path(filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise PathSecurityError(
                f"File extension '{file_ext}' not allowed. Allowed: {allowed_extensions}"
            )
    
    return filename


def secure_join(base_path: Union[str, Path], *paths: Union[str, Path]) -> Path:
    """
    Securely join paths and validate the result is within the base path.
    
    Args:
        base_path: The base directory path
        *paths: Path components to join
        
    Returns:
        Path: The securely joined path
        
    Raises:
        PathSecurityError: If the resulting path would be outside base_path
    """
    base_path = Path(base_path).resolve()
    
    # Join all path components
    result_path = base_path
    for path_component in paths:
        result_path = result_path / path_component
    
    # Resolve and validate
    result_path = result_path.resolve()
    
    if not _is_path_within_directory(result_path, base_path):
        raise PathSecurityError(
            f"Path traversal detected: resulting path {result_path} is outside base {base_path}"
        )
    
    return result_path


def _is_path_within_directory(path: Path, directory: Path) -> bool:
    """
    Check if a path is within a directory (handles edge cases properly).
    
    Args:
        path: The path to check
        directory: The directory that should contain the path
        
    Returns:
        bool: True if path is within directory
    """
    try:
        # Use relative_to to check if path is within directory
        path.relative_to(directory)
        return True
    except ValueError:
        return False


# Constants for common secure directories
def get_static_files_dir() -> Path:
    """Get the secure static files directory."""
    return Path(__file__).parent.parent / "api" / "static"


def get_temp_files_dir() -> Path:
    """Get the secure temporary files directory."""
    temp_dir = Path(__file__).parent.parent.parent / "temp"
    temp_dir.mkdir(exist_ok=True)
    return temp_dir


def get_data_dir() -> Path:
    """Get the secure data directory."""
    data_dir = Path(__file__).parent.parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


# Allowed file extensions for different operations
ALLOWED_STATIC_EXTENSIONS = {'.html', '.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico'}
ALLOWED_CONFIG_EXTENSIONS = {'.html', '.json'}
ALLOWED_DOCUMENT_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx', '.doc', '.rtf'}
ALLOWED_DATA_EXTENSIONS = {'.json', '.csv', '.pkl', '.db'}


def secure_file_response_path(file_path: str, base_dir: Path, allowed_extensions: set[str]) -> Path:
    """
    Securely validate a file path for HTTP file responses.
    
    Args:
        file_path: The requested file path
        base_dir: The base directory for file serving
        allowed_extensions: Set of allowed file extensions
        
    Returns:
        Path: The validated file path
        
    Raises:
        HTTPException: If the path is invalid or unsafe
    """
    try:
        # Validate filename
        filename = Path(file_path).name
        validate_filename(filename, allowed_extensions)
        
        # Construct the full path by joining base_dir with file_path
        # This ensures we're looking in the correct directory
        full_path = secure_join(base_dir, file_path)
        
        # Check if file exists
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check if it's actually a file
        if not full_path.is_file():
            raise HTTPException(status_code=404, detail="Path is not a file")
        
        return full_path
        
    except PathSecurityError as e:
        raise HTTPException(status_code=400, detail=f"Invalid file path: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="File access error")