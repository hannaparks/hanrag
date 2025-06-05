"""
Environment File Manager

This module provides functionality to read and write .env files,
allowing for dynamic configuration updates.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import re


class EnvFileManager:
    """Manages reading and writing of .env files"""
    
    def __init__(self, env_file_path: Optional[Union[str, Path]] = None):
        """
        Initialize the environment file manager.
        
        Args:
            env_file_path: Path to the .env file. If not provided, 
                          will look for .env in the current directory.
        """
        if env_file_path:
            self.env_file_path = Path(env_file_path)
        else:
            # Look for .env file in project root
            self.env_file_path = Path(__file__).parent.parent.parent / ".env"
            if not self.env_file_path.exists():
                # Try .env.local
                self.env_file_path = Path(__file__).parent.parent.parent / ".env.local"
    
    def read_env_file(self) -> Dict[str, str]:
        """
        Read the current .env file and return as a dictionary.
        
        Returns:
            Dictionary of environment variables
        """
        env_vars = {}
        
        if not self.env_file_path.exists():
            return env_vars
        
        with open(self.env_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse KEY=VALUE format
                match = re.match(r'^([A-Z_]+[A-Z0-9_]*)=(.*)$', line)
                if match:
                    key, value = match.groups()
                    # Remove quotes if present
                    value = value.strip('"').strip("'")
                    env_vars[key] = value
        
        return env_vars
    
    def write_env_file(self, env_vars: Dict[str, str], preserve_comments: bool = True) -> None:
        """
        Write environment variables to the .env file.
        
        Args:
            env_vars: Dictionary of environment variables to write
            preserve_comments: Whether to preserve existing comments
        """
        lines = []
        existing_keys = set()
        
        if preserve_comments and self.env_file_path.exists():
            # Read existing file to preserve comments and structure
            with open(self.env_file_path, 'r') as f:
                for line in f:
                    stripped = line.strip()
                    
                    # Preserve empty lines and comments
                    if not stripped or stripped.startswith('#'):
                        lines.append(line.rstrip())
                        continue
                    
                    # Check if this is a variable line
                    match = re.match(r'^([A-Z_]+[A-Z0-9_]*)=', stripped)
                    if match:
                        key = match.group(1)
                        existing_keys.add(key)
                        
                        # Update value if key is in new env_vars
                        if key in env_vars:
                            # Preserve quotes if value contains spaces or special characters
                            value = env_vars[key]
                            if ' ' in value or '=' in value or '#' in value:
                                value = f'"{value}"'
                            lines.append(f"{key}={value}")
                        else:
                            # Keep existing line
                            lines.append(line.rstrip())
                    else:
                        # Keep any other lines
                        lines.append(line.rstrip())
        
        # Add new variables that weren't in the original file
        new_keys = set(env_vars.keys()) - existing_keys
        if new_keys:
            if lines and lines[-1]:  # Add blank line if needed
                lines.append('')
            
            lines.append('# Updated configuration')
            for key in sorted(new_keys):
                value = env_vars[key]
                if ' ' in value or '=' in value or '#' in value:
                    value = f'"{value}"'
                lines.append(f"{key}={value}")
        
        # Write the file
        with open(self.env_file_path, 'w') as f:
            f.write('\n'.join(lines))
            if lines and lines[-1]:  # Ensure file ends with newline
                f.write('\n')
    
    def update_env_vars(self, updates: Dict[str, str]) -> None:
        """
        Update specific environment variables while preserving others.
        
        Args:
            updates: Dictionary of variables to update
        """
        current_vars = self.read_env_file()
        current_vars.update(updates)
        self.write_env_file(current_vars)
    
    def get_env_var(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a specific environment variable value.
        
        Args:
            key: The environment variable key
            default: Default value if key not found
            
        Returns:
            The value or default
        """
        env_vars = self.read_env_file()
        return env_vars.get(key, default)