"""API Authentication Module"""
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery
from typing import Optional
import hashlib
from datetime import datetime
import secrets

from src.config.settings import settings


# API Key header/query parameter names
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)


class APIKeyManager:
    """Manages API keys for authentication"""
    
    def __init__(self):
        # In production, these should come from a database or secure vault
        # For now, we'll use environment variables
        self._api_keys = {}
        self._load_api_keys()
    
    def _load_api_keys(self):
        """Load API keys from settings/environment"""
        # Get API key from settings or generate a default one
        api_key = getattr(settings, 'API_KEY', None)
        if not api_key:
            # Generate a secure default key for development
            api_key = secrets.token_urlsafe(32)
            print(f"Generated development API key: {api_key}")
        
        # Store hashed version of the key
        self._api_keys[self._hash_key(api_key)] = {
            "name": "default",
            "created_at": datetime.utcnow(),
            "permissions": ["read", "write", "admin"]
        }
    
    def _hash_key(self, api_key: str) -> str:
        """Hash API key for secure storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def validate_key(self, api_key: str) -> bool:
        """Validate an API key"""
        if not api_key:
            return False
        
        hashed_key = self._hash_key(api_key)
        return hashed_key in self._api_keys
    
    def get_key_info(self, api_key: str) -> Optional[dict]:
        """Get information about an API key"""
        if not api_key:
            return None
        
        hashed_key = self._hash_key(api_key)
        return self._api_keys.get(hashed_key)


# Global API key manager instance
api_key_manager = APIKeyManager()


async def get_api_key(
    api_key_header: str = Security(api_key_header),
    api_key_query: str = Security(api_key_query),
) -> str:
    """Validate API key from header or query parameter"""
    # Check header first, then query parameter
    api_key = api_key_header or api_key_query
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key missing",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if not api_key_manager.validate_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return api_key


async def require_admin(api_key: str = Security(get_api_key)) -> str:
    """Require admin permissions for endpoint"""
    key_info = api_key_manager.get_key_info(api_key)
    
    if not key_info or "admin" not in key_info.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permissions required"
        )
    
    return api_key


async def require_write(api_key: str = Security(get_api_key)) -> str:
    """Require write permissions for endpoint"""
    key_info = api_key_manager.get_key_info(api_key)
    
    if not key_info or "write" not in key_info.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Write permissions required"
        )
    
    return api_key


# Optional authentication - for endpoints that can work with or without auth
async def optional_api_key(
    api_key_header: str = Security(api_key_header),
    api_key_query: str = Security(api_key_query),
) -> Optional[str]:
    """Optional API key validation - doesn't throw error if missing"""
    api_key = api_key_header or api_key_query
    
    if api_key and api_key_manager.validate_key(api_key):
        return api_key
    
    return None