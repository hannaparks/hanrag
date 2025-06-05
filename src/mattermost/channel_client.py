from typing import List, Dict, Any, Optional
import aiohttp
from loguru import logger
from ..config.settings import settings


class MattermostClient:
    """Mattermost API client for channel operations"""
    
    def __init__(self):
        self.base_url = settings.MATTERMOST_URL.rstrip('/')
<<<<<<< HEAD
=======
        logger.info(f"Mattermost URL: {self.base_url}")
        
>>>>>>> 66c74c8
        self.token = settings.MATTERMOST_PERSONAL_TOKEN
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        self.timeout = aiohttp.ClientTimeout(total=30)
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Make HTTP request to Mattermost API"""
        
        url = f"{self.base_url}/api/v4{endpoint}"
        
        try:
            async with aiohttp.ClientSession(
                headers=self.headers,
                timeout=self.timeout
            ) as session:
                async with session.request(method, url, **kwargs) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Mattermost API error: {response.status} - {await response.text()}")
                        return None
                        
        except Exception as e:
            logger.error(f"Mattermost API request failed: {e}")
            return None
    
    async def get_channel_history(
        self,
        channel_id: str,
        per_page: int = 200,
        max_messages: int = 1000,
        before: Optional[str] = None,
        after: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract message history from Mattermost channel"""
        
        messages = []
        page = 0
        
        try:
            while len(messages) < max_messages:
                params = {
                    "per_page": per_page,
                    "page": page
                }
                
                if before:
                    params["before"] = before
                if after:
                    params["after"] = after
                
                response = await self._make_request(
                    "GET",
                    f"/channels/{channel_id}/posts",
                    params=params
                )
                
                if not response:
                    break
                
                posts = response.get("posts", {})
                
                if not posts:
                    break
                
                # Sort posts by creation time
                sorted_posts = sorted(
                    posts.values(),
                    key=lambda x: x.get("create_at", 0)
                )
                
                for post in sorted_posts:
                    if post.get("message") and post.get("message").strip():
                        message_data = {
                            "id": post["id"],
                            "message": post["message"],
                            "user_id": post["user_id"],
                            "create_at": post["create_at"],
                            "update_at": post.get("update_at", post["create_at"]),
                            "channel_id": channel_id,
                            "type": post.get("type", ""),
                            "props": post.get("props", {}),
                            "hashtags": post.get("hashtags", ""),
                            "pending_post_id": post.get("pending_post_id", ""),
                            "reply_count": post.get("reply_count", 0),
                            "root_id": post.get("root_id", ""),
                            "parent_id": post.get("parent_id", "")
                        }
                        messages.append(message_data)
                
                page += 1
                
                # Stop if we got fewer posts than requested (end of channel)
                if len(posts) < per_page:
                    break
            
            logger.info(f"Retrieved {len(messages)} messages from channel {channel_id}")
            return messages[:max_messages]
            
        except Exception as e:
            logger.error(f"Failed to get channel history: {e}")
            return []
    
    async def get_channel_info(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Get channel information"""
        
        try:
            response = await self._make_request("GET", f"/channels/{channel_id}")
            
            if response:
                logger.debug(f"Retrieved info for channel: {response.get('display_name', 'Unknown')}")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to get channel info: {e}")
            return None
    
    async def get_team_info(self, team_id: str) -> Optional[Dict[str, Any]]:
        """Get team information"""
        
        try:
            response = await self._make_request("GET", f"/teams/{team_id}")
            
            if response:
                logger.debug(f"Retrieved info for team: {response.get('display_name', 'Unknown')}")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to get team info: {e}")
            return None
    
    async def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information"""
        
        try:
            response = await self._make_request("GET", f"/users/{user_id}")
            
            if response:
                logger.debug(f"Retrieved info for user: {response.get('username', 'Unknown')}")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to get user info: {e}")
            return None
    
    async def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Get current user information"""
        
        try:
            response = await self._make_request("GET", "/users/me")
            
            if response:
                logger.info(f"Connected as user: {response.get('username', 'Unknown')}")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to get current user: {e}")
            return None
    
    async def test_connection(self) -> bool:
        """Test connection to Mattermost"""
        
        try:
            user = await self.get_current_user()
            return user is not None
            
        except Exception as e:
            logger.error(f"Mattermost connection test failed: {e}")
            return False
    
    async def get_channel_members(self, channel_id: str) -> List[Dict[str, Any]]:
        """Get channel members"""
        
        try:
            response = await self._make_request("GET", f"/channels/{channel_id}/members")
            
            if response:
                logger.debug(f"Retrieved {len(response)} members for channel {channel_id}")
                return response
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get channel members: {e}")
            return []
    
    async def search_posts(
        self,
        team_id: str,
        terms: str,
        is_or_search: bool = False,
        time_zone_offset: int = 0,
        include_deleted_channels: bool = False,
        page: int = 0,
        per_page: int = 60
    ) -> Optional[Dict[str, Any]]:
        """Search posts in team"""
        
        try:
            search_params = {
                "terms": terms,
                "is_or_search": is_or_search,
                "time_zone_offset": time_zone_offset,
                "include_deleted_channels": include_deleted_channels,
                "page": page,
                "per_page": per_page
            }
            
            response = await self._make_request(
                "POST",
                f"/teams/{team_id}/posts/search",
                json=search_params
            )
            
            if response:
                post_count = len(response.get("posts", {}))
                logger.debug(f"Found {post_count} posts matching search terms")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to search posts: {e}")
            return None
    
    async def get_posts_around(
        self,
        channel_id: str,
        post_id: str,
        before: int = 10,
        after: int = 10
    ) -> Optional[Dict[str, Any]]:
        """Get posts around a specific post"""
        
        try:
            params = {
                "before": before,
                "after": after
            }
            
            response = await self._make_request(
                "GET",
                f"/channels/{channel_id}/posts/{post_id}/context",
                params=params
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to get posts around {post_id}: {e}")
            return None
    
    async def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get file information"""
        
        try:
            response = await self._make_request("GET", f"/files/{file_id}/info")
            
            if response:
                logger.debug(f"Retrieved info for file: {response.get('name', 'Unknown')}")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to get file info: {e}")
            return None
    
    async def download_file(self, file_id: str) -> Optional[bytes]:
        """Download file content"""
        
        url = f"{self.base_url}/api/v4/files/{file_id}"
        
        try:
            async with aiohttp.ClientSession(
                headers=self.headers,
                timeout=self.timeout
            ) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        logger.debug(f"Downloaded file {file_id}, size: {len(content)} bytes")
                        return content
                    else:
                        logger.error(f"Failed to download file: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"File download failed: {e}")
            return None
    
    async def get_channel_stats(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Get channel statistics"""
        
        try:
            response = await self._make_request("GET", f"/channels/{channel_id}/stats")
            
            if response:
                logger.debug(f"Retrieved stats for channel {channel_id}")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to get channel stats: {e}")
            return None
    
    async def post_message(
        self,
        channel_id: str,
        message: str,
        root_id: Optional[str] = None,
        props: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Post a message to a channel"""
        
        try:
            post_data = {
                "channel_id": channel_id,
                "message": message
            }
            
            if root_id:
                post_data["root_id"] = root_id
            
            if props:
                post_data["props"] = props
            
            response = await self._make_request(
                "POST",
                "/posts",
                json=post_data
            )
            
            if response:
                logger.debug(f"Posted message to channel {channel_id}")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to post message: {e}")
            return None
    
    async def update_post(
        self,
        post_id: str,
        message: str,
        props: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Update an existing post"""
        
        try:
            update_data = {
                "id": post_id,
                "message": message
            }
            
            if props:
                update_data["props"] = props
            
            response = await self._make_request(
                "PUT",
                f"/posts/{post_id}",
                json=update_data
            )
            
            if response:
                logger.debug(f"Updated post {post_id}")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to update post: {e}")
            return None