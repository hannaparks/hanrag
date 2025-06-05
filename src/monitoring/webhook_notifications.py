"""
Webhook notification system for ingestion status updates
"""

import asyncio
import hmac
import hashlib
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, field, asdict
from enum import Enum
import aiohttp
from loguru import logger
from urllib.parse import urlparse

from ..config.settings import settings


class WebhookEventType(Enum):
    """Webhook event types"""
    INGESTION_STARTED = "ingestion.started"
    INGESTION_PROGRESS = "ingestion.progress"
    INGESTION_COMPLETED = "ingestion.completed"
    INGESTION_FAILED = "ingestion.failed"
    INGESTION_CANCELLED = "ingestion.cancelled"
    BATCH_COMPLETED = "batch.completed"
    SYSTEM_HEALTH = "system.health"


@dataclass
class WebhookPayload:
    """Webhook payload structure"""
    event_type: WebhookEventType
    event_id: str
    timestamp: str
    task_id: str
    source_type: str
    source_identifier: str
    status: str
    progress_percentage: float
    current_step: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    duration_seconds: Optional[float] = None
    
    @classmethod
    def from_task(cls, task, event_type: WebhookEventType) -> "WebhookPayload":
        """Create webhook payload from ingestion task"""
        return cls(
            event_type=event_type,
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            task_id=task.task_id,
            source_type=task.source_type.value,
            source_identifier=task.source_identifier,
            status=task.status.value,
            progress_percentage=task.progress_percentage,
            current_step=task.current_step,
            metadata=task.metadata,
            error_message=task.error_message,
            result=task.result,
            duration_seconds=task.duration_seconds
        )


@dataclass
class WebhookDelivery:
    """Webhook delivery attempt tracking"""
    delivery_id: str
    webhook_url: str
    payload: WebhookPayload
    attempt: int
    timestamp: datetime
    status: Literal["pending", "success", "failed", "retrying"]
    response_status: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None
    duration_ms: Optional[int] = None


@dataclass
class WebhookEndpoint:
    """Webhook endpoint configuration"""
    url: str
    enabled: bool = True
    secret: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3
    event_types: List[WebhookEventType] = field(default_factory=lambda: list(WebhookEventType))
    headers: Dict[str, str] = field(default_factory=dict)
    
    def should_deliver(self, event_type: WebhookEventType) -> bool:
        """Check if this endpoint should receive the event type"""
        return self.enabled and (not self.event_types or event_type in self.event_types)


class WebhookSigner:
    """Handle webhook signature generation and verification"""
    
    @staticmethod
    def generate_signature(payload: str, secret: str) -> str:
        """Generate HMAC-SHA256 signature for webhook payload"""
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    @staticmethod
    def verify_signature(payload: str, signature: str, secret: str) -> bool:
        """Verify webhook signature"""
        expected_signature = WebhookSigner.generate_signature(payload, secret)
        return hmac.compare_digest(signature, expected_signature)


class WebhookNotificationService:
    """Service for sending webhook notifications about ingestion status"""
    
    def __init__(self):
        self.endpoints: List[WebhookEndpoint] = []
        self.deliveries: Dict[str, WebhookDelivery] = {}
        self.delivery_history: List[WebhookDelivery] = []
        self.max_history_size = 1000
        self.session: Optional[aiohttp.ClientSession] = None
        self._setup_endpoints()
    
    def _setup_endpoints(self):
        """Setup webhook endpoints from configuration"""
        if not settings.ENABLE_WEBHOOK_NOTIFICATIONS:
            logger.info("Webhook notifications are disabled")
            return
        
        if not settings.WEBHOOK_ENDPOINTS:
            logger.info("No webhook endpoints configured")
            return
        
        # Parse comma-separated webhook URLs
        urls = [url.strip() for url in settings.WEBHOOK_ENDPOINTS.split(",") if url.strip()]
        
        for url in urls:
            try:
<<<<<<< HEAD
                parsed_url = urlparse(url)
                if not parsed_url.scheme or not parsed_url.netloc:
                    logger.warning(f"Invalid webhook URL: {url}")
                    continue
=======
                logger.info(f"Adding webhook URL: {url}")
>>>>>>> 66c74c8
                
                endpoint = WebhookEndpoint(
                    url=url,
                    secret=settings.WEBHOOK_SECRET,
                    timeout=settings.WEBHOOK_TIMEOUT,
                    retry_attempts=settings.WEBHOOK_RETRY_ATTEMPTS,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "RAG-Pipeline-Webhook/1.0",
                        "X-Webhook-Source": "hanrag-system"
                    }
                )
                self.endpoints.append(endpoint)
                logger.info(f"Configured webhook endpoint: {url}")
                
            except Exception as e:
                logger.error(f"Error configuring webhook endpoint {url}: {e}")
    
    async def start(self):
        """Start the webhook service"""
        if self.endpoints:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=2)
            timeout = aiohttp.ClientTimeout(total=settings.WEBHOOK_TIMEOUT)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
            logger.info(f"Webhook notification service started with {len(self.endpoints)} endpoints")
    
    async def stop(self):
        """Stop the webhook service"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Webhook notification service stopped")
    
    async def send_notification(
        self,
        task,
        event_type: WebhookEventType
    ) -> List[WebhookDelivery]:
        """Send webhook notification for task event"""
        
        if not self.endpoints or not self.session:
            return []
        
        # Create payload
        payload = WebhookPayload.from_task(task, event_type)
        
        # Send to all configured endpoints
        deliveries = []
        tasks = []
        
        for endpoint in self.endpoints:
            if endpoint.should_deliver(event_type):
                delivery_task = asyncio.create_task(
                    self._deliver_webhook(endpoint, payload)
                )
                tasks.append(delivery_task)
        
        if tasks:
            # Execute all deliveries concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect successful deliveries
            for result in results:
                if isinstance(result, WebhookDelivery):
                    deliveries.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Webhook delivery error: {result}")
        
        return deliveries
    
    async def _deliver_webhook(
        self,
        endpoint: WebhookEndpoint,
        payload: WebhookPayload
    ) -> WebhookDelivery:
        """Deliver webhook to a specific endpoint with retry logic"""
        
        delivery = WebhookDelivery(
            delivery_id=str(uuid.uuid4()),
            webhook_url=endpoint.url,
            payload=payload,
            attempt=0,
            timestamp=datetime.now(),
            status="pending"
        )
        
        self.deliveries[delivery.delivery_id] = delivery
        
        # Prepare payload
        payload_dict = asdict(payload)
        payload_dict["event_type"] = payload.event_type.value
        payload_json = str(payload_dict).replace("'", '"')  # Simple JSON conversion
        
        # Add signature if secret is configured
        headers = endpoint.headers.copy()
        if endpoint.secret:
            signature = WebhookSigner.generate_signature(payload_json, endpoint.secret)
            headers["X-Webhook-Signature"] = signature
        
        # Retry logic
        for attempt in range(1, endpoint.retry_attempts + 1):
            delivery.attempt = attempt
            delivery.status = "retrying" if attempt > 1 else "pending"
            delivery.timestamp = datetime.now()
            
            try:
<<<<<<< HEAD
=======
                # Webhook delivery without SSRF protection
                logger.debug(f"Sending webhook to {endpoint.url}")
                
>>>>>>> 66c74c8
                start_time = time.time()
                
                async with self.session.post(
                    endpoint.url,
                    data=payload_json,
                    headers=headers
                ) as response:
                    duration_ms = int((time.time() - start_time) * 1000)
                    response_body = await response.text()
                    
                    delivery.response_status = response.status
                    delivery.response_body = response_body[:1000]  # Limit body size
                    delivery.duration_ms = duration_ms
                    
                    if 200 <= response.status < 300:
                        delivery.status = "success"
                        logger.info(
                            f"Webhook delivered successfully: {endpoint.url} "
                            f"(attempt {attempt}, {duration_ms}ms, status {response.status})"
                        )
                        break
                    else:
                        delivery.error_message = f"HTTP {response.status}: {response_body[:200]}"
                        logger.warning(
                            f"Webhook delivery failed: {endpoint.url} "
                            f"(attempt {attempt}, status {response.status}): {response_body[:200]}"
                        )
                
            except asyncio.TimeoutError:
                delivery.error_message = "Request timeout"
                logger.warning(f"Webhook delivery timeout: {endpoint.url} (attempt {attempt})")
                
            except Exception as e:
                delivery.error_message = str(e)
                logger.warning(f"Webhook delivery error: {endpoint.url} (attempt {attempt}): {e}")
            
            # Wait before retry (exponential backoff)
            if attempt < endpoint.retry_attempts:
                wait_time = min(2 ** (attempt - 1), 30)  # Max 30 seconds
                await asyncio.sleep(wait_time)
        
        # Final status
        if delivery.status != "success":
            delivery.status = "failed"
            logger.error(
                f"Webhook delivery failed permanently: {endpoint.url} "
                f"(all {endpoint.retry_attempts} attempts failed)"
            )
        
        # Add to history
        self._add_to_history(delivery)
        
        return delivery
    
    def _add_to_history(self, delivery: WebhookDelivery):
        """Add delivery to history with size limit"""
        self.delivery_history.append(delivery)
        
        # Keep history size under limit
        if len(self.delivery_history) > self.max_history_size:
            self.delivery_history = self.delivery_history[-self.max_history_size:]
    
    async def send_batch_completion_notification(
        self,
        task_ids: List[str],
        batch_metadata: Dict[str, Any]
    ):
        """Send notification for batch completion"""
        
        if not self.endpoints or not self.session:
            return []
        
        # Create batch completion payload
        payload = WebhookPayload(
            event_type=WebhookEventType.BATCH_COMPLETED,
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            task_id=f"batch_{uuid.uuid4().hex[:8]}",
            source_type="batch",
            source_identifier="multiple",
            status="completed",
            progress_percentage=100.0,
            current_step="batch completed",
            metadata={
                "task_ids": task_ids,
                "batch_size": len(task_ids),
                **batch_metadata
            }
        )
        
        # Send to all endpoints
        deliveries = []
        tasks = []
        
        for endpoint in self.endpoints:
            if endpoint.should_deliver(WebhookEventType.BATCH_COMPLETED):
                delivery_task = asyncio.create_task(
                    self._deliver_webhook(endpoint, payload)
                )
                tasks.append(delivery_task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            deliveries = [r for r in results if isinstance(r, WebhookDelivery)]
        
        return deliveries
    
    async def send_health_notification(self, health_data: Dict[str, Any]):
        """Send system health notification"""
        
        if not self.endpoints or not self.session:
            return []
        
        payload = WebhookPayload(
            event_type=WebhookEventType.SYSTEM_HEALTH,
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            task_id="system_health",
            source_type="system",
            source_identifier="health_check",
            status="info",
            progress_percentage=100.0,
            current_step="health check",
            metadata=health_data
        )
        
        # Send to endpoints that want health notifications
        deliveries = []
        tasks = []
        
        for endpoint in self.endpoints:
            if endpoint.should_deliver(WebhookEventType.SYSTEM_HEALTH):
                delivery_task = asyncio.create_task(
                    self._deliver_webhook(endpoint, payload)
                )
                tasks.append(delivery_task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            deliveries = [r for r in results if isinstance(r, WebhookDelivery)]
        
        return deliveries
    
    def get_delivery_status(self, delivery_id: str) -> Optional[WebhookDelivery]:
        """Get delivery status by ID"""
        return self.deliveries.get(delivery_id)
    
    def get_delivery_history(
        self,
        limit: int = 100,
        status_filter: Optional[str] = None,
        endpoint_filter: Optional[str] = None
    ) -> List[WebhookDelivery]:
        """Get delivery history with optional filters"""
        
        history = self.delivery_history[-limit:] if limit else self.delivery_history
        
        if status_filter:
            history = [d for d in history if d.status == status_filter]
        
        if endpoint_filter:
            history = [d for d in history if endpoint_filter in d.webhook_url]
        
        return sorted(history, key=lambda d: d.timestamp, reverse=True)
    
    def get_delivery_statistics(self) -> Dict[str, Any]:
        """Get delivery statistics"""
        
        total_deliveries = len(self.delivery_history)
        
        if total_deliveries == 0:
            return {
                "total_deliveries": 0,
                "success_rate": 0.0,
                "average_duration_ms": 0.0,
                "status_distribution": {},
                "endpoint_distribution": {}
            }
        
        # Calculate statistics
        successful = len([d for d in self.delivery_history if d.status == "success"])
        success_rate = (successful / total_deliveries) * 100
        
        # Average duration (only successful deliveries)
        durations = [d.duration_ms for d in self.delivery_history if d.duration_ms is not None]
        average_duration = sum(durations) / len(durations) if durations else 0.0
        
        # Status distribution
        status_counts = {}
        for delivery in self.delivery_history:
            status_counts[delivery.status] = status_counts.get(delivery.status, 0) + 1
        
        # Endpoint distribution
        endpoint_counts = {}
        for delivery in self.delivery_history:
            endpoint_counts[delivery.webhook_url] = endpoint_counts.get(delivery.webhook_url, 0) + 1
        
        return {
            "total_deliveries": total_deliveries,
            "successful_deliveries": successful,
            "success_rate": success_rate,
            "average_duration_ms": average_duration,
            "status_distribution": status_counts,
            "endpoint_distribution": endpoint_counts,
            "configured_endpoints": len(self.endpoints),
            "active_endpoints": len([e for e in self.endpoints if e.enabled])
        }
    
    def add_endpoint(self, endpoint: WebhookEndpoint):
        """Add a new webhook endpoint"""
        self.endpoints.append(endpoint)
        logger.info(f"Added webhook endpoint: {endpoint.url}")
    
    def remove_endpoint(self, url: str) -> bool:
        """Remove webhook endpoint by URL"""
        original_count = len(self.endpoints)
        self.endpoints = [e for e in self.endpoints if e.url != url]
        removed = len(self.endpoints) < original_count
        
        if removed:
            logger.info(f"Removed webhook endpoint: {url}")
        
        return removed
    
    def update_endpoint(self, url: str, **kwargs) -> bool:
        """Update webhook endpoint configuration"""
        for endpoint in self.endpoints:
            if endpoint.url == url:
                for key, value in kwargs.items():
                    if hasattr(endpoint, key):
                        setattr(endpoint, key, value)
                logger.info(f"Updated webhook endpoint: {url}")
                return True
        return False


# Global webhook service instance
webhook_service = WebhookNotificationService()


# Convenience functions for common webhook events
async def notify_ingestion_started(task) -> List[WebhookDelivery]:
    """Send ingestion started notification"""
    return await webhook_service.send_notification(task, WebhookEventType.INGESTION_STARTED)


async def notify_ingestion_progress(task) -> List[WebhookDelivery]:
    """Send ingestion progress notification"""
    return await webhook_service.send_notification(task, WebhookEventType.INGESTION_PROGRESS)


async def notify_ingestion_completed(task) -> List[WebhookDelivery]:
    """Send ingestion completed notification"""
    return await webhook_service.send_notification(task, WebhookEventType.INGESTION_COMPLETED)


async def notify_ingestion_failed(task) -> List[WebhookDelivery]:
    """Send ingestion failed notification"""
    return await webhook_service.send_notification(task, WebhookEventType.INGESTION_FAILED)


async def notify_ingestion_cancelled(task) -> List[WebhookDelivery]:
    """Send ingestion cancelled notification"""
    return await webhook_service.send_notification(task, WebhookEventType.INGESTION_CANCELLED)