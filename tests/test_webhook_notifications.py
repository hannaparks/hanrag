"""
Tests for webhook notification system
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import aiohttp
from aiohttp import web

from src.monitoring.webhook_notifications import (
    WebhookNotificationService, WebhookEventType, WebhookPayload,
    WebhookEndpoint, WebhookSigner, notify_ingestion_started,
    notify_ingestion_completed, webhook_service
)
from src.ingestion.pipeline import IngestionTask, IngestionStatus, SourceType


class TestWebhookSigner:
    """Test webhook signature generation and verification"""
    
    def test_generate_signature(self):
        """Test signature generation"""
        payload = '{"test": "data"}'
        secret = "test_secret"
        
        signature = WebhookSigner.generate_signature(payload, secret)
        
        assert signature.startswith("sha256=")
        assert len(signature) == 71  # sha256= + 64 hex chars
    
    def test_verify_signature(self):
        """Test signature verification"""
        payload = '{"test": "data"}'
        secret = "test_secret"
        
        signature = WebhookSigner.generate_signature(payload, secret)
        
        # Valid signature should verify
        assert WebhookSigner.verify_signature(payload, signature, secret)
        
        # Invalid signature should not verify
        assert not WebhookSigner.verify_signature(payload, "invalid", secret)
        
        # Wrong secret should not verify
        assert not WebhookSigner.verify_signature(payload, signature, "wrong_secret")


class TestWebhookPayload:
    """Test webhook payload creation"""
    
    def test_from_task(self):
        """Test creating payload from ingestion task"""
        task = IngestionTask(
            task_id="test_task",
            source_type=SourceType.TEXT_CONTENT,
            source_identifier="test_source",
            status=IngestionStatus.COMPLETED,
            progress_percentage=100.0,
            current_step="Completed",
            metadata={"test": "value"}
        )
        
        payload = WebhookPayload.from_task(task, WebhookEventType.INGESTION_COMPLETED)
        
        assert payload.event_type == WebhookEventType.INGESTION_COMPLETED
        assert payload.task_id == "test_task"
        assert payload.source_type == "text_content"
        assert payload.status == "completed"
        assert payload.progress_percentage == 100.0
        assert payload.metadata == {"test": "value"}


class TestWebhookEndpoint:
    """Test webhook endpoint configuration"""
    
    def test_should_deliver_all_events(self):
        """Test endpoint that accepts all events"""
        endpoint = WebhookEndpoint(url="http://test.com")
        
        # Should deliver all event types by default
        assert endpoint.should_deliver(WebhookEventType.INGESTION_STARTED)
        assert endpoint.should_deliver(WebhookEventType.INGESTION_COMPLETED)
        assert endpoint.should_deliver(WebhookEventType.SYSTEM_HEALTH)
    
    def test_should_deliver_filtered_events(self):
        """Test endpoint with event type filtering"""
        endpoint = WebhookEndpoint(
            url="http://test.com",
            event_types=[WebhookEventType.INGESTION_COMPLETED]
        )
        
        # Should only deliver specified event types
        assert not endpoint.should_deliver(WebhookEventType.INGESTION_STARTED)
        assert endpoint.should_deliver(WebhookEventType.INGESTION_COMPLETED)
        assert not endpoint.should_deliver(WebhookEventType.SYSTEM_HEALTH)
    
    def test_disabled_endpoint(self):
        """Test disabled endpoint"""
        endpoint = WebhookEndpoint(url="http://test.com", enabled=False)
        
        # Should not deliver any events when disabled
        assert not endpoint.should_deliver(WebhookEventType.INGESTION_COMPLETED)


class TestWebhookNotificationService:
    """Test webhook notification service"""
    
    @pytest.fixture
    def mock_webhook_server(self):
        """Create a mock webhook server for testing"""
        received_webhooks = []
        
        async def webhook_handler(request):
            body = await request.text()
            received_webhooks.append({
                'headers': dict(request.headers),
                'body': body,
                'method': request.method
            })
            return web.Response(status=200, text='OK')
        
        app = web.Application()
        app.router.add_post('/webhook', webhook_handler)
        
        return app, received_webhooks
    
    @pytest.fixture
    def service(self):
        """Create webhook service for testing"""
        service = WebhookNotificationService()
        service.endpoints = []  # Clear any default endpoints
        return service
    
    @pytest.fixture
    async def started_service(self, service):
        """Create and start webhook service"""
        await service.start()
        yield service
        await service.stop()
    
    def test_setup_endpoints_from_config(self):
        """Test endpoint setup from configuration"""
        with patch('src.monitoring.webhook_notifications.settings') as mock_settings:
            mock_settings.ENABLE_WEBHOOK_NOTIFICATIONS = True
            mock_settings.WEBHOOK_ENDPOINTS = "http://test1.com,http://test2.com"
            mock_settings.WEBHOOK_SECRET = "test_secret"
            mock_settings.WEBHOOK_TIMEOUT = 30
            mock_settings.WEBHOOK_RETRY_ATTEMPTS = 3
            
            service = WebhookNotificationService()
            
            assert len(service.endpoints) == 2
            assert service.endpoints[0].url == "http://test1.com"
            assert service.endpoints[1].url == "http://test2.com"
            assert service.endpoints[0].secret == "test_secret"
    
    def test_setup_endpoints_disabled(self):
        """Test endpoint setup when disabled"""
        with patch('src.monitoring.webhook_notifications.settings') as mock_settings:
            mock_settings.ENABLE_WEBHOOK_NOTIFICATIONS = False
            
            service = WebhookNotificationService()
            
            assert len(service.endpoints) == 0
    
    async def test_send_notification_no_endpoints(self, started_service):
        """Test sending notification with no endpoints"""
        task = IngestionTask(
            task_id="test",
            source_type=SourceType.TEXT_CONTENT,
            source_identifier="test"
        )
        
        deliveries = await started_service.send_notification(
            task, WebhookEventType.INGESTION_STARTED
        )
        
        assert len(deliveries) == 0
    
    async def test_send_notification_success(self, started_service):
        """Test successful webhook delivery"""
        # Add test endpoint
        endpoint = WebhookEndpoint(url="http://httpbin.org/post")
        started_service.add_endpoint(endpoint)
        
        task = IngestionTask(
            task_id="test",
            source_type=SourceType.TEXT_CONTENT,
            source_identifier="test",
            status=IngestionStatus.COMPLETED
        )
        
        deliveries = await started_service.send_notification(
            task, WebhookEventType.INGESTION_COMPLETED
        )
        
        # Should have attempted delivery
        assert len(deliveries) == 1
        
        delivery = deliveries[0]
        assert delivery.webhook_url == "http://httpbin.org/post"
        assert delivery.payload.task_id == "test"
        assert delivery.attempt > 0
    
    async def test_delivery_statistics(self, started_service):
        """Test delivery statistics"""
        # Initially empty
        stats = started_service.get_delivery_statistics()
        assert stats["total_deliveries"] == 0
        assert stats["success_rate"] == 0.0
        
        # Add some mock deliveries to history
        from src.monitoring.webhook_notifications import WebhookDelivery
        
        delivery1 = WebhookDelivery(
            delivery_id="1",
            webhook_url="http://test.com",
            payload=Mock(),
            attempt=1,
            timestamp=datetime.now(),
            status="success",
            duration_ms=100
        )
        
        delivery2 = WebhookDelivery(
            delivery_id="2",
            webhook_url="http://test.com",
            payload=Mock(),
            attempt=1,
            timestamp=datetime.now(),
            status="failed"
        )
        
        started_service.delivery_history = [delivery1, delivery2]
        
        stats = started_service.get_delivery_statistics()
        assert stats["total_deliveries"] == 2
        assert stats["successful_deliveries"] == 1
        assert stats["success_rate"] == 50.0
        assert stats["average_duration_ms"] == 100.0
    
    def test_add_remove_endpoints(self, service):
        """Test adding and removing endpoints"""
        endpoint = WebhookEndpoint(url="http://test.com")
        
        # Add endpoint
        service.add_endpoint(endpoint)
        assert len(service.endpoints) == 1
        assert service.endpoints[0].url == "http://test.com"
        
        # Remove endpoint
        removed = service.remove_endpoint("http://test.com")
        assert removed
        assert len(service.endpoints) == 0
        
        # Try to remove non-existent endpoint
        removed = service.remove_endpoint("http://nonexistent.com")
        assert not removed
    
    def test_update_endpoint(self, service):
        """Test updating endpoint configuration"""
        endpoint = WebhookEndpoint(url="http://test.com", enabled=True)
        service.add_endpoint(endpoint)
        
        # Update endpoint
        updated = service.update_endpoint("http://test.com", enabled=False, timeout=60)
        assert updated
        assert not service.endpoints[0].enabled
        assert service.endpoints[0].timeout == 60
        
        # Try to update non-existent endpoint
        updated = service.update_endpoint("http://nonexistent.com", enabled=True)
        assert not updated


class TestConvenienceFunctions:
    """Test convenience functions for webhook notifications"""
    
    @pytest.fixture
    def mock_task(self):
        """Create a mock ingestion task"""
        return IngestionTask(
            task_id="test_task",
            source_type=SourceType.TEXT_CONTENT,
            source_identifier="test_source"
        )
    
    @patch('src.monitoring.webhook_notifications.webhook_service')
    async def test_notify_ingestion_started(self, mock_service, mock_task):
        """Test ingestion started notification"""
        mock_service.send_notification = AsyncMock(return_value=[])
        
        await notify_ingestion_started(mock_task)
        
        mock_service.send_notification.assert_called_once_with(
            mock_task, WebhookEventType.INGESTION_STARTED
        )
    
    @patch('src.monitoring.webhook_notifications.webhook_service')
    async def test_notify_ingestion_completed(self, mock_service, mock_task):
        """Test ingestion completed notification"""
        mock_service.send_notification = AsyncMock(return_value=[])
        
        await notify_ingestion_completed(mock_task)
        
        mock_service.send_notification.assert_called_once_with(
            mock_task, WebhookEventType.INGESTION_COMPLETED
        )


@pytest.mark.integration
class TestWebhookIntegration:
    """Integration tests for webhook notifications"""
    
    async def test_webhook_with_real_server(self):
        """Test webhook delivery to a real HTTP server"""
        # This would require a real webhook endpoint
        # For now, we'll skip this test in CI/CD
        pytest.skip("Integration test - requires real webhook endpoint")
    
    async def test_pipeline_integration(self):
        """Test webhook integration with ingestion pipeline"""
        # This would test the full integration with the pipeline
        # For now, we'll skip this test
        pytest.skip("Integration test - requires full pipeline setup")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])