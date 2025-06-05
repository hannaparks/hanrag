"""
Webhook types and enums - separate module to avoid circular imports
"""

from enum import Enum


class WebhookEventType(Enum):
    """Webhook event types"""
    INGESTION_STARTED = "ingestion.started"
    INGESTION_PROGRESS = "ingestion.progress"
    INGESTION_COMPLETED = "ingestion.completed"
    INGESTION_FAILED = "ingestion.failed"
    INGESTION_CANCELLED = "ingestion.cancelled"
    BATCH_COMPLETED = "batch.completed"
    SYSTEM_HEALTH = "system.health"