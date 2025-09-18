"""
Celery configuration for background task processing
"""

import logging
import ssl
from celery import Celery
from celery.signals import setup_logging

from .config import settings

logger = logging.getLogger(__name__)

# Create Celery instance
celery_app = Celery(
    "biomlstudio",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.tasks.ml_tasks",
        "app.tasks.data_processing",
        "app.tasks.model_training"
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task routing
    task_routes={
        "app.tasks.ml_tasks.*": {"queue": "ml_queue"},
        "app.tasks.data_processing.*": {"queue": "data_queue"},
        "app.tasks.model_training.*": {"queue": "training_queue"},
    },
    
    # Result backend settings
    result_expires=3600,  # 1 hour
    
    # Task execution settings
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    task_soft_time_limit=settings.MAX_TRAINING_TIME_SECONDS,
    task_time_limit=settings.MAX_TRAINING_TIME_SECONDS + 60,
    broker_connection_retry_on_startup=True,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Security
    worker_hijack_root_logger=False,
    worker_log_color=False,
)

# Queue configuration
celery_app.conf.task_queues = {
    "ml_queue": {
        "exchange": "ml_queue",
        "exchange_type": "direct",
        "routing_key": "ml_queue",
    },
    "data_queue": {
        "exchange": "data_queue", 
        "exchange_type": "direct",
        "routing_key": "data_queue",
    },
    "training_queue": {
        "exchange": "training_queue",
        "exchange_type": "direct", 
        "routing_key": "training_queue",
    },
}

# Enable TLS/SSL for Redis (Upstash uses TLS). Accept cert without verification if needed.
# If your environment has proper CA certificates installed, you can change CERT_NONE to CERT_REQUIRED.
celery_app.conf.broker_use_ssl = {
    "ssl_cert_reqs": ssl.CERT_NONE,
}
celery_app.conf.redis_backend_use_ssl = {
    "ssl_cert_reqs": ssl.CERT_NONE,
}

# Transport options for SSL
celery_app.conf.broker_transport_options = {
    "ssl_cert_reqs": ssl.CERT_NONE,
}
celery_app.conf.result_backend_transport_options = {
    "ssl_cert_reqs": ssl.CERT_NONE,
}


@setup_logging.connect
def config_loggers(*args, **kwargs):
    """Configure logging for Celery workers"""
    from logging.config import dictConfig
    
    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "[%(asctime)s: %(levelname)s/%(name)s] %(message)s",
            },
        },
        "handlers": {
            "console": {
                "level": settings.LOG_LEVEL,
                "class": "logging.StreamHandler",
                "formatter": "default",
            },
        },
        "loggers": {
            "celery": {
                "level": settings.LOG_LEVEL,
                "handlers": ["console"],
                "propagate": False,
            },
            "app": {
                "level": settings.LOG_LEVEL,
                "handlers": ["console"],
                "propagate": False,
            },
        },
        "root": {
            "level": settings.LOG_LEVEL,
            "handlers": ["console"],
        },
    })


# Celery beat schedule for periodic tasks
celery_app.conf.beat_schedule = {
    "cleanup-expired-jobs": {
        "task": "app.tasks.maintenance.cleanup_expired_jobs",
        "schedule": 3600.0,  # Every hour
    },
    "update-model-metrics": {
        "task": "app.tasks.maintenance.update_model_metrics",
        "schedule": 1800.0,  # Every 30 minutes
    },
    "health-check": {
        "task": "app.tasks.maintenance.health_check",
        "schedule": 300.0,  # Every 5 minutes
    },
}
