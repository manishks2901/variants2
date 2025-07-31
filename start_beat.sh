#!/bin/bash

# Start Celery beat scheduler (for periodic tasks)
celery -A app.services.video_service.celery_app beat --loglevel=info
