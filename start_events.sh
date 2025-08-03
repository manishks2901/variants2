#!/bin/bash

# Start Celery Events monitor (Flower alternative)
echo "Starting Celery Events Monitor..."
celery -A app.services.video_service.celery_app events
