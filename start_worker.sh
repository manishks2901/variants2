#!/bin/bash

# Start Celery worker for video processing
celery -A app.services.video_service.celery_app worker --loglevel=info --concurrency=2
