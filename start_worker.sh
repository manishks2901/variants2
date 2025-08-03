#!/bin/bash

# Start Celery worker for video processing with enhanced logging and task events
echo "Starting Celery worker with enhanced logging and task events..."
# Remove --logfile to show logs in terminal, Python logging will handle file output
celery -A app.services.video_service.celery_app worker --loglevel=info --concurrency=2 -E
