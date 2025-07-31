#!/bin/bash

# Start all services

echo "Starting Redis server..."
redis-server --daemonize yes

echo "Starting FastAPI server..."
python run.py &

echo "Starting Celery worker..."
celery -A app.services.video_service.celery_app worker --loglevel=info --concurrency=2 &

echo "Starting Celery beat scheduler..."
celery -A app.services.video_service.celery_app beat --loglevel=info &

echo "All services started!"
echo "API available at: http://localhost:8000"
echo "API docs available at: http://localhost:8000/docs"

# Keep script running
wait
