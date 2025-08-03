#!/usr/bin/env python3
"""
Celery monitoring script to watch task events in real-time
Usage: python monitor_celery.py
"""

import time
from celery import Celery
from decouple import config

# Initialize Celery app for monitoring
celery_app = Celery(
    'video_processing',
    broker=config('CELERY_BROKER_URL', default='redis://localhost:6379/0'),
    backend=config('CELERY_RESULT_BACKEND', default='redis://localhost:6379/0'),
)

def monitor_tasks():
    """Monitor Celery tasks in real-time"""
    print("🔍 Starting Celery task monitor...")
    print("📊 Waiting for task events...")
    print("-" * 50)
    
    def on_event(event):
        """Handle task events"""
        task_id = event.get('uuid', 'Unknown')
        task_name = event.get('name', 'Unknown')
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.get('timestamp', time.time())))
        
        if event['type'] == 'task-sent':
            print(f"📤 [{timestamp}] Task SENT: {task_name} (ID: {task_id[:8]}...)")
            
        elif event['type'] == 'task-received':
            print(f"📥 [{timestamp}] Task RECEIVED: {task_name} (ID: {task_id[:8]}...)")
            
        elif event['type'] == 'task-started':
            print(f"🚀 [{timestamp}] Task STARTED: {task_name} (ID: {task_id[:8]}...)")
            
        elif event['type'] == 'task-succeeded':
            runtime = event.get('runtime', 'Unknown')
            print(f"✅ [{timestamp}] Task SUCCEEDED: {task_name} (ID: {task_id[:8]}...) Runtime: {runtime}s")
            
        elif event['type'] == 'task-failed':
            exception = event.get('exception', 'Unknown error')
            print(f"❌ [{timestamp}] Task FAILED: {task_name} (ID: {task_id[:8]}...) Error: {exception}")
            
        elif event['type'] == 'task-retried':
            print(f"🔄 [{timestamp}] Task RETRIED: {task_name} (ID: {task_id[:8]}...)")
            
        elif event['type'] == 'task-revoked':
            print(f"🚫 [{timestamp}] Task REVOKED: {task_name} (ID: {task_id[:8]}...)")
    
    # Start monitoring
    try:
        with celery_app.connection() as connection:
            recv = celery_app.events.Receiver(connection, handlers={
                'task-sent': on_event,
                'task-received': on_event,
                'task-started': on_event,
                'task-succeeded': on_event,
                'task-failed': on_event,
                'task-retried': on_event,
                'task-revoked': on_event,
            })
            recv.capture(limit=None, timeout=None, wakeup=True)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped.")
    except Exception as e:
        print(f"❌ Monitoring error: {e}")

if __name__ == "__main__":
    monitor_tasks()
