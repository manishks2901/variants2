#!/usr/bin/env python3
"""
Celery status checker - shows active tasks, workers, and queues
Usage: python celery_status.py
"""

import sys
from celery import Celery
from decouple import config
import json
from datetime import datetime

# Initialize Celery app
celery_app = Celery(
    'video_processing',
    broker=config('CELERY_BROKER_URL', default='redis://localhost:6379/0'),
    backend=config('CELERY_RESULT_BACKEND', default='redis://localhost:6379/0'),
)

def get_worker_stats():
    """Get worker statistics"""
    print("🏭 WORKER STATUS")
    print("=" * 50)
    
    try:
        # Get active workers
        inspect = celery_app.control.inspect()
        
        # Active workers
        active_workers = inspect.active()
        if active_workers:
            print(f"📊 Active Workers: {len(active_workers)}")
            for worker_name, tasks in active_workers.items():
                print(f"  • {worker_name}: {len(tasks)} active tasks")
                for task in tasks:
                    task_name = task.get('name', 'Unknown')
                    task_id = task.get('id', 'Unknown')[:8]
                    print(f"    - {task_name} (ID: {task_id}...)")
        else:
            print("❌ No active workers found")
        
        print()
        
        # Worker stats
        stats = inspect.stats()
        if stats:
            print("📈 WORKER STATISTICS")
            print("-" * 30)
            for worker_name, worker_stats in stats.items():
                print(f"Worker: {worker_name}")
                pool_stats = worker_stats.get('pool', {})
                total_tasks = worker_stats.get('total', {})
                print(f"  • Pool processes: {pool_stats.get('processes', 'Unknown')}")
                print(f"  • Total tasks: {total_tasks}")
                print()
        
        # Registered tasks
        registered = inspect.registered()
        if registered:
            print("📋 REGISTERED TASKS")
            print("-" * 30)
            for worker_name, tasks in registered.items():
                print(f"Worker: {worker_name}")
                for task in tasks:
                    print(f"  • {task}")
                print()
        
    except Exception as e:
        print(f"❌ Error getting worker stats: {e}")

def get_queue_info():
    """Get queue information"""
    print("📬 QUEUE INFORMATION")
    print("=" * 50)
    
    try:
        inspect = celery_app.control.inspect()
        
        # Reserved tasks (queued)
        reserved = inspect.reserved()
        if reserved:
            total_queued = sum(len(tasks) for tasks in reserved.values())
            print(f"📫 Queued tasks: {total_queued}")
            for worker_name, tasks in reserved.items():
                if tasks:
                    print(f"  Worker {worker_name}: {len(tasks)} queued")
        else:
            print("📭 No queued tasks")
        
        print()
        
    except Exception as e:
        print(f"❌ Error getting queue info: {e}")

def check_broker_connection():
    """Check broker connection"""
    print("🔌 BROKER CONNECTION")
    print("=" * 50)
    
    try:
        with celery_app.connection() as conn:
            conn.ensure_connection(max_retries=3)
            print("✅ Broker connection: OK")
            print(f"📍 Broker URL: {celery_app.conf.broker_url}")
            print(f"💾 Result backend: {celery_app.conf.result_backend}")
    except Exception as e:
        print(f"❌ Broker connection failed: {e}")
    
    print()

def main():
    """Main status checker"""
    print(f"🚀 CELERY STATUS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    check_broker_connection()
    get_worker_stats()
    get_queue_info()
    
    print("✨ Status check complete!")

if __name__ == "__main__":
    main()
