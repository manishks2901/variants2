from celery import Celery
from datetime import datetime
import os
import logging
import uuid
from typing import Optional, Callable
from decouple import config
import pymongo
import asyncio

from .ffmpeg_service import FFmpegTransformationService
from .s3_service import S3Service
from .punchline_service import VideoPunchlineGenerator
from ..database import get_database, connect_to_mongo
from ..models import JobStatus

# Initialize Celery
celery_app = Celery(
    'video_processing',
    broker=config('CELERY_BROKER_URL', default='redis://localhost:6379/0'),
    backend=config('CELERY_RESULT_BACKEND', default='redis://localhost:6379/0'),
    include=['app.services.video_service']
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    result_expires=3600,  # 1 hour
)

def get_sync_database():
    """Get synchronous database connection for Celery workers"""
    try:
        mongodb_uri = config("MONGODB_URI")
        client = pymongo.MongoClient(mongodb_uri)
        db = client.get_database("video-variants")
        return db
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB (sync): {e}")
        return None

async def ensure_database_connection():
    """Ensure database connection is established"""
    db = get_database()
    if db is None:
        await connect_to_mongo()
        db = get_database()
    return db

class VideoProcessingService:
    
    def __init__(self):
        self.s3_service = S3Service()
        self.temp_dir = config('TEMP_DIR', default='/tmp/video-variants')
        os.makedirs(self.temp_dir, exist_ok=True)
    
    async def start_processing_task(self, job_id: str, variants_count: int = 1, min_transformations: int = 9):
        """Start asynchronous video processing task for multiple variants"""
        try:
            # Send task to Celery worker
            task = process_video_task.delay(job_id, variants_count, min_transformations)
            
            # Update job with task ID
            db = get_database()
            await db.jobs.update_one(
                {"id": job_id},
                {"$set": {"celery_task_id": task.id}}
            )
            
            logging.info(f"Started processing task for job {job_id} with {variants_count} variant(s), task ID {task.id}")
            
        except Exception as e:
            logging.error(f"Failed to start processing task for job {job_id}: {e}")
            # Update job status to failed
            db = get_database()
            await db.jobs.update_one(
                {"id": job_id},
                {"$set": {
                    "status": JobStatus.FAILED,
                    "error_message": f"Failed to start processing: {str(e)}"
                }}
            )
            raise

@celery_app.task(bind=True)
def process_video_task(self, job_id: str, variants_count: int = 1, min_transformations: int = 9):
    """
    Celery task for processing multiple video variants - uses sync operations
    """
    try:
        return _process_video_sync(self, job_id, variants_count, min_transformations)
    except Exception as e:
        logging.error(f"Task failed for job {job_id}: {e}")
        raise

def _process_video_sync(task, job_id: str, variants_count: int = 1, min_transformations: int = 9):
    """
    Synchronous function to process multiple video variants
    """
    # Get synchronous database connection
    db = get_sync_database()
    if db is None:
        raise Exception("Could not connect to database")
    
    s3_service = S3Service()
    temp_dir = config('TEMP_DIR', default='/tmp/video-variants')
    
    try:
        # Update job status to processing
        db.jobs.update_one(
            {"id": job_id},
            {"$set": {
                "status": JobStatus.PROCESSING,
                "started_at": datetime.utcnow(),
                "progress": 0.0
            }}
        )
        
        # Get job details
        job = db.jobs.find_one({"id": job_id})
        if not job:
            raise Exception(f"Job {job_id} not found")
        
        # Create progress callback
        def update_progress(progress: float):
            db.jobs.update_one(
                {"id": job_id},
                {"$set": {"progress": progress}}
            )
            # Update Celery task progress
            task.update_state(state='PROGRESS', meta={'progress': progress})
        
        update_progress(5.0)  # Starting
        
        # Download input file from S3
        input_filename = f"input_{job_id}_{job['original_filename']}"
        local_input_path = os.path.join(temp_dir, input_filename)
        
        logging.info(f"Downloading input file for job {job_id}")
        # Use asyncio.run for the async S3 download operation
        asyncio.run(s3_service.download_file(job['s3_input_key'], local_input_path))
        update_progress(10.0)  # Downloaded
        
        # Get video information
        video_info = asyncio.run(FFmpegTransformationService.get_video_info(local_input_path))
        db.jobs.update_one(
            {"id": job_id},
            {"$set": {
                "video_info": video_info,
                "duration": video_info.get('duration')
            }}
        )
        update_progress(15.0)  # Video info retrieved
        
        # Process multiple variants
        variants = []
        base_progress = 15.0
        progress_per_variant = 75.0 / variants_count  # 15% to 90% for processing
        
        for variant_index in range(variants_count):
            variant_id = str(uuid.uuid4())
            logging.info(f"Processing variant {variant_index + 1}/{variants_count} for job {job_id}")
            
            # Create variant-specific output filename
            base_name = os.path.splitext(job['original_filename'])[0]
            extension = os.path.splitext(job['original_filename'])[1]
            variant_filename = f"variant_{variant_index + 1}_{base_name}{extension}"
            s3_output_key = f"output/{job_id}/{variant_filename}"
            local_output_path = os.path.join(temp_dir, f"output_{variant_id}_{variant_filename}")
            
            # Create progress callback for this variant
            variant_start_progress = base_progress + (variant_index * progress_per_variant)
            variant_end_progress = base_progress + ((variant_index + 1) * progress_per_variant)
            
            def variant_progress(progress: float):
                mapped_progress = variant_start_progress + (progress * (progress_per_variant / 100))
                update_progress(mapped_progress)
            
            try:
                # Check if punchlines are enabled for this job
                enable_punchlines = job.get('metadata', {}).get('enable_punchlines', False)
                punchline_variant = job.get('metadata', {}).get('punchline_variant', 1)
                
                punchline_data = None
                if enable_punchlines:
                    try:
                        # Initialize punchline generator
                        punchline_generator = VideoPunchlineGenerator()
                        if punchline_generator.is_available():
                            logging.info(f"Generating punchlines for variant {variant_index + 1}")
                            # Get punchline data
                            punchline_data = asyncio.run(punchline_generator.process_video_with_punchlines(
                                local_input_path, 
                                punchline_variant
                            ))
                        else:
                            logging.warning("Punchlines requested but API keys not configured")
                    except Exception as e:
                        logging.error(f"Punchline generation failed for variant {variant_index + 1}: {e}")
                        # Continue with normal processing if punchlines fail
                
                # Apply transformations for this variant
                if enable_punchlines and punchline_data:
                    # Create a temporary file with punchlines applied first
                    temp_punchline_path = os.path.join(temp_dir, f"punchline_{variant_id}_{variant_filename}")
                    
                    # Apply punchlines to video
                    punchline_generator.create_variant_with_punchlines(
                        local_input_path,
                        punchline_data['punchlines'],
                        temp_punchline_path,
                        punchline_data['style']
                    )
                    
                    # Now apply other transformations to the punchline-enhanced video
                    applied_transformations = asyncio.run(FFmpegTransformationService.apply_transformations(
                        temp_punchline_path,
                        local_output_path,
                        min_transformations,
                        variant_progress
                    ))
                    
                    # Clean up temporary punchline file
                    if os.path.exists(temp_punchline_path):
                        os.remove(temp_punchline_path)
                else:
                    # Apply transformations normally
                    applied_transformations = asyncio.run(FFmpegTransformationService.apply_transformations(
                        local_input_path,
                        local_output_path,
                        min_transformations,
                        variant_progress
                    ))
                
                # Upload processed variant to S3
                logging.info(f"Uploading variant {variant_index + 1} for job {job_id}")
                s3_output_url = asyncio.run(s3_service.upload_processed_file(
                    local_output_path, 
                    s3_output_key
                ))
                
                # Get output file size
                output_file_size = os.path.getsize(local_output_path)
                
                # Create variant record
                variant = {
                    "variant_id": variant_id,
                    "s3_output_key": s3_output_key,
                    "applied_transformations": applied_transformations,
                    "transformation_count": len(applied_transformations),
                    "file_size": output_file_size,
                    "punchline_data": punchline_data if enable_punchlines and punchline_data else None
                }
                
                variants.append(variant)
                
                # Update variants completed count
                db.jobs.update_one(
                    {"id": job_id},
                    {"$set": {"variants_completed": len(variants)}}
                )
                
                # Clean up local variant file
                if os.path.exists(local_output_path):
                    os.remove(local_output_path)
                
                logging.info(f"Completed variant {variant_index + 1}/{variants_count} for job {job_id} with {len(applied_transformations)} transformations")
                
            except Exception as variant_error:
                logging.error(f"Failed to process variant {variant_index + 1} for job {job_id}: {variant_error}")
                # Continue with other variants instead of failing the entire job
                continue
        
        update_progress(90.0)  # All variants processed
        
        # Check if we have at least one successful variant
        if not variants:
            raise Exception("No variants were successfully processed")
        
        # Update job as completed
        db.jobs.update_one(
            {"id": job_id},
            {"$set": {
                "status": JobStatus.COMPLETED,
                "progress": 100.0,
                "completed_at": datetime.utcnow(),
                "variants": variants,
                "variants_completed": len(variants)
            }}
        )
        
        logging.info(f"Job {job_id} completed successfully with {len(variants)}/{variants_count} variants")
        
        # Clean up input file
        try:
            if os.path.exists(local_input_path):
                os.remove(local_input_path)
        except Exception as cleanup_error:
            logging.warning(f"Failed to clean up input file for job {job_id}: {cleanup_error}")
        
        return {
            "job_id": job_id,
            "status": "completed",
            "variants_requested": variants_count,
            "variants_completed": len(variants),
            "variants": variants
        }
        
    except Exception as e:
        logging.error(f"Processing failed for job {job_id}: {e}")
        
        # Update job as failed
        db.jobs.update_one(
            {"id": job_id},
            {"$set": {
                "status": JobStatus.FAILED,
                "error_message": str(e),
                "completed_at": datetime.utcnow()
            }}
        )
        
        # Clean up any local files
        try:
            input_filename = f"input_{job_id}_{job.get('original_filename', 'unknown')}"
            local_input_path = os.path.join(temp_dir, input_filename)
            
            if os.path.exists(local_input_path):
                os.remove(local_input_path)
                
            # Clean up any partial variant files
            for file in os.listdir(temp_dir):
                if file.startswith(f"output_{job_id}_") or file.startswith(f"variant_{job_id}_"):
                    os.remove(os.path.join(temp_dir, file))
        except:
            pass
        
        raise

# Celery periodic tasks (optional)
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    'cleanup-old-files': {
        'task': 'app.services.video_service.cleanup_old_files',
        'schedule': crontab(hour=2, minute=0),  # Run daily at 2 AM
    },
}

@celery_app.task
def cleanup_old_files():
    """Clean up old temporary files"""
    import time
    temp_dir = config('TEMP_DIR', default='/tmp/video-variants')
    
    try:
        current_time = time.time()
        max_age = 24 * 60 * 60  # 24 hours
        
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age:
                    os.remove(file_path)
                    logging.info(f"Removed old file: {filename}")
                    
        logging.info("Cleanup completed successfully")
        
    except Exception as e:
        logging.error(f"Cleanup failed: {e}")

# Health check task
@celery_app.task
def health_check():
    """Health check task for monitoring"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
