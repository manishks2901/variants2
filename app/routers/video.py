from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
import aiofiles
import os
from typing import Optional
from decouple import config

from ..models import UploadResponse, VideoProcessingJob, JobStatus
from ..database import get_database
from ..services.s3_service import S3Service
from ..services.video_service import VideoProcessingService
from ..services.punchline_service import VideoPunchlineGenerator

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    priority: Optional[str] = Form("normal"),
    variations: Optional[int] = Form(1),
    min_transformations: Optional[int] = Form(9),
    enable_punchlines: Optional[bool] = Form(False),
    punchline_variant: Optional[int] = Form(1)
):
    """
    Upload a video file and start processing job to generate multiple variations
    """
    try:
        # Validate file type - check both content type and file extension
        valid_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv']
        file_extension = os.path.splitext(file.filename)[1].lower() if file.filename else ""
        
        is_valid_file = False
        if file.content_type and file.content_type.startswith('video/'):
            is_valid_file = True
        elif file_extension in valid_extensions:
            is_valid_file = True
        
        if not is_valid_file:
            raise HTTPException(
                status_code=400, 
                detail=f"Only video files are allowed. Received content-type: {file.content_type}, extension: {file_extension}"
            )
        
        # Validate variations count (limit to reasonable number)
        if variations < 1 or variations > 10:
            raise HTTPException(status_code=400, detail="Variations count must be between 1 and 10")
        
        # Create job record
        job = VideoProcessingJob(
            original_filename=file.filename,
            s3_input_key=f"input/{file.filename}",
            priority=priority,
            variants_requested=variations,
            metadata={
                "min_transformations": min_transformations,
                "enable_punchlines": enable_punchlines,
                "punchline_variant": punchline_variant if enable_punchlines else None
            }
        )
        
        # Save job to database
        db = get_database()
        await db.jobs.insert_one(job.dict())
        
        # Upload file to S3
        s3_service = S3Service()
        upload_url = await s3_service.upload_file(file, job.s3_input_key)
        
        # Start processing task
        video_service = VideoProcessingService()
        await video_service.start_processing_task(job.id, variations, min_transformations)
        
        return UploadResponse(
            job_id=job.id,
            upload_url=upload_url,
            message=f"Video uploaded successfully. Processing {variations} variation(s) started."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/process/{job_id}")
async def reprocess_video(
    job_id: str,
    variants_count: Optional[int] = 1,
    min_transformations: Optional[int] = 9
):
    """
    Reprocess an existing video with different parameters
    """
    try:
        db = get_database()
        job = await db.jobs.find_one({"id": job_id})
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job["status"] == JobStatus.PROCESSING:
            raise HTTPException(status_code=400, detail="Job is already processing")
        
        # Validate variants count
        if variants_count < 1 or variants_count > 10:
            raise HTTPException(status_code=400, detail="Variants count must be between 1 and 10")
        
        # Reset job status
        await db.jobs.update_one(
            {"id": job_id},
            {"$set": {
                "status": JobStatus.PENDING,
                "progress": 0.0,
                "error_message": None,
                "variants_requested": variants_count,
                "variants_completed": 0,
                "variants": [],
                "started_at": None,
                "completed_at": None
            }}
        )
        
        # Start processing task
        video_service = VideoProcessingService()
        await video_service.start_processing_task(job_id, variants_count, min_transformations)
        
        return {"message": f"Video reprocessing started for {variants_count} variant(s)", "job_id": job_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reprocessing failed: {str(e)}")

@router.delete("/cancel/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a pending or processing job
    """
    try:
        db = get_database()
        job = await db.jobs.find_one({"id": job_id})
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job["status"] in [JobStatus.COMPLETED, JobStatus.FAILED]:
            raise HTTPException(status_code=400, detail="Cannot cancel completed or failed job")
        
        # Update job status
        await db.jobs.update_one(
            {"id": job_id},
            {"$set": {"status": JobStatus.CANCELLED}}
        )
        
        # TODO: Cancel celery task if running
        
        return {"message": "Job cancelled successfully", "job_id": job_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cancellation failed: {str(e)}")

@router.get("/download/{job_id}")
async def download_processed_video(job_id: str):
    """
    Get download URLs for all processed video variants
    """
    try:
        db = get_database()
        job = await db.jobs.find_one({"id": job_id})
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job["status"] != JobStatus.COMPLETED:
            raise HTTPException(status_code=400, detail="Job is not completed yet")
        
        # Generate presigned URLs for all variants
        s3_service = S3Service()
        download_urls = []
        
        for variant in job.get("variants", []):
            download_url = await s3_service.generate_presigned_url(variant["s3_output_key"])
            download_urls.append({
                "variant_id": variant["variant_id"],
                "download_url": download_url,
                "filename": f"variant_{variant['variant_id'][:8]}_{job['original_filename']}",
                "applied_transformations": variant["applied_transformations"],
                "file_size": variant.get("file_size")
            })
        
        return {
            "job_id": job_id,
            "variants_count": len(download_urls),
            "variants": download_urls
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download URL generation failed: {str(e)}")

@router.get("/punchline-status")
async def get_punchline_status():
    """
    Check if punchline generation is available (API keys configured)
    """
    try:
        punchline_service = VideoPunchlineGenerator()
        return {
            "available": punchline_service.is_available(),
            "message": "Punchline generation available" if punchline_service.is_available() else "API keys not configured"
        }
    except Exception as e:
        return {
            "available": False,
            "message": f"Error checking punchline service: {str(e)}"
        }

@router.get("/jobs/{job_id}/punchlines")
async def get_job_punchlines(job_id: str):
    """
    Get punchline data for a specific job
    """
    try:
        db = get_database()
        job = await db.jobs.find_one({"id": job_id})
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        punchline_variants = []
        for variant in job.get("variants", []):
            punchline_data = variant.get("punchline_data")
            if punchline_data:
                punchline_variants.append({
                    "variant_id": variant["variant_id"],
                    "transcript": punchline_data.get("transcript"),
                    "punchlines": punchline_data.get("punchlines"),
                    "style": punchline_data.get("style")
                })
        
        return {
            "job_id": job_id,
            "has_punchlines": len(punchline_variants) > 0,
            "punchline_variants": punchline_variants
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get punchline data: {str(e)}")
