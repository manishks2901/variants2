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
from ..services.punchline_service import EnhancedVideoPunchlineGenerator
from ..services.ffmpeg_service import FFmpegTransformationService

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    priority: Optional[str] = Form("normal"),
    variations: Optional[int] = Form(None),
    variants_count: Optional[int] = Form(None),
    enable_punchlines: Optional[bool] = Form(False),
    punchline_variant: Optional[int] = Form(1),
    strategy: Optional[str] = Form("enhanced_metrics")
):
    """
    Upload a video file and start processing job to generate multiple variations
    
    Parameters:
    - strategy: Transformation strategy to use
        - "standard": 16-24 fully random transformations (balanced quality + variation)
        - "seven_layer": 7-layer pipeline for maximum similarity reduction
    """
    try:
        # Use variants_count if provided, otherwise use variations, default to 1
        final_variants_count = variants_count or variations or 1
        
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
        if final_variants_count < 1 or final_variants_count > 10:
            raise HTTPException(status_code=400, detail="Variations count must be between 1 and 10")
        
        # Validate strategy
        valid_strategies = ["standard", "seven_layer", "enhanced_metrics"]
        if strategy not in valid_strategies:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid strategy. Must be one of: {', '.join(valid_strategies)}"
            )
        
        # Create job record
        job = VideoProcessingJob(
            original_filename=file.filename,
            s3_input_key=f"input/{file.filename}",
            priority=priority,
            variants_requested=final_variants_count,
            metadata={
                "enable_punchlines": enable_punchlines,
                "punchline_variant": punchline_variant if enable_punchlines else None,
                "strategy": strategy
            }
        )
        
        # Save job to database
        db = get_database()
        await db.jobs.insert_one(job.dict())
        
        # Upload file to S3
        s3_service = S3Service()
        upload_url = await s3_service.upload_file(file, job.s3_input_key)
        
        # Start processing task (with chosen strategy)
        video_service = VideoProcessingService()
        await video_service.start_processing_task(job.id, final_variants_count, strategy)
        
        if strategy == "standard":
            strategy_info = "standard random transformations"
        elif strategy == "seven_layer":
            strategy_info = "7-layer pipeline (maximum similarity reduction)"
        elif strategy == "enhanced_metrics":
            strategy_info = "enhanced metrics optimization (targets pHash<20, SSIM<0.20, ORB<3000, Audio<0.25, Metadata<0.30)"
        else:
            strategy_info = strategy
            
        return UploadResponse(
            job_id=job.id,
            upload_url=upload_url,
            message=f"Video uploaded successfully. Processing {final_variants_count} variation(s) with {strategy_info} started."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/process/{job_id}")
async def reprocess_video(
    job_id: str,
    variants_count: Optional[int] = 1,
    strategy: Optional[str] = "enhanced_metrics"
):
    """
    Reprocess an existing video with different parameters
    
    Parameters:
    - strategy: Transformation strategy to use
        - "standard": 16-24 fully random transformations (balanced quality + variation)  
        - "seven_layer": 7-layer pipeline for maximum similarity reduction
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
        
        # Validate strategy
        valid_strategies = ["standard", "seven_layer", "enhanced_metrics"]
        if strategy not in valid_strategies:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid strategy. Must be one of: {', '.join(valid_strategies)}"
            )
        
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
                "completed_at": None,
                "metadata.strategy": strategy  # Update strategy in metadata
            }}
        )
        
        # Start processing task with chosen strategy
        video_service = VideoProcessingService()
        await video_service.start_processing_task(job_id, variants_count, strategy)
        
        strategy_info = "standard random transformations" if strategy == "standard" else "7-layer pipeline (maximum similarity reduction)"
        return {"message": f"Video reprocessing started for {variants_count} variant(s) using {strategy_info}", "job_id": job_id}
        
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
        punchline_service = EnhancedVideoPunchlineGenerator()
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

@router.get("/strategies")
async def get_transformation_strategies():
    """
    Get information about available transformation strategies
    """
    try:
        strategy_info = FFmpegTransformationService.get_transformation_strategy_info()
        return strategy_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get strategy information: {str(e)}")
