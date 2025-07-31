from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from datetime import datetime
import uuid

from ..models import JobStatusResponse, JobStatus
from ..database import get_database
from ..services.s3_service import S3Service

router = APIRouter()

@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status and progress of a specific job with simplified variant URLs
    """
    try:
        db = get_database()
        if db is None:
            raise HTTPException(status_code=500, detail="Database connection not available")
            
        job = await db.jobs.find_one({"id": job_id})
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Base response structure
        response = {
            "id": job_id,
            "status": job.get("status", "pending"),
            "created_at": job.get("created_at"),
            "completed_at": job.get("completed_at"),
            "progress": job.get("progress", 0.0)
        }
        
        # Add variant URLs for completed jobs
        if job.get("status") == JobStatus.COMPLETED and job.get("variants"):
            s3_service = S3Service()
            
            for i, variant in enumerate(job["variants"]):
                if isinstance(variant, dict) and variant.get("s3_output_key"):
                    try:
                        download_url = await s3_service.generate_presigned_url(variant["s3_output_key"])
                        response[f"variant_{i}_url"] = download_url
                    except Exception as s3_error:
                        # Skip this variant if URL generation fails
                        continue
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@router.get("/jobs")
async def get_jobs(
    status: Optional[JobStatus] = None,
    limit: int = Query(default=10, ge=1, le=100),
    skip: int = Query(default=0, ge=0)
):
    """
    Get list of jobs with optional filtering
    """
    try:
        db = get_database()
        if db is None:
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        # Build query filter
        query = {}
        if status:
            query["status"] = status
        
        # Get jobs with pagination
        cursor = db.jobs.find(query).sort("created_at", -1).skip(skip).limit(limit)
        jobs = await cursor.to_list(length=limit)
        
        results = []
        s3_service = S3Service()
        
        for job in jobs:
            # Generate download URLs for completed variants
            variants_with_urls = []
            if job.get("status") == JobStatus.COMPLETED and job.get("variants"):
                for variant in job["variants"]:
                    # Handle both old and new variant formats
                    if isinstance(variant, dict):
                        # Ensure required fields exist with defaults
                        variant_data = {
                            "variant_id": variant.get("variant_id", str(uuid.uuid4())),
                            "s3_output_key": variant.get("s3_output_key", variant.get("s3_key", "")),
                            "applied_transformations": variant.get("applied_transformations", []),
                            "transformation_count": variant.get("transformation_count", len(variant.get("applied_transformations", []))),
                            "file_size": variant.get("file_size", None),
                            "processing_time": variant.get("processing_time", None),
                            "download_url": None
                        }
                        
                        # Only try to generate download URL if we have a valid s3_output_key
                        if variant_data["s3_output_key"]:
                            try:
                                download_url = await s3_service.generate_presigned_url(variant_data["s3_output_key"])
                                variant_data["download_url"] = download_url
                            except Exception as s3_error:
                                pass  # Keep variant without download URL
                        
                        variants_with_urls.append(variant_data)
            else:
                # Handle non-completed jobs or jobs without variants
                if job.get("variants"):
                    for variant in job["variants"]:
                        if isinstance(variant, dict):
                            variant_data = {
                                "variant_id": variant.get("variant_id", str(uuid.uuid4())),
                                "s3_output_key": variant.get("s3_output_key", variant.get("s3_key", "")),
                                "applied_transformations": variant.get("applied_transformations", []),
                                "transformation_count": variant.get("transformation_count", len(variant.get("applied_transformations", []))),
                                "file_size": variant.get("file_size", None),
                                "processing_time": variant.get("processing_time", None),
                                "download_url": None
                            }
                            variants_with_urls.append(variant_data)
            
            results.append(JobStatusResponse(
                job_id=job["id"],
                status=job.get("status", JobStatus.PENDING),
                progress=job.get("progress", 0.0),
                original_filename=job.get("original_filename", ""),
                variants_requested=job.get("variants_requested", 1),
                variants_completed=job.get("variants_completed", 0),
                variants=variants_with_urls,
                created_at=job.get("created_at", datetime.utcnow()),
                started_at=job.get("started_at"),
                completed_at=job.get("completed_at"),
                error_message=job.get("error_message"),
                duration=job.get("duration")
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get jobs: {str(e)}")

@router.get("/jobs/{job_id}/progress")
async def get_job_progress(job_id: str):
    """
    Get real-time progress of a specific job
    """
    try:
        db = get_database()
        job = await db.jobs.find_one({"id": job_id}, {"progress": 1, "status": 1})
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {
            "job_id": job_id,
            "progress": job["progress"],
            "status": job["status"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job progress: {str(e)}")

@router.get("/stats")
async def get_processing_stats():
    """
    Get overall processing statistics
    """
    try:
        db = get_database()
        
        # Get counts by status
        pipeline = [
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]
        
        status_counts = {}
        async for result in db.jobs.aggregate(pipeline):
            status_counts[result["_id"]] = result["count"]
        
        # Get total processing time for completed jobs
        completed_jobs = await db.jobs.find(
            {"status": JobStatus.COMPLETED, "started_at": {"$exists": True}, "completed_at": {"$exists": True}}
        ).to_list(length=None)
        
        total_processing_time = 0
        avg_processing_time = 0
        
        if completed_jobs:
            processing_times = []
            for job in completed_jobs:
                if job.get("started_at") and job.get("completed_at"):
                    duration = (job["completed_at"] - job["started_at"]).total_seconds()
                    processing_times.append(duration)
            
            if processing_times:
                total_processing_time = sum(processing_times)
                avg_processing_time = total_processing_time / len(processing_times)
        
        return {
            "total_jobs": sum(status_counts.values()),
            "status_counts": status_counts,
            "completed_jobs": status_counts.get(JobStatus.COMPLETED, 0),
            "failed_jobs": status_counts.get(JobStatus.FAILED, 0),
            "pending_jobs": status_counts.get(JobStatus.PENDING, 0),
            "processing_jobs": status_counts.get(JobStatus.PROCESSING, 0),
            "avg_processing_time_seconds": round(avg_processing_time, 2),
            "total_processing_time_seconds": round(total_processing_time, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
