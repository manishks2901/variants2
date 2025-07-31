from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"

class PunchlineConfig(BaseModel):
    enabled: bool = False
    variant_style: int = 1  # 1 or 2
    max_punchlines: int = 2
    blackscreen_duration: float = 1.0

class TransformationConfig(BaseModel):
    name: str
    probability: float
    parameters: Optional[Dict[str, Any]] = {}
    punchline_config: Optional[PunchlineConfig] = None

class VideoVariant(BaseModel):
    variant_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    s3_output_key: str
    applied_transformations: List[str] = []
    transformation_count: int = 0
    file_size: Optional[int] = None
    processing_time: Optional[float] = None
    download_url: Optional[str] = None
    punchline_data: Optional[Dict[str, Any]] = None  # Store punchline/transcript info

class VideoProcessingJob(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_filename: str
    s3_input_key: str
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL
    progress: float = 0.0
    error_message: Optional[str] = None
    variants_requested: int = 1
    variants_completed: int = 0
    variants: List[VideoVariant] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration: Optional[float] = None
    video_info: Optional[Dict[str, Any]] = {}
    metadata: Optional[Dict[str, Any]] = {}

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float
    message: str
    created_at: datetime
    estimated_completion: Optional[datetime] = None

class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float
    original_filename: str
    variants_requested: int
    variants_completed: int
    variants: List[VideoVariant]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    duration: Optional[float] = None

class UploadResponse(BaseModel):
    job_id: str
    upload_url: str
    message: str

async def init_models():
    """Initialize database collections and indexes"""
    from .database import get_database
    
    db = get_database()
    if db is None:
        return
    
    # Create indexes for better query performance
    await db.jobs.create_index("id", unique=True)
    await db.jobs.create_index("status")
    await db.jobs.create_index("created_at")
    await db.jobs.create_index("priority")
