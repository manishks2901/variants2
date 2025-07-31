from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from decouple import config

from .routers import video, jobs
from .database import connect_to_mongo, close_mongo_connection
from .models import init_models

# Initialize FastAPI app
app = FastAPI(
    title="Video Transformation API",
    description="Backend system for video processing with job tracking",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
os.makedirs(config("TEMP_DIR", default="/tmp/video-variants"), exist_ok=True)
os.makedirs(config("RESULTS_DIR", default="./results"), exist_ok=True)

# Event handlers
@app.on_event("startup")
async def startup_event():
    await connect_to_mongo()
    await init_models()

@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()

# Include routers
app.include_router(video.router, prefix="/api/v1", tags=["video"])
app.include_router(jobs.router, prefix="/api/v1", tags=["jobs"])

@app.get("/")
async def root():
    return {"message": "Video Transformation API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
