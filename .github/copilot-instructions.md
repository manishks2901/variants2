<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Video Transformation API - Copilot Instructions

This is a Python FastAPI backend system for video processing with job tracking, MongoDB database, and S3 object storage.

## Project Structure
- `app/main.py` - FastAPI application entry point
- `app/models.py` - Pydantic models and database schemas
- `app/database.py` - MongoDB connection and utilities
- `app/routers/` - API route handlers
- `app/services/` - Business logic services
  - `ffmpeg_service.py` - Video transformation logic (ported from TypeScript)
  - `s3_service.py` - AWS S3 operations
  - `video_service.py` - Celery task orchestration

## Key Technologies
- **FastAPI** for REST API
- **MongoDB** with Motor (async driver) for job tracking
- **AWS S3** for video file storage
- **Celery** with Redis for async job processing
- **FFmpeg** for video transformations
- **Pydantic** for data validation

## Core Features
1. **Video Upload** - Upload videos and get job IDs
2. **Job Tracking** - Real-time progress monitoring
3. **Async Processing** - Background video transformation
4. **S3 Integration** - Reliable file storage
5. **Instagram Bypass** - Specialized transformations for copyright bypass

## Video Transformations
The system applies multiple transformations optimized for Instagram copyright bypass:
- **Guaranteed transformations** (always applied): hue shift, micro crop/pan, temporal shift, rotation, LUT filters
- **Audio transformations**: fingerprint breaking, EQ adjustments  
- **Optional enhancements**: text overlays, noise, vignette effects

## API Endpoints
- `POST /api/v1/upload` - Upload video and start processing
- `GET /api/v1/jobs/{job_id}` - Get job status and progress
- `GET /api/v1/jobs` - List all jobs with filtering
- `GET /api/v1/download/{job_id}` - Get download URL for processed video
- `GET /api/v1/stats` - Processing statistics

## Environment Variables
- MongoDB connection, AWS credentials, Redis URL, temp directories
- See `.env` file for complete configuration

## Development Guidelines
- Use async/await for all database and I/O operations
- Implement proper error handling and logging
- Follow FastAPI best practices for route handlers
- Use Pydantic models for request/response validation
- Implement progress tracking for long-running operations
