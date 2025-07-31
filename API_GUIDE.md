# 🚀 Video Transformation API - Complete Setup & Testing Guide

## 📋 **Available API Routes**

### **Base URL**: `http://localhost:8000`

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|------------|
| **GET** | `/` | Root endpoint | None |
| **GET** | `/health` | Health check | None |
| **POST** | `/api/v1/upload` | Upload video & start processing | `file`, `variations`, `priority`, `min_transformations`, `enable_punchlines`, `punchline_variant` |
| **GET** | `/api/v1/jobs/{job_id}` | Get job status (simplified URLs) | Path: `job_id` |
| **GET** | `/api/v1/jobs` | List all jobs | Query: `status`, `limit`, `skip` |
| **GET** | `/api/v1/jobs/{job_id}/progress` | Get detailed job progress | Path: `job_id` |
| **GET** | `/api/v1/download/{job_id}` | Get download URLs (detailed) | Path: `job_id` |
| **POST** | `/api/v1/process/{job_id}` | Manually trigger processing | Path: `job_id` |
| **DELETE** | `/api/v1/cancel/{job_id}` | Cancel job | Path: `job_id` |
| **GET** | `/api/v1/stats` | Processing statistics | None |
| **GET** | `/api/v1/punchline-status` | Check punchline generation availability | None |
| **GET** | `/api/v1/jobs/{job_id}/punchlines` | Get punchline data for job | Path: `job_id` |

---

## 🛠️ **Environment Setup Guide**

### **Option 1: Local Development (Recommended)**

#### **1. Prerequisites**
```bash
# Install system requirements
brew install python3 redis ffmpeg

# Verify installations
python3 --version
redis-server --version
ffmpeg -version
```

#### **2. Project Setup**
```bash
# Navigate to project directory
cd /Users/manishkumarsharma/Documents/variants2

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

#### **3. Environment Configuration**
Create a `.env` file in the project root:
```bash
# MongoDB Configuration
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=video_variants

# AWS S3 Configuration (required for file storage)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=eu-north-1
S3_BUCKET_NAME=your-bucket-name

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Directory Configuration
TEMP_DIR=/tmp/video-variants
RESULTS_DIR=./results

# API Configuration
PORT=8000
```

#### **4. Start Services**

**Method A: Individual Services (Recommended for development)**
```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start FastAPI Server
source venv/bin/activate
python run.py

# Terminal 3: Start Celery Worker
source venv/bin/activate
celery -A app.services.video_service.celery_app worker --loglevel=info --concurrency=2

# Terminal 4: Start Celery Beat (optional, for scheduled tasks)
source venv/bin/activate
celery -A app.services.video_service.celery_app beat --loglevel=info
```

**Method B: All-in-one script**
```bash
chmod +x start_all.sh
./start_all.sh
```

### **Option 2: Docker (Production-like)**

```bash
# Start all services with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## 🧪 **Testing with Postman**

### **1. Setup Postman Environment**

Create a new environment in Postman with these variables:
- `base_url`: `http://localhost:8000`
- `api_base`: `{{base_url}}/api/v1`

### **2. API Testing Collection**

#### **A. Health Check**
```
GET {{base_url}}/health
```
Expected Response:
```json
{
    "status": "healthy"
}
```

#### **B. Upload Video (Main Endpoint)**
```
POST {{api_base}}/upload
Content-Type: multipart/form-data

Body (form-data):
- file: [Select video file - .mp4, .mov, .avi, etc.]
- variations: 3 (number, default: 1)
- priority: normal (text, optional)
- min_transformations: 9 (number, optional)
- enable_punchlines: true (boolean, default: false)
- punchline_variant: 1 (number, 1 or 2, default: 1)
```

**Note:** Punchline generation requires API keys for ElevenLabs and Groq services. Check availability with the `/punchline-status` endpoint first.

Expected Response:
```json
{
    "job_id": "uuid-string",
    "message": "Video uploaded successfully. Processing started.",
    "variants_requested": 3,
    "estimated_time": "3-5 minutes"
}
```

#### **C. Check Job Status (Simplified Format)**
```
GET {{api_base}}/jobs/{{job_id}}
```

**Pending Response:**
```json
{
    "id": "job-uuid",
    "status": "pending",
    "created_at": "2025-07-30T10:00:00",
    "completed_at": null,
    "progress": 0.0
}
```

**Completed Response:**
```json
{
    "id": "job-uuid",
    "status": "completed",
    "created_at": "2025-07-30T10:00:00",
    "completed_at": "2025-07-30T10:05:00",
    "progress": 100.0,
    "variant_0_url": "https://bucket.s3.amazonaws.com/file1.mp4?signed-url",
    "variant_1_url": "https://bucket.s3.amazonaws.com/file2.mp4?signed-url",
    "variant_2_url": "https://bucket.s3.amazonaws.com/file3.mp4?signed-url"
}
```

#### **D. Get Detailed Download Info**
```
GET {{api_base}}/download/{{job_id}}
```

Response:
```json
{
    "job_id": "job-uuid",
    "variants_count": 3,
    "variants": [
        {
            "variant_id": "variant-uuid",
            "download_url": "https://signed-s3-url",
            "filename": "variant_12345678_original.mp4",
            "applied_transformations": ["hue_shift", "micro_crop", "audio_layering"],
            "file_size": 15728640
        }
    ]
}
```

#### **E. List All Jobs**
```
GET {{api_base}}/jobs?status=completed&limit=10&skip=0
```

#### **F. Get Processing Statistics**
```
GET {{api_base}}/stats
```

Response:
```json
{
    "total_jobs": 25,
    "status_counts": {
        "completed": 20,
        "failed": 2,
        "pending": 1,
        "processing": 2
    },
    "completed_jobs": 20,
    "failed_jobs": 2,
    "pending_jobs": 1,
}
```

#### **G. Check Punchline Generation Availability**
```
GET {{api_base}}/punchline-status
```

Response:
```json
{
    "available": true,
    "message": "Punchline generation available"
}
```

#### **H. Get Job Punchline Data**
```
GET {{api_base}}/jobs/{{job_id}}/punchlines
```

Response:
```json
{
    "job_id": "job-uuid",
    "has_punchlines": true,
    "punchline_variants": [
        {
            "variant_id": "variant-uuid",
            "transcript": "Full video transcript from audio...",
            "punchlines": [
                {
                    "text": "Amazing quote here",
                    "suggested_timestamp": "0:05"
                },
                {
                    "text": "Another great moment",
                    "suggested_timestamp": "0:25"
                }
            ],
            "style": {
                "bg_color": "black",
                "text_color": "white",
                "font_size": 50,
                "border_color": "red",
                "border_width": 2,
                "duration": 1.0
            }
        }
    ]
    "processing_jobs": 2,
    "avg_processing_time_seconds": 180.5,
    "total_processing_time_seconds": 3610.0
}
```

### **3. Advanced Testing Scenarios**

#### **Test with Different Variation Counts**
```
# Upload with 1 variation
POST {{api_base}}/upload
Body: file + variations=1

# Upload with 5 variations  
POST {{api_base}}/upload
Body: file + variations=5

# Upload with 10 variations
POST {{api_base}}/upload
Body: file + variations=10
```

#### **Monitor Job Progress**
Set up a recurring request in Postman:
```
GET {{api_base}}/jobs/{{job_id}}
# Run every 3 seconds until status is 'completed' or 'failed'
```

---

## 🎯 **Quick Test Workflow**

### **1. Verify Environment**
```bash
# Check if services are running
curl http://localhost:8000/health

# Check Redis
redis-cli ping

# Check if Celery worker is processing
curl http://localhost:8000/api/v1/stats
```

### **2. Test Upload**
```bash
# Using curl (alternative to Postman)
curl -X POST "http://localhost:8000/api/v1/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@variant2.mp4" \
  -F "variations=2"
```

### **3. Monitor Processing**
```bash
# Replace JOB_ID with actual job ID from upload response
curl http://localhost:8000/api/v1/jobs/JOB_ID
```

### **4. Download Results**
The job status response will contain direct download URLs:
- `variant_0_url`
- `variant_1_url`
- etc.

---

## 🐛 **Troubleshooting**

### **Common Issues**

1. **Port Already in Use**
   ```bash
   # Find and kill processes on port 8000
   lsof -i :8000
   kill -9 PID
   ```

2. **Redis Connection Failed**
   ```bash
   # Start Redis server
   redis-server
   ```

3. **MongoDB Connection Issues**
   ```bash
   # Install and start MongoDB
   brew install mongodb-community
   brew services start mongodb-community
   ```

4. **S3 Upload Failures**
   - Verify AWS credentials in `.env`
   - Ensure bucket exists and has proper permissions
   - Check AWS region configuration

5. **FFmpeg Not Found**
   ```bash
   # Install FFmpeg
   brew install ffmpeg
   ```

---

## 📊 **Expected Processing Times**

| Variations | Approximate Time | Transformations Applied |
|------------|------------------|------------------------|
| 1 variant  | 1-2 minutes     | 9-10 random effects    |
| 3 variants | 3-4 minutes     | 9-10 effects each      |
| 5 variants | 5-7 minutes     | 9-10 effects each      |
| 10 variants| 10-15 minutes   | 9-10 effects each      |

---

## 🎉 **Ready to Test!**

Your API is now ready for testing. The system includes:
- ✅ 22 advanced video transformation metrics
- ✅ Simplified API response format with direct variant URLs
- ✅ Comprehensive job tracking and progress monitoring
- ✅ S3 integration for reliable file storage
- ✅ Background processing with Celery
- ✅ Real-time status updates

**Start with the health check, then try uploading a test video with 2-3 variations to see the system in action!**
