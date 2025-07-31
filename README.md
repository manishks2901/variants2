# Video Transformation API

A Python FastAPI backend system for video processing with job tracking, MongoDB database, and S3 object storage. This system applies multiple video transformations optimized for Instagram copyright bypass.

## Features

- **Asynchronous Video Processing**: Upload videos and track processing progress in real-time
- **Job Management**: Complete job lifecycle tracking with status updates
- **Cloud Storage**: AWS S3 integration for reliable file storage
- **Advanced Video Transformations**: 13+ specialized transformations for copyright bypass
- **REST API**: Comprehensive API with automatic documentation
- **Scalable Architecture**: Celery-based task queue for horizontal scaling

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │   MongoDB       │    │   AWS S3        │
│   (REST API)    │◄──►│   (Job Data)    │    │   (Files)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Celery Queue  │◄──►│   Redis         │    │   FFmpeg        │
│   (Processing)  │    │   (Broker)      │    │   (Transform)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Video Transformations

### Core Transformations (Always Applied)
- **Subtle Hue Shift**: ±15° color adjustment to break pHash fingerprinting
- **Micro Crop & Pan**: 1-3% crop with slight repositioning for SSIM reduction
- **Temporal Shift**: 97-103% speed variation to alter timing fingerprints
- **Audio Fingerprint Break**: Multi-layer audio modification (<60% match)
- **Micro Rotation**: ±1.5° rotation with edge compensation
- **LUT Filter**: Gamma/contrast/brightness adjustments
- **Frame Offset**: Skip 1-2 initial frames to offset temporal patterns
- **Metadata Randomization**: Strip and randomize creation timestamps

### Optional Enhancements (Randomly Applied)
- **Vignette Overlay**: Subtle darkening effect
- **Pixel Noise**: 1-2% intensity noise injection  
- **Audio EQ**: Bass/treble adjustments
- **Text Overlays**: Low-opacity watermarks
- **Trim Microseconds**: Remove 0.1-0.5s from start

## API Endpoints

### Video Processing
- `POST /api/v1/upload` - Upload video and start processing
- `POST /api/v1/process/{job_id}` - Reprocess existing video
- `DELETE /api/v1/cancel/{job_id}` - Cancel processing job
- `GET /api/v1/download/{job_id}` - Get download URL

### Job Management  
- `GET /api/v1/jobs/{job_id}` - Get specific job status
- `GET /api/v1/jobs` - List jobs with filtering
- `GET /api/v1/jobs/{job_id}/progress` - Real-time progress
- `GET /api/v1/stats` - Processing statistics

### System
- `GET /` - API status
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation

## Installation

### Prerequisites
- Python 3.8+
- FFmpeg
- Redis Server
- MongoDB (or MongoDB Atlas)
- AWS Account with S3 access

### Setup Steps

1. **Clone and Install Dependencies**
```bash
git clone <repository-url>
cd variants2
pip install -r requirements.txt
```

2. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your credentials
```

3. **Install System Dependencies**
```bash
# macOS
brew install ffmpeg redis

# Ubuntu/Debian
sudo apt-get install ffmpeg redis-server

# Start Redis
redis-server
```

4. **Setup AWS S3**
- Create S3 bucket
- Configure IAM user with S3 permissions
- Add credentials to `.env`

5. **Setup MongoDB**
- Use MongoDB Atlas or local installation
- Add connection string to `.env`

## Running the Application

### Development Mode

1. **Start API Server**
```bash
python run.py
```

2. **Start Celery Worker** (separate terminal)
```bash
chmod +x start_worker.sh
./start_worker.sh
```

3. **Start Celery Beat** (separate terminal, optional)
```bash
chmod +x start_beat.sh
./start_beat.sh
```

### Production Mode
```bash
chmod +x start_all.sh
./start_all.sh
```

## Usage Examples

### Upload and Process Video
```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@video.mp4" \
  -F "priority=normal" \
  -F "min_transformations=9"
```

### Check Job Status
```bash
curl -X GET "http://localhost:8000/api/v1/jobs/{job_id}" \
  -H "accept: application/json"
```

### Download Processed Video
```bash
curl -X GET "http://localhost:8000/api/v1/download/{job_id}" \
  -H "accept: application/json"
```

## Environment Variables

```bash
# Server Configuration
PORT=8000

# MongoDB
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/video-variants

# AWS S3
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=eu-north-1
S3_BUCKET_NAME=your-bucket-name

# Storage
TEMP_DIR=/tmp/video-variants
RESULTS_DIR=./results

# Redis/Celery
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

## Project Structure

```
variants2/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── database.py          # MongoDB connection
│   ├── models.py            # Pydantic models
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── video.py         # Video upload/processing routes
│   │   └── jobs.py          # Job management routes
│   └── services/
│       ├── __init__.py
│       ├── ffmpeg_service.py    # Video transformation logic
│       ├── s3_service.py        # AWS S3 operations
│       └── video_service.py     # Celery task orchestration
├── .env                     # Environment configuration
├── requirements.txt         # Python dependencies
├── run.py                   # Development server
├── start_worker.sh          # Celery worker startup
├── start_beat.sh           # Celery beat startup
├── start_all.sh            # All services startup
└── README.md               # This file
```

## Monitoring and Debugging

### View API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation.

### Monitor Celery Tasks
```bash
# View active tasks
celery -A app.services.video_service.celery_app inspect active

# View registered tasks
celery -A app.services.video_service.celery_app inspect registered

# Monitor in real-time
celery -A app.services.video_service.celery_app events
```

### Check Logs
- FastAPI logs: Console output from `python run.py`
- Celery logs: Console output from worker/beat processes
- Application logs: Check MongoDB for job error messages

## Performance Considerations

- **Concurrent Processing**: Adjust Celery worker concurrency based on CPU cores
- **Memory Usage**: FFmpeg transformations can be memory-intensive
- **Storage**: Temporary files are cleaned up automatically
- **Timeout**: Tasks timeout after 30 minutes
- **S3 Costs**: Monitor transfer and storage costs

## Scaling

### Horizontal Scaling
- Add more Celery workers on different machines
- Use Redis Cluster for high availability
- Implement MongoDB replica sets

### Optimization
- Pre-warming worker processes
- Batch processing for multiple videos
- Caching frequently used transformations
- CDN integration for download URLs

## Security

- **API Authentication**: Implement JWT or API key authentication
- **File Validation**: Strict video format validation
- **Rate Limiting**: Implement request rate limiting
- **S3 Security**: Use signed URLs with expiration
- **Environment Variables**: Never commit credentials to version control

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Open an issue on GitHub
- Check the API documentation at `/docs`
- Review logs for debugging information
