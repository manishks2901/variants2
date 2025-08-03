# Docker Deployment Guide

This guide explains how to deploy the Video Transformation API using Docker and Docker Compose.

## 📋 Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 4GB RAM
- At least 10GB disk space

## 🚀 Quick Start

### Development Environment

1. **Setup environment file:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Start development environment:**
   ```bash
   ./docker-manage.sh dev
   ```

3. **Access services:**
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Flower (Celery Monitor): http://localhost:5555
   - MongoDB Express: http://localhost:8081 (admin/pass)

### Production Environment

1. **Setup production environment:**
   ```bash
   cp .env.prod.example .env.prod
   # Edit .env.prod with your production configuration
   ```

2. **Start production services:**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

## 🛠️ Management Scripts

Use the provided `docker-manage.sh` script for easy management:

```bash
# Build images
./docker-manage.sh build

# Start development environment
./docker-manage.sh dev

# Start production environment
./docker-manage.sh start

# View logs
./docker-manage.sh logs
./docker-manage.sh logs api     # API logs only
./docker-manage.sh logs worker  # Worker logs only

# Check status
./docker-manage.sh status

# Stop services
./docker-manage.sh stop

# Backup data
./docker-manage.sh backup

# Clean up everything
./docker-manage.sh clean
```

## 🏗️ Architecture

The Docker setup includes:

- **API Container**: FastAPI application
- **Worker Container**: Celery worker for video processing
- **Beat Container**: Celery beat scheduler
- **Redis Container**: Message broker and result backend
- **MongoDB Container**: Database for job tracking
- **Flower Container**: Celery monitoring (development)
- **Nginx Container**: Reverse proxy (production)

## 📁 Directory Structure

```
.
├── docker-compose.yml          # Development configuration
├── docker-compose.dev.yml      # Development overrides
├── docker-compose.prod.yml     # Production configuration
├── Dockerfile                  # Application image
├── nginx.conf                  # Nginx configuration
├── init-mongo.js              # MongoDB initialization
├── docker-manage.sh           # Management script
├── .dockerignore              # Docker ignore patterns
├── .env.example               # Development environment template
├── .env.prod.example          # Production environment template
└── temp/                      # Temporary files
    ├── logs/                  # Application logs
    ├── results/               # Processed videos
    └── backups/               # Data backups
```

## 🔧 Configuration

### Environment Variables

#### Required Variables
```bash
# Database
MONGODB_URL=mongodb://admin:password@mongodb:27017/video_variants?authSource=admin
REDIS_URL=redis://redis:6379/0

# AWS S3
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET_NAME=your_bucket_name

# AI Services
ELEVENLABS_API_KEY=your_api_key
GROQ_API_KEY=your_api_key
```

#### Production Additional Variables
```bash
# Security
MONGO_ROOT_PASSWORD=secure_password
REDIS_PASSWORD=secure_password
SECRET_KEY=your_secret_key

# Performance
WORKERS=4
WORKER_CONCURRENCY=4
```

### Volume Mounts

- `./temp:/tmp/video-variants` - Temporary video processing files
- `./results:/app/results` - Processed video outputs
- `./logs:/app/logs` - Application logs
- `mongodb_data:/data/db` - MongoDB data persistence
- `redis_data:/data` - Redis data persistence

## 🔍 Monitoring

### Health Checks

All services include health checks:
- **API**: `GET /health`
- **Worker**: Celery ping
- **Redis**: Redis ping
- **MongoDB**: Connection check

### Logging

Logs are available via:
```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f api
docker-compose logs -f worker
docker-compose logs -f redis
docker-compose logs -f mongodb
```

### Monitoring Tools

- **Flower**: Celery task monitoring at http://localhost:5555
- **MongoDB Express**: Database browser at http://localhost:8081 (dev only)
- **Docker Stats**: `docker stats` for resource usage

## 🔒 Security

### Production Security Checklist

- [ ] Change default passwords in `.env.prod`
- [ ] Configure SSL certificates for HTTPS
- [ ] Set up firewall rules
- [ ] Enable authentication for monitoring tools
- [ ] Configure backup strategy
- [ ] Set up log rotation
- [ ] Configure resource limits

### Network Security

The setup uses a custom Docker network for service isolation. Services communicate internally using service names.

## 📊 Performance Tuning

### Resource Limits

Production configuration includes resource limits:
- **API**: 1GB RAM, 1 CPU
- **Worker**: 2GB RAM, 2 CPU
- **Redis**: 512MB max memory

### Scaling Workers

To scale workers horizontally:
```bash
docker-compose -f docker-compose.prod.yml up -d --scale worker=4
```

### Database Optimization

MongoDB includes optimized indexes and connection pooling. For high-load scenarios, consider:
- MongoDB replica sets
- Read replicas
- Connection pooling tuning

## 🔄 Backup and Recovery

### Automated Backups

```bash
# Create backup
./docker-manage.sh backup

# Backup contents:
# - MongoDB dump
# - Redis snapshot
# - Application logs
```

### Manual Backup

```bash
# MongoDB
docker-compose exec mongodb mongodump --out /tmp/backup
docker cp $(docker-compose ps -q mongodb):/tmp/backup ./backups/

# Redis
docker-compose exec redis redis-cli BGSAVE
docker cp $(docker-compose ps -q redis):/data/dump.rdb ./backups/
```

### Recovery

```bash
# MongoDB restore
docker cp ./backups/mongodb $(docker-compose ps -q mongodb):/tmp/restore
docker-compose exec mongodb mongorestore /tmp/restore

# Redis restore
docker cp ./backups/dump.rdb $(docker-compose ps -q redis):/data/
docker-compose restart redis
```

## 🐛 Troubleshooting

### Common Issues

1. **Port conflicts**: Change port mappings in docker-compose.yml
2. **Permission denied**: Ensure proper file permissions with `chmod 755`
3. **Out of memory**: Increase Docker memory limits or reduce worker concurrency
4. **Database connection failed**: Check MongoDB credentials and network connectivity

### Debug Mode

Enable debug logging:
```bash
# Set in .env
LOG_LEVEL=DEBUG

# Or use development compose
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Health Check Commands

```bash
# Check API health
curl http://localhost:8000/health

# Check Redis
docker-compose exec redis redis-cli ping

# Check MongoDB
docker-compose exec mongodb mongo --eval "db.adminCommand('ping')"

# Check Celery workers
docker-compose exec api celery -A app.services.video_service.celery_app inspect active
```

## 📈 Scaling

For production scaling consider:

1. **Horizontal scaling**: Multiple worker instances
2. **Load balancing**: Multiple API instances behind load balancer  
3. **Database scaling**: MongoDB sharding or replica sets
4. **Caching**: Redis cluster for high availability
5. **CDN**: Content delivery network for video files

## 🆘 Support

For issues and questions:
1. Check logs: `docker-compose logs`
2. Verify health checks: `./docker-manage.sh status`
3. Check resource usage: `docker stats`
4. Review configuration files
5. Consult application documentation
