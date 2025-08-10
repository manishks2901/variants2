# Complete EC2 Deployment Guide

## 🚀 Quick Deployment Checklist

### Phase 1: AWS Setup
- [ ] Launch EC2 instance (t3.large recommended)
- [ ] Configure Security Groups (SSH, HTTP, HTTPS)
- [ ] Set up Elastic IP (optional but recommended)
- [ ] Point your domain to the EC2 IP

### Phase 2: Server Setup
```bash
# 1. Connect to your EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# 2. Run the setup script
curl -fsSL https://raw.githubusercontent.com/manishks2901/variants2/master/deployment/ec2-setup.sh | bash

# 3. Log out and back in for Docker group changes
exit
ssh -i your-key.pem ubuntu@your-ec2-ip
```

### Phase 3: Application Deployment
```bash
# 1. Run the deployment script
curl -fsSL https://raw.githubusercontent.com/manishks2901/variants2/master/deployment/deploy.sh | bash

# 2. Configure your environment variables in .env.prod
nano /opt/video-api/.env.prod

# Required variables to update:
# - MONGO_ROOT_PASSWORD
# - REDIS_PASSWORD
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - S3_BUCKET_NAME
# - ELEVENLABS_API_KEY (optional)
# - GROQ_API_KEY (optional)
```

### Phase 4: Web Server Setup
```bash
# 1. Configure Nginx
sudo cp /opt/video-api/deployment/nginx-config /etc/nginx/sites-available/video-api
sudo ln -s /etc/nginx/sites-available/video-api /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default

# 2. Update domain in config (replace your-domain.com)
sudo sed -i 's/your-domain.com/yourdomain.com/g' /etc/nginx/sites-available/video-api
sudo nginx -t
sudo systemctl reload nginx
```

### Phase 5: SSL Setup (if you have a domain)
```bash
# Run SSL setup script
/opt/video-api/deployment/setup-ssl.sh yourdomain.com
```

### Phase 6: Application Management
```bash
# Make management script executable
chmod +x /opt/video-api/deployment/manage.sh

# Create symlink for easy access
sudo ln -s /opt/video-api/deployment/manage.sh /usr/local/bin/video-api

# Now you can use: video-api start|stop|status|logs etc.
```

## 📊 Testing Your Deployment

### 1. Health Check
```bash
# Test API directly
curl http://your-ec2-ip:8000/health

# Test through Nginx
curl http://your-domain.com/health
```

### 2. Upload Test
```bash
# Test video upload
curl -X POST \
  http://your-domain.com/api/v1/upload \
  -F "file=@test-video.mp4" \
  -F "options={\"transformations\":[\"hue_shift\",\"micro_crop\"]}"
```

### 3. Job Status Test
```bash
# Check job status (replace JOB_ID)
curl http://your-domain.com/api/v1/jobs/JOB_ID
```

## 🔧 Management Commands

After deployment, use these commands to manage your application:

```bash
# Service management
video-api start      # Start all services
video-api stop       # Stop all services
video-api restart    # Restart all services
video-api status     # Show service status

# Monitoring
video-api logs       # Show recent logs
video-api logs-f     # Follow logs real-time
video-api monitor    # Show system resources
video-api health     # Run health checks

# Maintenance
video-api update     # Update from git
video-api backup     # Backup database
video-api cleanup    # Clean up Docker
```

## 🚨 Troubleshooting

### Common Issues

1. **Services not starting**
   ```bash
   video-api logs
   docker-compose -f /opt/video-api/docker-compose.prod.yml ps
   ```

2. **Out of disk space**
   ```bash
   df -h
   video-api cleanup
   sudo apt autoremove
   ```

3. **High memory usage**
   ```bash
   video-api monitor
   docker stats
   ```

4. **FFmpeg errors**
   ```bash
   docker exec -it video-api-worker-prod bash
   ffmpeg -version
   ```

### Log Locations
- Application logs: `/opt/video-api/logs/`
- Nginx logs: `/var/log/nginx/`
- Docker logs: `docker-compose logs`

## 🔒 Security Best Practices

1. **Keep system updated**
   ```bash
   sudo apt update && sudo apt upgrade
   ```

2. **Monitor logs regularly**
   ```bash
   video-api logs | grep ERROR
   ```

3. **Backup regularly**
   ```bash
   # Set up daily backups
   echo "0 2 * * * /usr/local/bin/video-api backup" | crontab -
   ```

4. **Use strong passwords**
   - Update all default passwords in `.env.prod`
   - Use AWS IAM roles when possible

## 📈 Scaling Considerations

### Vertical Scaling
- Upgrade to c5.xlarge or c5.2xlarge for more CPU
- Add more EBS storage for video processing

### Horizontal Scaling
- Use Application Load Balancer
- Deploy multiple EC2 instances
- Use RDS for MongoDB (Amazon DocumentDB)
- Use ElastiCache for Redis

### Performance Optimization
- Enable CloudFront CDN for video delivery
- Use S3 Transfer Acceleration
- Implement video compression presets
- Monitor with CloudWatch

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review application logs: `video-api logs`
3. Check system resources: `video-api monitor`
4. Verify health status: `video-api health`

## 🎉 Success!

Your Video Transformation API is now deployed and ready to process videos!

Access your API at:
- Direct: `http://your-ec2-ip:8000`
- With domain: `https://your-domain.com`
- API docs: `https://your-domain.com/docs`
