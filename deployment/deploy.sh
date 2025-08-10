#!/bin/bash

# Application Deployment Script
# Run this after the EC2 setup is complete

set -e

echo "🚀 Starting application deployment..."

# Clone your repository (replace with your actual repo URL)
cd /opt/video-api
git clone https://github.com/manishks2901/variants2.git .

# Make scripts executable
chmod +x deployment/ec2-setup.sh
chmod +x deployment/deploy.sh
chmod +x start_*.sh
chmod +x docker-manage.sh

echo "📝 Setting up environment files..."

# Copy production environment template
cp .env.prod.example .env.prod

echo "⚠️  IMPORTANT: Please edit .env.prod with your actual values:"
echo "   - MongoDB passwords"
echo "   - Redis passwords"
echo "   - AWS credentials"
echo "   - API keys"
echo ""
echo "📝 Edit the file now with: nano .env.prod"
read -p "Press Enter after you've updated .env.prod..."

# Build and start the application
echo "🐳 Building and starting the application..."
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check if services are running
echo "🔍 Checking service status..."
docker-compose -f docker-compose.prod.yml ps

echo "✅ Application deployment completed!"
echo "🌐 Your API should be accessible at http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000"
