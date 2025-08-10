#!/bin/bash

# Pre-deployment Verification Script
# Run this before deploying to check prerequisites

set -e

echo "🔍 Pre-deployment verification for Video Transformation API"
echo "==========================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    if [ $2 -eq 0 ]; then
        echo "✅ $1"
    else
        echo "❌ $1"
    fi
}

# Check AWS credentials
echo "☁️ Checking AWS credentials..."
if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
    print_status "AWS credentials found in environment" 0
elif [ -f ~/.aws/credentials ]; then
    print_status "AWS credentials found in ~/.aws/credentials" 0
else
    print_status "AWS credentials not found - you'll need to configure these" 1
    echo "   💡 Run: aws configure"
fi

# Check if .env.prod exists and has required variables
echo "📝 Checking environment configuration..."
if [ -f ".env.prod" ]; then
    print_status ".env.prod file exists" 0
    
    # Check for required variables
    required_vars=("MONGO_ROOT_PASSWORD" "REDIS_PASSWORD" "AWS_ACCESS_KEY_ID" "S3_BUCKET_NAME")
    for var in "${required_vars[@]}"; do
        if grep -q "^${var}=" .env.prod && ! grep -q "^${var}=.*CHANGE_THIS" .env.prod; then
            print_status "$var is configured" 0
        else
            print_status "$var needs to be configured" 1
        fi
    done
else
    print_status ".env.prod file not found" 1
    echo "   💡 Copy from deployment/.env.prod.template"
fi

# Check S3 bucket accessibility
echo "🪣 Checking S3 bucket access..."
if command_exists aws; then
    BUCKET_NAME=$(grep "^S3_BUCKET_NAME=" .env.prod 2>/dev/null | cut -d'=' -f2 | tr -d '"' || echo "")
    if [ -n "$BUCKET_NAME" ] && [ "$BUCKET_NAME" != "your-video-processing-bucket" ]; then
        if aws s3 ls "s3://$BUCKET_NAME" >/dev/null 2>&1; then
            print_status "S3 bucket '$BUCKET_NAME' is accessible" 0
        else
            print_status "S3 bucket '$BUCKET_NAME' is not accessible" 1
            echo "   💡 Check bucket name and permissions"
        fi
    else
        print_status "S3 bucket name not configured" 1
    fi
else
    print_status "AWS CLI not available for bucket check" 1
fi

# Check Docker and Docker Compose
echo "🐳 Checking Docker setup..."
if command_exists docker; then
    if docker info >/dev/null 2>&1; then
        print_status "Docker is running" 0
    else
        print_status "Docker is installed but not running" 1
    fi
else
    print_status "Docker not installed" 1
fi

if command_exists docker-compose; then
    print_status "Docker Compose is available" 0
else
    print_status "Docker Compose not installed" 1
fi

# Check required files
echo "📁 Checking required files..."
required_files=("docker-compose.prod.yml" "Dockerfile" "requirements.txt" "app/main.py")
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        print_status "$file exists" 0
    else
        print_status "$file missing" 1
    fi
done

# Check disk space
echo "💾 Checking disk space..."
available_space=$(df . | awk 'NR==2 {print $4}')
required_space=5000000  # 5GB in KB
if [ "$available_space" -gt "$required_space" ]; then
    print_status "Sufficient disk space available ($(($available_space/1024/1024))GB)" 0
else
    print_status "Insufficient disk space (need at least 5GB)" 1
fi

# Summary
echo ""
echo "📋 Pre-deployment Summary:"
echo "=========================="
echo "✅ = Ready to proceed"
echo "❌ = Needs attention before deployment"
echo ""
echo "Next steps:"
echo "1. Fix any ❌ issues above"
echo "2. Upload this project to your EC2 instance"
echo "3. Run the deployment scripts"
echo ""
echo "🚀 Happy deploying!"
