#!/bin/bash

# EC2 Setup Script for Video Transformation API
# Run this script on your fresh EC2 Ubuntu instance

set -e

echo "🚀 Starting EC2 setup for Video Transformation API..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential packages
echo "🔧 Installing essential packages..."
sudo apt install -y \
    curl \
    wget \
    git \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    htop \
    nano \
    vim \
    tree

# Install Docker
echo "🐳 Installing Docker..."
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose (standalone)
echo "📦 Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Nginx
echo "🌐 Installing Nginx..."
sudo apt install -y nginx

# Install Certbot for SSL
echo "🔒 Installing Certbot for SSL..."
sudo apt install -y certbot python3-certbot-nginx

# Create application directory
echo "📁 Creating application directory..."
sudo mkdir -p /opt/video-api
sudo chown $USER:$USER /opt/video-api

# Create required directories
mkdir -p /opt/video-api/{temp,results,logs,backups}

# Install AWS CLI
echo "☁️ Installing AWS CLI..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip

# Enable and start services
echo "🔄 Enabling services..."
sudo systemctl enable docker
sudo systemctl enable nginx
sudo systemctl start docker
sudo systemctl start nginx

# Create swap file (recommended for video processing)
echo "💾 Creating swap file..."
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

echo "✅ EC2 setup completed!"
echo "🔄 Please log out and log back in for Docker group changes to take effect"
echo "🚀 Ready to deploy your application!"
