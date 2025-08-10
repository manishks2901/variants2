#!/bin/bash

# SSL Setup Script for Video Transformation API
# Run this after your domain is pointing to your EC2 instance

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <your-domain.com>"
    echo "Example: $0 myvideoapi.com"
    exit 1
fi

DOMAIN=$1
echo "🔒 Setting up SSL for domain: $DOMAIN"

# Update Nginx configuration with actual domain
echo "📝 Updating Nginx configuration..."
sudo sed -i "s/your-domain.com/$DOMAIN/g" /etc/nginx/sites-available/video-api

# Test Nginx configuration
echo "🧪 Testing Nginx configuration..."
sudo nginx -t

# Reload Nginx
echo "🔄 Reloading Nginx..."
sudo systemctl reload nginx

# Obtain SSL certificate
echo "🔒 Obtaining SSL certificate from Let's Encrypt..."
sudo certbot --nginx -d $DOMAIN -d www.$DOMAIN --non-interactive --agree-tos --email admin@$DOMAIN

# Enable HTTPS configuration in Nginx
echo "🔧 Enabling HTTPS configuration..."
sudo sed -i 's/# return 301 https/return 301 https/' /etc/nginx/sites-available/video-api
sudo sed -i 's/^#[[:space:]]*server {/server {/' /etc/nginx/sites-available/video-api
sudo sed -i '/^#[[:space:]]*listen 443/,/^#[[:space:]]*}/ s/^#[[:space:]]*//' /etc/nginx/sites-available/video-api

# Test and reload Nginx again
sudo nginx -t
sudo systemctl reload nginx

# Set up auto-renewal
echo "🔄 Setting up SSL certificate auto-renewal..."
(crontab -l 2>/dev/null; echo "0 12 * * * /usr/bin/certbot renew --quiet") | crontab -

echo "✅ SSL setup completed!"
echo "🌐 Your API is now accessible at https://$DOMAIN"
echo "🔒 SSL certificate will auto-renew every 12 hours"
