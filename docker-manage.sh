#!/bin/bash

# Docker Management Script for Video Transformation API

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to check if .env file exists
check_env_file() {
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            print_warning ".env file not found. Copying from .env.example"
            cp .env.example .env
            print_warning "Please edit .env file with your actual configuration"
        else
            print_error ".env file not found and no .env.example available"
            exit 1
        fi
    fi
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p temp results logs
    chmod 755 temp results logs
}

# Function to build images
build_images() {
    print_status "Building Docker images..."
    docker-compose build --no-cache
}

# Function to start services
start_services() {
    print_status "Starting services..."
    docker-compose up -d
    
    print_status "Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    print_status "Checking service health..."
    docker-compose ps
}

# Function to start development environment
start_dev() {
    print_status "Starting development environment..."
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
    
    print_status "Development services started with:"
    echo "  - API: http://localhost:8000"
    echo "  - Flower (Celery monitoring): http://localhost:5555"
    echo "  - MongoDB Express: http://localhost:8081 (admin/pass)"
}

# Function to stop services
stop_services() {
    print_status "Stopping services..."
    docker-compose down
}

# Function to restart services
restart_services() {
    print_status "Restarting services..."
    docker-compose restart
}

# Function to view logs
view_logs() {
    if [ -z "$1" ]; then
        docker-compose logs -f
    else
        docker-compose logs -f "$1"
    fi
}

# Function to clean up
cleanup() {
    print_warning "This will remove all containers, images, and volumes. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_status "Cleaning up..."
        docker-compose down -v --rmi all
        docker system prune -f
        print_status "Cleanup completed"
    else
        print_status "Cleanup cancelled"
    fi
}

# Function to show status
show_status() {
    print_status "Service Status:"
    docker-compose ps
    
    print_status "\nResource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
}

# Function to backup data
backup_data() {
    BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    print_status "Creating backup in $BACKUP_DIR..."
    
    # Backup MongoDB
    docker-compose exec mongodb mongodump --out /tmp/backup
    docker cp $(docker-compose ps -q mongodb):/tmp/backup "$BACKUP_DIR/mongodb"
    
    # Note: Redis backup skipped - using external Redis Cloud instance
    print_status "Note: Redis backup skipped (using external Redis Cloud)"
    
    print_status "Backup completed: $BACKUP_DIR"
}

# Main script logic
case "$1" in
    "build")
        check_docker
        check_env_file
        create_directories
        build_images
        ;;
    "start"|"up")
        check_docker
        check_env_file
        create_directories
        start_services
        ;;
    "dev")
        check_docker
        check_env_file
        create_directories
        start_dev
        ;;
    "stop"|"down")
        stop_services
        ;;
    "restart")
        restart_services
        ;;
    "logs")
        view_logs "$2"
        ;;
    "status")
        show_status
        ;;
    "backup")
        backup_data
        ;;
    "clean")
        cleanup
        ;;
    *)
        echo "Video Transformation API Docker Management"
        echo ""
        echo "Usage: $0 {build|start|dev|stop|restart|logs|status|backup|clean}"
        echo ""
        echo "Commands:"
        echo "  build     - Build Docker images"
        echo "  start     - Start production services"
        echo "  dev       - Start development environment with hot reload"
        echo "  stop      - Stop all services"
        echo "  restart   - Restart all services"
        echo "  logs      - View logs (optionally specify service name)"
        echo "  status    - Show service status and resource usage"
        echo "  backup    - Create backup of data"
        echo "  clean     - Clean up all containers, images, and volumes"
        echo ""
        echo "Examples:"
        echo "  $0 start              # Start production environment"
        echo "  $0 dev                # Start development environment"
        echo "  $0 logs api           # View API logs"
        echo "  $0 logs worker        # View worker logs"
        exit 1
        ;;
esac
