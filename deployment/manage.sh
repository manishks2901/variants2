#!/bin/bash

# Application Management Script
# Provides easy commands to manage your video transformation API

set -e

APP_DIR="/opt/video-api"
COMPOSE_FILE="docker-compose.prod.yml"

show_help() {
    echo "🚀 Video Transformation API Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start       Start all services"
    echo "  stop        Stop all services"
    echo "  restart     Restart all services"
    echo "  status      Show service status"
    echo "  logs        Show application logs"
    echo "  logs-f      Follow application logs"
    echo "  update      Update application from git"
    echo "  backup      Backup database"
    echo "  restore     Restore database from backup"
    echo "  cleanup     Clean up old docker images"
    echo "  monitor     Show system resources"
    echo "  health      Check application health"
    echo "  help        Show this help message"
}

start_services() {
    echo "🚀 Starting all services..."
    cd $APP_DIR
    docker-compose -f $COMPOSE_FILE up -d
    echo "✅ Services started"
}

stop_services() {
    echo "🛑 Stopping all services..."
    cd $APP_DIR
    docker-compose -f $COMPOSE_FILE down
    echo "✅ Services stopped"
}

restart_services() {
    echo "🔄 Restarting all services..."
    cd $APP_DIR
    docker-compose -f $COMPOSE_FILE restart
    echo "✅ Services restarted"
}

show_status() {
    echo "📊 Service Status:"
    cd $APP_DIR
    docker-compose -f $COMPOSE_FILE ps
    echo ""
    echo "🐳 Docker System Info:"
    docker system df
}

show_logs() {
    echo "📋 Application Logs:"
    cd $APP_DIR
    docker-compose -f $COMPOSE_FILE logs --tail=100
}

follow_logs() {
    echo "📋 Following Application Logs (Ctrl+C to exit):"
    cd $APP_DIR
    docker-compose -f $COMPOSE_FILE logs -f
}

update_app() {
    echo "🔄 Updating application..."
    cd $APP_DIR
    
    # Backup current version
    git stash
    
    # Pull latest changes
    git pull origin master
    
    # Rebuild and restart
    docker-compose -f $COMPOSE_FILE build --no-cache
    docker-compose -f $COMPOSE_FILE up -d
    
    echo "✅ Application updated"
}

backup_db() {
    echo "💾 Creating database backup..."
    BACKUP_NAME="backup_$(date +%Y%m%d_%H%M%S)"
    cd $APP_DIR
    
    docker exec video-api-mongodb-prod mongodump \
        --uri="mongodb://admin:$(grep MONGO_ROOT_PASSWORD .env.prod | cut -d'=' -f2)@localhost:27017/video_variants?authSource=admin" \
        --out="/backups/$BACKUP_NAME"
    
    echo "✅ Backup created: $BACKUP_NAME"
}

cleanup_docker() {
    echo "🧹 Cleaning up Docker..."
    docker system prune -f
    docker image prune -a -f
    echo "✅ Docker cleanup completed"
}

monitor_system() {
    echo "📊 System Resources:"
    echo "CPU Usage:"
    top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}'
    echo ""
    echo "Memory Usage:"
    free -h
    echo ""
    echo "Disk Usage:"
    df -h
    echo ""
    echo "Docker Container Stats:"
    docker stats --no-stream
}

check_health() {
    echo "🔍 Checking application health..."
    
    # Check if containers are running
    cd $APP_DIR
    if ! docker-compose -f $COMPOSE_FILE ps | grep -q "Up"; then
        echo "❌ Some containers are not running"
        return 1
    fi
    
    # Check API health endpoint
    if curl -f -s http://localhost:8000/health > /dev/null; then
        echo "✅ API health check passed"
    else
        echo "❌ API health check failed"
        return 1
    fi
    
    # Check database connection
    if docker exec video-api-mongodb-prod mongo --eval "db.stats()" > /dev/null 2>&1; then
        echo "✅ Database connection OK"
    else
        echo "❌ Database connection failed"
        return 1
    fi
    
    echo "✅ All health checks passed"
}

case "$1" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    logs-f)
        follow_logs
        ;;
    update)
        update_app
        ;;
    backup)
        backup_db
        ;;
    cleanup)
        cleanup_docker
        ;;
    monitor)
        monitor_system
        ;;
    health)
        check_health
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo "Use '$0 help' for available commands"
        exit 1
        ;;
esac
