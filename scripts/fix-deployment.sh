#!/bin/bash

# HanRAG Deployment Fix Script
# This script fixes common deployment issues

set -e

VPS_IP="168.231.68.82"
VPS_USER="root"
APP_DIR="/root/newmmrag"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_step() {
    echo -e "${BLUE}[FIX]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to run commands on VPS
run_on_vps() {
    local cmd="$1"
    ssh -o StrictHostKeyChecking=no "$VPS_USER@$VPS_IP" "$cmd"
}

# Fix Docker installation and start
fix_docker() {
    print_step "Fixing Docker installation and startup..."
    
    # Stop the service first
    run_on_vps "systemctl stop rag-system || true"
    
    # Remove old Docker if exists
    run_on_vps "apt remove -y docker.io docker-compose || true"
    
    # Install Docker properly
    run_on_vps "curl -fsSL https://get.docker.com -o get-docker.sh"
    run_on_vps "sh get-docker.sh"
    
    # Add current user to docker group
    run_on_vps "usermod -aG docker root"
    
    # Start and enable Docker
    run_on_vps "systemctl start docker"
    run_on_vps "systemctl enable docker"
    
    # Wait for Docker to be ready
    print_step "Waiting for Docker to start..."
    run_on_vps "sleep 10"
    
    # Test Docker
    if run_on_vps "docker --version"; then
        print_success "Docker is now working"
    else
        print_error "Docker still not working"
        exit 1
    fi
}

# Fix Qdrant setup
fix_qdrant() {
    print_step "Setting up Qdrant vector database..."
    
    # Stop and remove existing Qdrant container
    run_on_vps "docker stop qdrant 2>/dev/null || true"
    run_on_vps "docker rm qdrant 2>/dev/null || true"
    
    # Create storage directory
    run_on_vps "mkdir -p /root/qdrant_storage"
    
    # Start Qdrant with correct configuration
    run_on_vps "docker run -d \\
        --name qdrant \\
        --restart unless-stopped \\
        -p 6333:6333 \\
        -p 6334:6334 \\
        -v /root/qdrant_storage:/qdrant/storage \\
        qdrant/qdrant"
    
    # Wait for Qdrant to start
    print_step "Waiting for Qdrant to start..."
    run_on_vps "sleep 15"
    
    # Test Qdrant connection
    for i in {1..10}; do
        if run_on_vps "curl -f http://localhost:6333/health 2>/dev/null"; then
            print_success "Qdrant is running and healthy"
            return 0
        fi
        print_step "Waiting for Qdrant... (attempt $i/10)"
        run_on_vps "sleep 5"
    done
    
    print_error "Qdrant failed to start properly"
    run_on_vps "docker logs qdrant"
    exit 1
}

# Fix environment configuration
fix_environment() {
    print_step "Fixing environment configuration..."
    
    # Ensure correct Qdrant port in environment
    run_on_vps "cd $APP_DIR && sed -i 's/QDRANT_PORT=6334/QDRANT_PORT=6333/g' .env.production || true"
    
    # Generate API key if missing
    run_on_vps "cd $APP_DIR && if grep -q 'API_KEY=your_secure_api_key_here' .env.production; then
        NEW_API_KEY=\$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
        sed -i \"s/API_KEY=your_secure_api_key_here/API_KEY=\$NEW_API_KEY/g\" .env.production
        echo \"Generated new API key: \$NEW_API_KEY\"
    fi"
    
    print_success "Environment configuration updated"
}

# Restart services
restart_services() {
    print_step "Restarting services..."
    
    # Restart Redis
    run_on_vps "systemctl restart redis-server"
    
    # Restart RAG system
    run_on_vps "systemctl daemon-reload"
    run_on_vps "systemctl restart rag-system"
    
    # Wait for service to start
    print_step "Waiting for services to start..."
    run_on_vps "sleep 10"
    
    # Check service status
    if run_on_vps "systemctl is-active --quiet rag-system"; then
        print_success "RAG system is running"
    else
        print_error "RAG system failed to start"
        run_on_vps "systemctl status rag-system"
        run_on_vps "journalctl -u rag-system -n 20 --no-pager"
        return 1
    fi
}

# Test the deployment
test_deployment() {
    print_step "Testing deployment..."
    
    # Test health endpoint
    for i in {1..10}; do
        if run_on_vps "curl -f http://localhost:8000/health 2>/dev/null"; then
            print_success "Health check passed"
            break
        fi
        if [ $i -eq 10 ]; then
            print_error "Health check failed after 10 attempts"
            return 1
        fi
        print_step "Waiting for health check... (attempt $i/10)"
        run_on_vps "sleep 5"
    done
    
    # Test external access
    if curl -f "http://$VPS_IP/health" --connect-timeout 10 2>/dev/null; then
        print_success "External access working"
    else
        print_error "External access failed - check firewall"
    fi
}

# Show current status
show_status() {
    echo
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}                      DEPLOYMENT STATUS                        ${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo
    
    echo -e "${BLUE}🐳 Docker Status:${NC}"
    run_on_vps "docker --version" || echo "Docker not working"
    run_on_vps "docker ps | grep qdrant" || echo "Qdrant not running"
    echo
    
    echo -e "${BLUE}⚙️ Service Status:${NC}"
    run_on_vps "systemctl is-active rag-system" || echo "RAG system not running"
    run_on_vps "systemctl is-active redis-server" || echo "Redis not running"
    echo
    
    echo -e "${BLUE}🌐 Health Checks:${NC}"
    if run_on_vps "curl -f http://localhost:6333/health 2>/dev/null"; then
        echo "✅ Qdrant: Healthy"
    else
        echo "❌ Qdrant: Not responding"
    fi
    
    if run_on_vps "curl -f http://localhost:8000/health 2>/dev/null"; then
        echo "✅ RAG API: Healthy"
    else
        echo "❌ RAG API: Not responding"
    fi
    
    if curl -f "http://$VPS_IP/health" --connect-timeout 5 2>/dev/null; then
        echo "✅ External Access: Working"
    else
        echo "❌ External Access: Failed"
    fi
    
    echo
    echo -e "${BLUE}📊 Resource Usage:${NC}"
    run_on_vps "free -h | head -2"
    run_on_vps "df -h | grep -E '/$|/root'"
    echo
    
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
}

# Main fix process
main() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    HanRAG Deployment Fix                    ║"
    echo "║                                                              ║"
    echo "║  This script will fix common deployment issues              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo
    
    # Show current status first
    echo -e "${YELLOW}Current Status:${NC}"
    show_status
    echo
    
    # Fix issues
    fix_docker
    fix_qdrant
    fix_environment
    restart_services
    test_deployment
    
    echo
    echo -e "${GREEN}🎉 Deployment fix completed!${NC}"
    echo
    show_status
    
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "1. Test your Mattermost commands:"
    echo "   /inject --help"
    echo "   /ask test question"
    echo
    echo "2. Monitor logs:"
    echo "   ssh $VPS_USER@$VPS_IP 'journalctl -u rag-system -f'"
    echo
    echo "3. If issues persist, check the logs and restart:"
    echo "   ssh $VPS_USER@$VPS_IP 'systemctl restart rag-system'"
}

# Handle arguments
case "${1:-}" in
    --status)
        show_status
        exit 0
        ;;
    --docker-only)
        fix_docker
        exit 0
        ;;
    --qdrant-only)
        fix_qdrant
        exit 0
        ;;
    --help|-h)
        echo "Usage: $0 [options]"
        echo
        echo "Options:"
        echo "  --status       Show current deployment status"
        echo "  --docker-only  Fix Docker issues only"
        echo "  --qdrant-only  Fix Qdrant issues only"
        echo "  --help, -h     Show this help"
        exit 0
        ;;
esac

# Run main fix
main "$@"