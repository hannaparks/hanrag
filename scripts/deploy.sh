#!/bin/bash

# HanRAG VPS Deployment Script
<<<<<<< HEAD
# This script deploys the HanRAG system to a production VPS

set -e  # Exit on any error

# Configuration
VPS_IP="168.231.68.82"  # Update with your VPS IP
VPS_USER="root"
APP_DIR="/root/newmmrag"
SERVICE_NAME="rag-system"
PYTHON_VERSION="python3"
=======
# This script automates the deployment of HanRAG to a VPS

set -e  # Exit on error

# Configuration
VPS_IP="${VPS_IP_OVERRIDE:-168.231.68.82}"
VPS_USER="root"
REMOTE_DIR="/root/newmmrag"
LOCAL_ENV_FILE=".env.production"
>>>>>>> 66c74c8

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
<<<<<<< HEAD
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if we're running on local machine or VPS
is_local() {
    if [[ "$HOSTNAME" == *"local"* ]] || [[ "$HOSTNAME" == *"MacBook"* ]] || [[ "$PWD" == *"/Users/"* ]]; then
        return 0
    else
        return 1
    fi
}

# Function to run commands on VPS via SSH
run_on_vps() {
    local cmd="$1"
    if is_local; then
        ssh -o StrictHostKeyChecking=no "$VPS_USER@$VPS_IP" "$cmd"
    else
        eval "$cmd"
    fi
}

# Function to copy files to VPS
copy_to_vps() {
    local local_path="$1"
    local remote_path="$2"
    if is_local; then
        scp -o StrictHostKeyChecking=no -r "$local_path" "$VPS_USER@$VPS_IP:$remote_path"
    else
        cp -r "$local_path" "$remote_path"
    fi
}

# Check prerequisites
check_prerequisites() {
    print_step "Checking prerequisites..."
    
    if is_local; then
        print_step "Running from local machine - will deploy to VPS"
        
        # Check if .env.production exists
        if [[ ! -f ".env.production" ]]; then
            print_error ".env.production file not found!"
            print_warning "Please create .env.production with your production settings."
            print_warning "You can copy from .env.example and update the values."
            exit 1
        fi
        
        # Check SSH connectivity
        if ! ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$VPS_USER@$VPS_IP" "echo 'SSH connection successful'"; then
            print_error "Cannot connect to VPS via SSH!"
            print_warning "Please ensure:"
            print_warning "1. VPS is running and accessible"
            print_warning "2. SSH key is configured or password authentication is enabled"
            print_warning "3. VPS IP ($VPS_IP) is correct"
            exit 1
        fi
        
        print_success "Prerequisites check passed"
    else
        print_step "Running on VPS - proceeding with local installation"
    fi
}

# Install system dependencies
install_system_dependencies() {
    print_step "Installing system dependencies..."
    
    run_on_vps "apt update && apt upgrade -y"
    
    run_on_vps "apt install -y \\
        python3 \\
        python3-pip \\
        python3-venv \\
        nginx \\
        redis-server \\
        supervisor \\
        git \\
        curl \\
        htop \\
        ufw \\
        logrotate"
    
    # Install Docker
    run_on_vps "curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh"
    
    # Start and enable services
    run_on_vps "systemctl enable redis-server && systemctl start redis-server"
    run_on_vps "systemctl enable docker && systemctl start docker"
    
    print_success "System dependencies installed"
}

# Setup application directory
setup_app_directory() {
    print_step "Setting up application directory..."
    
    # Create app directory
    run_on_vps "mkdir -p $APP_DIR"
    run_on_vps "mkdir -p $APP_DIR/data"
    run_on_vps "mkdir -p $APP_DIR/logs"
    run_on_vps "mkdir -p /root/qdrant_storage"
    
    print_success "Application directory created"
}

# Deploy application code
deploy_application() {
    print_step "Deploying application code..."
    
    if is_local; then
        # Copy entire project to VPS (excluding venv and cache)
        print_step "Copying project files to VPS..."
        
        # Create temporary deployment package
        tar --exclude='./venv' \\
            --exclude='./__pycache__' \\
            --exclude='./.*' \\
            --exclude='./scripts' \\
            -czf deploy-package.tar.gz ./src ./requirements*.txt ./README.md ./CLAUDE.md ./SECURITY.md ./docs ./tests ./todo.md ./newmmrag-monitor.service ./grafana-dashboard.json ./prometheus.yml.example
        
        # Copy to VPS
        copy_to_vps "deploy-package.tar.gz" "$APP_DIR/"
        copy_to_vps ".env.production" "$APP_DIR/"
        
        # Extract on VPS
        run_on_vps "cd $APP_DIR && tar -xzf deploy-package.tar.gz && rm deploy-package.tar.gz"
        
        # Clean up local temp file
        rm deploy-package.tar.gz
        
    else
        print_step "Already on VPS - code should be present"
    fi
    
    print_success "Application code deployed"
}

# Setup Python environment
setup_python_environment() {
    print_step "Setting up Python environment..."
    
    # Create virtual environment
    run_on_vps "cd $APP_DIR && $PYTHON_VERSION -m venv venv"
    
    # Upgrade pip and install dependencies
    run_on_vps "cd $APP_DIR && source venv/bin/activate && pip install --upgrade pip"
    run_on_vps "cd $APP_DIR && source venv/bin/activate && pip install -r requirements.txt"
    
    # Install additional dependencies if hybrid requirements exist
    run_on_vps "cd $APP_DIR && if [ -f requirements_hybrid.txt ]; then source venv/bin/activate && pip install -r requirements_hybrid.txt; fi"
    
    print_success "Python environment setup complete"
}

# Setup Docker containers
setup_docker_containers() {
    print_step "Setting up Docker containers..."
    
    # Start Qdrant vector database
    run_on_vps "docker stop qdrant 2>/dev/null || true"
    run_on_vps "docker rm qdrant 2>/dev/null || true"
    
    run_on_vps "docker run -d \\
        --name qdrant \\
        --restart always \\
        -p 6333:6333 \\
        -v /root/qdrant_storage:/qdrant/storage \\
        qdrant/qdrant"
    
    # Wait for Qdrant to start
    print_step "Waiting for Qdrant to start..."
    run_on_vps "sleep 10"
    
    # Test Qdrant connection
    if run_on_vps "curl -f http://localhost:6333/health"; then
        print_success "Qdrant is running and healthy"
    else
        print_warning "Qdrant may not be fully ready yet"
    fi
}

# Create systemd service
create_systemd_service() {
    print_step "Creating systemd service..."
    
    # Create service file
    run_on_vps "cat > /etc/systemd/system/$SERVICE_NAME.service << 'EOF'
[Unit]
Description=HanRAG System for Mattermost
After=network.target docker.service redis.service
Requires=docker.service redis.service
=======
NC='\033[0m' # No Color

echo -e "${GREEN}Starting HanRAG deployment to VPS ${VPS_IP}...${NC}"

# Check prerequisites
echo -e "\n${YELLOW}Checking prerequisites...${NC}"

if [ ! -f "$LOCAL_ENV_FILE" ]; then
    echo -e "${RED}Error: $LOCAL_ENV_FILE not found!${NC}"
    echo "Please create it from .env.production.example"
    exit 1
fi

# Generate secure API key if not set
if grep -q "your_secure_api_key_here" "$LOCAL_ENV_FILE"; then
    echo -e "${YELLOW}Generating secure API key...${NC}"
    API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    sed -i.bak "s/your_secure_api_key_here/$API_KEY/g" "$LOCAL_ENV_FILE"
    echo -e "${GREEN}API key generated and saved${NC}"
fi

# Create deployment package
echo -e "\n${YELLOW}Creating deployment package...${NC}"
tar -czf deploy.tar.gz \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='venv' \
    --exclude='data/*' \
    --exclude='logs/*' \
    --exclude='.env' \
    --exclude='deploy.tar.gz' \
    .

echo -e "${GREEN}Deployment package created${NC}"

# Deploy to VPS
echo -e "\n${YELLOW}Deploying to VPS...${NC}"

# Upload files
echo "Uploading deployment package..."
scp deploy.tar.gz ${VPS_USER}@${VPS_IP}:~/

echo "Uploading environment file..."
scp ${LOCAL_ENV_FILE} ${VPS_USER}@${VPS_IP}:~/.env.production

# Execute deployment on VPS
echo -e "\n${YELLOW}Executing deployment on VPS...${NC}"

ssh ${VPS_USER}@${VPS_IP} << 'ENDSSH'
set -e

echo "Setting up VPS environment..."

# Update system
apt update && apt upgrade -y

# Install required packages
apt install -y python3 python3-pip python3-venv nginx redis-server git curl htop

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
fi

# Start and enable Redis
systemctl enable redis-server
systemctl start redis-server

# Create application directory
mkdir -p /root/newmmrag
cd /root/newmmrag

# Extract deployment package
tar -xzf ~/deploy.tar.gz
rm ~/deploy.tar.gz

# Move environment file
mv ~/.env.production .env.production

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create data directories
mkdir -p data logs

# Start Qdrant in Docker
if docker ps -a | grep -q qdrant; then
    echo "Restarting existing Qdrant container..."
    docker restart qdrant
else
    echo "Starting new Qdrant container..."
    docker run -d -p 6333:6333 \
        --name qdrant \
        --restart always \
        -v /root/qdrant_storage:/qdrant/storage \
        qdrant/qdrant
fi

# Create systemd service
cat > /etc/systemd/system/rag-system.service << 'EOF'
[Unit]
Description=RAG System for Mattermost
After=network.target docker.service redis.service
>>>>>>> 66c74c8

[Service]
Type=simple
User=root
<<<<<<< HEAD
WorkingDirectory=$APP_DIR
Environment=\"PATH=$APP_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin\"
Environment=\"PYTHONPATH=$APP_DIR\"
ExecStart=$APP_DIR/venv/bin/python -m uvicorn src.api.endpoints:app --host 0.0.0.0 --port 8000
=======
WorkingDirectory=/root/newmmrag
Environment="PATH=/root/newmmrag/venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONPATH=/root/newmmrag"
ExecStart=/root/newmmrag/venv/bin/python -m uvicorn src.api.endpoints:app --host 0.0.0.0 --port 8000
>>>>>>> 66c74c8
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
<<<<<<< HEAD
SyslogIdentifier=rag-system

[Install]
WantedBy=multi-user.target
EOF"
    
    # Reload systemd and enable service
    run_on_vps "systemctl daemon-reload"
    run_on_vps "systemctl enable $SERVICE_NAME"
    
    print_success "Systemd service created and enabled"
}

# Configure Nginx
configure_nginx() {
    print_step "Configuring Nginx..."
    
    # Create nginx site configuration
    run_on_vps "cat > /etc/nginx/sites-available/$SERVICE_NAME << 'EOF'
server {
    listen 80;
    server_name $VPS_IP;
    
    client_max_body_size 50M;
    client_timeout 300s;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection \"1; mode=block\";
    add_header Strict-Transport-Security \"max-age=31536000; includeSubDomains\";
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \\$host;
        proxy_set_header X-Real-IP \\$remote_addr;
        proxy_set_header X-Forwarded-For \\$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \\$scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
        proxy_send_timeout 300s;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://localhost:8000/health;
        access_log off;
    }
}
EOF"
    
    # Enable site
    run_on_vps "ln -sf /etc/nginx/sites-available/$SERVICE_NAME /etc/nginx/sites-enabled/"
    run_on_vps "rm -f /etc/nginx/sites-enabled/default"
    
    # Test nginx configuration
    if run_on_vps "nginx -t"; then
        run_on_vps "systemctl reload nginx"
        print_success "Nginx configured and reloaded"
    else
        print_error "Nginx configuration test failed"
        exit 1
    fi
}

# Configure firewall
configure_firewall() {
    print_step "Configuring firewall..."
    
    # Reset firewall and set defaults
    run_on_vps "ufw --force reset"
    run_on_vps "ufw default deny incoming"
    run_on_vps "ufw default allow outgoing"
    
    # Allow essential services
    run_on_vps "ufw allow 22/tcp comment 'SSH'"
    run_on_vps "ufw allow 80/tcp comment 'HTTP'"
    run_on_vps "ufw allow 443/tcp comment 'HTTPS'"
    run_on_vps "ufw allow 8000/tcp comment 'RAG API'"
    
    # Enable firewall
    run_on_vps "ufw --force enable"
    
    print_success "Firewall configured"
}

# Start services
start_services() {
    print_step "Starting services..."
    
    # Start the application
    run_on_vps "systemctl start $SERVICE_NAME"
    
    # Wait a moment for startup
    sleep 5
    
    # Check service status
    if run_on_vps "systemctl is-active --quiet $SERVICE_NAME"; then
        print_success "RAG system service is running"
    else
        print_error "RAG system service failed to start"
        run_on_vps "systemctl status $SERVICE_NAME"
        run_on_vps "journalctl -u $SERVICE_NAME -n 20 --no-pager"
        exit 1
    fi
}

# Test deployment
test_deployment() {
    print_step "Testing deployment..."
    
    # Test health endpoint locally on VPS
    if run_on_vps "curl -f http://localhost:8000/health"; then
        print_success "Local health check passed"
    else
        print_error "Local health check failed"
        exit 1
    fi
    
    # Test health endpoint via nginx
    if run_on_vps "curl -f http://localhost/health"; then
        print_success "Nginx proxy health check passed"
    else
        print_warning "Nginx proxy health check failed - check nginx configuration"
    fi
    
    # Test from external IP (if running locally)
    if is_local; then
        if curl -f "http://$VPS_IP/health" --connect-timeout 10; then
            print_success "External health check passed"
        else
            print_warning "External health check failed - check firewall and networking"
        fi
    fi
}

# Setup monitoring and maintenance
setup_monitoring() {
    print_step "Setting up monitoring and maintenance..."
    
    # Create health check script
    run_on_vps "cat > /root/check-rag-health.sh << 'EOF'
#!/bin/bash
if ! curl -sf http://localhost:8000/health > /dev/null; then
    echo \"\\$(date): RAG System is down! Restarting...\" >> /var/log/rag-health.log
    systemctl restart $SERVICE_NAME
fi
EOF"
    
    run_on_vps "chmod +x /root/check-rag-health.sh"
    
    # Create backup script
    run_on_vps "cat > /root/backup-rag.sh << 'EOF'
#!/bin/bash
BACKUP_DIR=\"/root/backups/rag-\\$(date +%Y%m%d-%H%M)\"
mkdir -p \\$BACKUP_DIR

# Backup Qdrant data
if docker exec qdrant tar -czf /tmp/qdrant-backup.tar.gz /qdrant/storage 2>/dev/null; then
    docker cp qdrant:/tmp/qdrant-backup.tar.gz \\$BACKUP_DIR/
    docker exec qdrant rm /tmp/qdrant-backup.tar.gz
fi

# Backup application data
tar -czf \\$BACKUP_DIR/app-data.tar.gz $APP_DIR/data/ 2>/dev/null || true

# Backup environment
cp $APP_DIR/.env.production \\$BACKUP_DIR/ 2>/dev/null || true

# Clean old backups (keep 7 days)
find /root/backups -name \"rag-*\" -mtime +7 -delete 2>/dev/null || true

echo \"Backup completed: \\$BACKUP_DIR\"
EOF"
    
    run_on_vps "chmod +x /root/backup-rag.sh"
    
    # Setup log rotation
    run_on_vps "cat > /etc/logrotate.d/rag-system << 'EOF'
$APP_DIR/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 root root
    postrotate
        systemctl reload $SERVICE_NAME
    endscript
}
EOF"
    
    print_success "Monitoring and maintenance scripts created"
}

# Display deployment info
show_deployment_info() {
    print_success "🎉 Deployment completed successfully!"
    echo
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}                    DEPLOYMENT SUMMARY                         ${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo
    echo -e "${BLUE}🌐 Application URLs:${NC}"
    echo -e "   Health Check: http://$VPS_IP/health"
    echo -e "   API Base:     http://$VPS_IP/"
    echo -e "   Docs:         http://$VPS_IP/docs"
    echo
    echo -e "${BLUE}🔧 Mattermost Slash Command URLs:${NC}"
    echo -e "   /inject:      http://$VPS_IP/mattermost/inject"
    echo -e "   /ask:         http://$VPS_IP/mattermost/ask"
    echo
    echo -e "${BLUE}⚙️  Service Management:${NC}"
    echo -e "   Status:       sudo systemctl status $SERVICE_NAME"
    echo -e "   Restart:      sudo systemctl restart $SERVICE_NAME"
    echo -e "   Logs:         sudo journalctl -u $SERVICE_NAME -f"
    echo
    echo -e "${BLUE}📊 Monitoring:${NC}"
    echo -e "   Docker:       docker ps"
    echo -e "   Resources:    htop"
    echo -e "   Health:       /root/check-rag-health.sh"
    echo -e "   Backup:       /root/backup-rag.sh"
    echo
    echo -e "${YELLOW}⚠️  Next Steps:${NC}"
    echo -e "   1. Update your Mattermost slash commands to use the URLs above"
    echo -e "   2. Test the API with your API key"
    echo -e "   3. Set up SSL certificate for HTTPS (recommended)"
    echo -e "   4. Configure regular backups and monitoring"
    echo
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
}

# Main deployment flow
main() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    HanRAG VPS Deployment                    ║"
    echo "║                                                              ║"
    echo "║  This script will deploy the HanRAG system to your VPS      ║"
    echo "║  Target: $VPS_USER@$VPS_IP                              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo
    
    # Deployment steps
    check_prerequisites
    install_system_dependencies
    setup_app_directory
    deploy_application
    setup_python_environment
    setup_docker_containers
    create_systemd_service
    configure_nginx
    configure_firewall
    start_services
    test_deployment
    setup_monitoring
    show_deployment_info
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [options]"
        echo
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --local        Force local mode (skip SSH)"
        echo "  --test         Run deployment test only"
        echo
        echo "Environment variables:"
        echo "  VPS_IP         Override VPS IP address (default: $VPS_IP)"
        echo "  VPS_USER       Override VPS username (default: $VPS_USER)"
        echo
        exit 0
        ;;
    --local)
        export FORCE_LOCAL=true
        ;;
    --test)
        test_deployment
        exit 0
        ;;
esac

# Override IP if provided
if [[ -n "${VPS_IP_OVERRIDE:-}" ]]; then
    VPS_IP="$VPS_IP_OVERRIDE"
fi

# Run main deployment
main "$@"
=======

[Install]
WantedBy=multi-user.target
EOF

# Configure Nginx
cat > /etc/nginx/sites-available/rag-system << 'EOF'
server {
    listen 80;
    server_name _;
    
    client_max_body_size 50M;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
EOF

# Enable Nginx site
rm -f /etc/nginx/sites-enabled/default
ln -sf /etc/nginx/sites-available/rag-system /etc/nginx/sites-enabled/
nginx -t
systemctl restart nginx

# Configure firewall
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 8000/tcp
echo "y" | ufw enable

# Create monitoring script
cat > /root/check-rag-health.sh << 'EOF'
#!/bin/bash
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "RAG System is down at $(date)! Restarting..." >> /root/rag-monitor.log
    systemctl restart rag-system
fi
EOF
chmod +x /root/check-rag-health.sh

# Create backup script
cat > /root/backup-rag.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/root/backups/rag-$(date +%Y%m%d-%H%M%S)"
mkdir -p $BACKUP_DIR

# Backup Qdrant data
docker exec qdrant tar -czf /tmp/qdrant-backup.tar.gz /qdrant/storage
docker cp qdrant:/tmp/qdrant-backup.tar.gz $BACKUP_DIR/
docker exec qdrant rm /tmp/qdrant-backup.tar.gz

# Backup application data
tar -czf $BACKUP_DIR/app-data.tar.gz /root/newmmrag/data/

# Backup environment
cp /root/newmmrag/.env.production $BACKUP_DIR/

# Keep only last 7 backups
find /root/backups -type d -name "rag-*" -mtime +7 -exec rm -rf {} +

echo "Backup completed: $BACKUP_DIR"
EOF
chmod +x /root/backup-rag.sh

# Add cron jobs
(crontab -l 2>/dev/null || true; echo "*/5 * * * * /root/check-rag-health.sh") | crontab -
(crontab -l 2>/dev/null || true; echo "0 2 * * * /root/backup-rag.sh") | crontab -

# Enable and start service
systemctl daemon-reload
systemctl enable rag-system
systemctl start rag-system

echo "Deployment complete!"
ENDSSH

# Clean up local files
rm -f deploy.tar.gz

# Test deployment
echo -e "\n${YELLOW}Testing deployment...${NC}"
sleep 5

if curl -s http://${VPS_IP}/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Health check passed${NC}"
else
    echo -e "${RED}✗ Health check failed${NC}"
    echo "Check logs with: ssh ${VPS_USER}@${VPS_IP} 'journalctl -u rag-system -n 50'"
fi

echo -e "\n${GREEN}Deployment completed!${NC}"
echo -e "\nAccess your RAG system at:"
echo -e "  ${YELLOW}http://${VPS_IP}${NC}"
echo -e "\nAPI Documentation:"
echo -e "  ${YELLOW}http://${VPS_IP}/docs${NC}"
echo -e "\nConfiguration UI:"
echo -e "  ${YELLOW}http://${VPS_IP}/configure${NC}"
echo -e "\nMonitoring Dashboard:"
echo -e "  ${YELLOW}http://${VPS_IP}/dashboard${NC}"
echo -e "\n${YELLOW}Update your Mattermost slash commands to:${NC}"
echo -e "  /inject: ${GREEN}http://${VPS_IP}/mattermost/inject${NC}"
echo -e "  /ask: ${GREEN}http://${VPS_IP}/mattermost/ask${NC}"
echo -e "\n${YELLOW}SSH into your VPS:${NC}"
echo -e "  ${GREEN}ssh ${VPS_USER}@${VPS_IP}${NC}"
echo -e "\n${YELLOW}View logs:${NC}"
echo -e "  ${GREEN}ssh ${VPS_USER}@${VPS_IP} 'journalctl -u rag-system -f'${NC}"
>>>>>>> 66c74c8
