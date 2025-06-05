# HanRAG - Mattermost RAG System

A production-ready Retrieval Augmented Generation (RAG) system integrated with Mattermost, providing AI-powered search and question-answering capabilities through slash commands and REST APIs.

## Features

- **Multi-Strategy Retrieval**: Vector, BM25, and hybrid search capabilities
- **Mattermost Integration**: Native slash commands (`/inject`, `/ask`) for seamless workflow
- **Smart Chunking**: Context-aware text chunking preserving conversation threads
- **Production Ready**: Monitoring, analytics, and deployment automation
- **Flexible Storage**: Qdrant vector database with SQLite metadata store
- **Quality Controls**: Built-in content filtering and response validation
- **Security**: API key authentication, rate limiting, and environment-based configuration

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Mattermost    │───▶│   FastAPI       │───▶│   RAG Pipeline  │
│   Slash Cmds    │    │   REST API      │    │   Multi-Modal   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │   Storage       │    │   LLM Providers │
│   & Analytics   │    │   Qdrant +      │    │   Claude +      │
│                 │    │   SQLite        │    │   OpenAI        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### Local Development

1. **Clone and Setup**
   ```bash
   git clone https://github.com/panic80/HanRAG.git
   cd HanRAG
   chmod +x scripts/*.sh
   ./scripts/start_local.sh
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

3. **Test the System**
   ```bash
   python scripts/test_deployment.py local
   ```

### Production VPS Deployment

#### Prerequisites

- Ubuntu/Debian VPS with root access
- Docker and Docker Compose
- Domain name (optional, for HTTPS)

#### 1. Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3 python3-pip python3-venv git docker.io docker-compose nginx

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker
```

#### 2. Deploy Application

```bash
# Clone repository
git clone https://github.com/hannaparks/HanRAG.git
cd HanRAG

# Make scripts executable
chmod +x scripts/*.sh

# Run deployment script
./scripts/deploy.sh
```

#### 3. Configure Environment

Create production environment file:
```bash
sudo nano /opt/rag-system/.env.production
```

Add your configuration:
```env
# API Keys
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
MATTERMOST_TOKEN=your_mattermost_token_here
MATTERMOST_URL=https://your-mattermost-instance.com

# Database
QDRANT_URL=http://localhost:6333
METADATA_DB_PATH=/opt/rag-system/data/metadata.db

# Server
API_HOST=0.0.0.0
API_PORT=8000
ENVIRONMENT=production

# RAG Configuration
DEFAULT_MODEL=claude-3-sonnet-20240229
EMBEDDING_MODEL=text-embedding-ada-002
MAX_CHUNK_SIZE=1000
RETRIEVAL_COUNT=10
```

#### 4. Start Services

```bash
# Start the system service
sudo systemctl start rag-system
sudo systemctl enable rag-system

# Check status
sudo systemctl status rag-system

# View logs
sudo journalctl -u rag-system -f
```

#### 5. Configure Nginx (Optional)

For HTTPS and domain setup:

```bash
sudo nano /etc/nginx/sites-available/rag-system
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/rag-system /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Dependencies

### Python Dependencies

The system requires Python 3.8+ with the following key dependencies:

```txt
# Core Framework
fastapi>=0.104.1
uvicorn[standard]>=0.24.0

# RAG & AI
anthropic>=0.7.8
openai>=1.3.5
qdrant-client>=1.6.9
sentence-transformers>=2.2.2

# Data Processing
pandas>=2.1.3
numpy>=1.24.3
nltk>=3.8.1

# Web & HTTP
httpx>=0.25.2
aiohttp>=3.9.1

# Database
sqlite3 (built-in)
sqlalchemy>=2.0.23

# Utilities
python-dotenv>=1.0.0
pydantic>=2.5.0
```

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    docker.io \
    docker-compose \
    nginx \
    git \
    curl \
    wget

# CentOS/RHEL
sudo yum install -y \
    python3 \
    python3-pip \
    docker \
    docker-compose \
    nginx \
    git \
    curl \
    wget
```

## Installation Methods

### Method 1: Automated Script (Recommended)

```bash
./scripts/start_local.sh    # Local development
./scripts/deploy.sh         # Production deployment
```

### Method 2: Manual Installation

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings

# 5. Start server
python -m uvicorn src.api.endpoints:app --host 0.0.0.0 --port 8000
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | Required |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | Required |
| `MATTERMOST_TOKEN` | Mattermost bot token | Required |
| `MATTERMOST_URL` | Mattermost server URL | Required |
| `QDRANT_URL` | Qdrant database URL | `http://localhost:6333` |
| `API_HOST` | Server bind address | `0.0.0.0` |
| `API_PORT` | Server port | `8000` |
| `DEFAULT_MODEL` | Claude model for generation | `claude-3-sonnet-20240229` |
| `EMBEDDING_MODEL` | Model for embeddings | `text-embedding-ada-002` |

### Mattermost Integration

1. **Create Bot Account**
   - Go to Mattermost → System Console → Integrations → Bot Accounts
   - Create new bot with appropriate permissions

2. **Configure Webhooks**
   - Set webhook URL to: `http://your-server:8000/mattermost/commands`
   - Enable slash commands: `/inject`, `/ask`

3. **Set Bot Token**
   - Add bot token to your environment configuration

## API Endpoints

### Core Endpoints

- `POST /mattermost/commands` - Mattermost slash command handler
- `POST /query` - Direct query endpoint
- `POST /ingest` - Content ingestion endpoint
- `GET /health` - Health check
- `GET /monitoring/dashboard` - Monitoring dashboard

### Usage Examples

```bash
# Health check
curl http://localhost:8000/health

# Direct query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the latest project update?"}'

# Ingest content
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"content": "Project documentation...", "source": "docs"}'
```

## Monitoring & Maintenance

### View Logs
```bash
# Application logs
sudo journalctl -u rag-system -f

# Qdrant logs
docker logs qdrant-container

# Nginx logs (if configured)
sudo tail -f /var/log/nginx/access.log
```

### Performance Monitoring
- Access dashboard at: `http://your-server:8000/monitoring/dashboard`
- Monitor costs and usage analytics
- Track retrieval quality metrics

### Backup & Recovery
```bash
# Backup vector database
docker exec qdrant-container qdrant-backup

# Backup metadata
cp /opt/rag-system/data/metadata.db /backup/location/
```

## Troubleshooting

### Common Issues

1. **Service Won't Start**
   ```bash
   sudo systemctl status rag-system
   sudo journalctl -u rag-system --no-pager
   ```

2. **Qdrant Connection Issues**
   ```bash
   docker ps | grep qdrant
   curl http://localhost:6333/health
   ```

3. **API Key Issues**
   ```bash
   # Check environment file
   sudo cat /opt/rag-system/.env.production
   ```

4. **Permission Issues**
   ```bash
   sudo chown -R rag-user:rag-user /opt/rag-system
   sudo chmod +x /opt/rag-system/scripts/*.sh
   ```

### Performance Tuning

- Adjust `MAX_CHUNK_SIZE` for your content type
- Tune `RETRIEVAL_COUNT` based on response quality needs
- Monitor memory usage and scale Qdrant accordingly
- Use caching for frequently accessed content

## Development

### Running Tests
```bash
pytest tests/test_channel_processing.py -v
```

### Development Server
```bash
python -m uvicorn src.api.endpoints:app --reload --host 0.0.0.0 --port 8000
```

### Contributing
1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- GitHub Issues: https://github.com/panic80/HanRAG/issues
- Documentation: See `CLAUDE.md` for detailed technical information
