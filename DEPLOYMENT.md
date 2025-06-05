# Quick Deployment Guide

## 🚀 Deploy to Hostinger VPS (168.231.68.82)

### 1. Prepare Environment File

```bash
# Copy and edit production environment
cp .env.example .env.production
nano .env.production
```

**Required values to update:**
- `ANTHROPIC_API_KEY` - Your Claude API key
- `OPENAI_API_KEY` - Your OpenAI API key  
- `API_KEY` - Generate with: `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- `MATTERMOST_PERSONAL_TOKEN` - Your Mattermost token
- `MATTERMOST_INJECT_TOKEN` - From Mattermost slash command
- `MATTERMOST_ASK_TOKEN` - From Mattermost slash command

### 2. Run Pre-Deployment Check

```bash
./scripts/pre_deployment_check.sh
```

Fix any issues before proceeding.

### 3. Deploy

```bash
./scripts/deploy_production.sh
```

This will:
- ✅ Install all dependencies on VPS
- ✅ Setup Python virtual environment
- ✅ Start Redis and Qdrant
- ✅ Configure systemd service
- ✅ Setup Nginx (if available)
- ✅ Configure health monitoring

### 4. Verify Deployment

```bash
# Test from local machine
python scripts/test_deployment.py production

# Or manually check
curl http://168.231.68.82:8000/health
```

### 5. Update Mattermost

Update your slash commands in Mattermost to:
- `/inject`: `http://168.231.68.82:8000/mattermost/inject`
- `/ask`: `http://168.231.68.82:8000/mattermost/ask`

## 📊 Monitor Your Deployment

### Check Service Status
```bash
ssh root@168.231.68.82 'systemctl status rag-system'
```

### View Logs
```bash
ssh root@168.231.68.82 'journalctl -u rag-system -f'
```

### Check Resource Usage
```bash
ssh root@168.231.68.82 'htop'
```

## 🔧 Common Commands

### Restart Service
```bash
ssh root@168.231.68.82 'systemctl restart rag-system'
```

### Update Code
```bash
# From local machine
./scripts/deploy_production.sh

# Or manually on VPS
ssh root@168.231.68.82
cd /root/newmmrag
git pull
systemctl restart rag-system
```

### Check Circuit Breakers
```bash
curl -H "X-API-Key: your-api-key" \
  http://168.231.68.82:8000/api/circuit-breakers/stats
```

### View Cache Stats
```bash
curl -H "X-API-Key: your-api-key" \
  http://168.231.68.82:8000/api/cache/stats
```

## 🚨 Troubleshooting

### Service Won't Start
```bash
# Check logs
ssh root@168.231.68.82 'journalctl -u rag-system -n 100'

# Test manually
ssh root@168.231.68.82
cd /root/newmmrag
source venv/bin/activate
python -m uvicorn src.api.endpoints:app --host 0.0.0.0 --port 8000
```

### Qdrant Issues
```bash
ssh root@168.231.68.82
docker ps
docker logs qdrant
docker restart qdrant
```

### Redis Issues
```bash
ssh root@168.231.68.82
systemctl status redis-server
redis-cli ping
```

### High Memory Usage
```bash
# Check what's using memory
ssh root@168.231.68.82 'ps aux --sort=-%mem | head -20'

# Restart services
ssh root@168.231.68.82 'systemctl restart rag-system'
```

## 🔒 Security Notes

1. **Change default API key** after deployment
2. **Enable firewall** if not already enabled
3. **Consider HTTPS** with Let's Encrypt for production
4. **Regular backups** of Qdrant data and BM25 index

## 📞 Support

- Check logs first: `journalctl -u rag-system`
- Review docs in `/docs` folder
- Test with `scripts/test_deployment.py`