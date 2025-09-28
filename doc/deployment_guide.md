# Deployment Guide

## Environment Configuration

### Local Development
```env
HOST=0.0.0.0
PORT=8000
DEBUG=True
SERVER_URL=http://localhost:8000
```

### Production Deployment
```env
HOST=0.0.0.0
PORT=80
DEBUG=False
SERVER_URL=https://api.myalgofax.com
```

### Cloud Deployment (AWS/Azure/GCP)
```env
HOST=0.0.0.0
PORT=8000
DEBUG=False
SERVER_URL=https://your-domain.com
```

## Deployment Steps

1. **Copy environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Update .env for your environment:**
   - Change `SERVER_URL` to your domain
   - Set `DEBUG=False` for production
   - Adjust `PORT` if needed

3. **Deploy:**
   ```bash
   # All code automatically uses settings from .env
   python main.py
   ```

## No Manual Updates Needed!

✅ **What happens automatically:**
- All API endpoints use `settings.base_url`
- WebSocket connections use environment config
- Test scripts use environment config
- Documentation examples use environment config

❌ **What you DON'T need to do:**
- Update hardcoded URLs in code
- Modify multiple files
- Change API endpoints manually

## Environment Variables Priority

1. **Environment variables** (highest priority)
2. **.env file** 
3. **Default values** (lowest priority)

This means you can override any setting without changing code!