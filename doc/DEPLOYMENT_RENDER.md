# Deploy to Render.com

## 🚀 Quick Deployment Steps

### 1. Prepare Repository
```bash
# Ensure all files are committed
git add .
git commit -m "Ready for Render deployment"
git push origin main
```

### 2. Deploy on Render

1. **Go to [Render.com](https://render.com)**
2. **Connect GitHub** repository
3. **Create New Web Service**
4. **Configure:**
   - **Name:** `myalgofax-api`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

### 3. Environment Variables
Add these in Render dashboard:
```
HOST=0.0.0.0
DEBUG=False
SERVER_URL=https://your-app-name.onrender.com
```

### 4. Auto-Deploy Settings
- ✅ **Auto-Deploy:** Enabled
- ✅ **Branch:** main
- ✅ **Root Directory:** (leave empty)

## 🤖 ML Model Training

### Automatic Training (Recommended)
Models train automatically on startup using:
- Existing JSON files in repository
- Live NSE data if no files found

### Manual Training
```bash
# Local training before deployment
python train_models.py

# This creates training files:
# - live_active_securities.json
# - live_gainers.json  
# - live_losers.json
# - live_volume_gainers.json
# - live_derivatives.json
```

### Training Data Sources
1. **JSON Response Files:** `response_*.json`
2. **Live API Data:** `live_*.json`
3. **Processed Data:** `processed_*.json`

## 📁 Required Files for Deployment

✅ **Core Files:**
- `main.py` - Main application
- `requirements.txt` - Dependencies
- `render.yaml` - Render configuration

✅ **Training Files:**
- `train_models.py` - ML training script
- `startup.py` - Startup tasks
- JSON data files (optional)

✅ **Configuration:**
- `.env` - Environment variables
- `config.py` - Settings management

## 🔧 Deployment Configuration

### render.yaml
```yaml
services:
  - type: web
    name: myalgofax-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Environment Variables
```env
HOST=0.0.0.0
DEBUG=False
SERVER_URL=https://your-app.onrender.com
```

## 🚦 Post-Deployment

### 1. Test Endpoints
```bash
# Test basic endpoint
curl https://your-app.onrender.com/

# Test AI analysis
curl https://your-app.onrender.com/api/v1/ai/enhanced-market-analysis
```

### 2. Monitor Logs
- Check Render dashboard for deployment logs
- Monitor ML training completion
- Verify WebSocket connections

### 3. Update DNS (Optional)
- Add custom domain in Render
- Update `SERVER_URL` environment variable

## 🔄 Continuous Deployment

### Auto-Deploy Triggers
- ✅ Push to main branch
- ✅ Environment variable changes
- ✅ Manual redeploy

### Update Process
1. Make code changes locally
2. Test with `python main.py`
3. Commit and push to GitHub
4. Render auto-deploys

## 💡 Pro Tips

### Performance
- Use Render's **Starter Plan** for development
- Upgrade to **Standard** for production
- Enable **Auto-Sleep** to save costs

### Monitoring
- Check `/` endpoint for health
- Monitor ML model training logs
- Use Render metrics dashboard

### Troubleshooting
- Check build logs for dependency issues
- Verify environment variables
- Test locally before deploying

## 🎯 Production Checklist

- [ ] All dependencies in requirements.txt
- [ ] Environment variables configured
- [ ] JSON training data included
- [ ] WebSocket endpoints tested
- [ ] API documentation updated
- [ ] Custom domain configured (optional)

## 📞 Support

- **Render Docs:** https://render.com/docs
- **FastAPI Docs:** https://fastapi.tiangolo.com
- **API Status:** Check `/` endpoint

---

**Your API will be live at:** `https://your-app-name.onrender.com`

🎉 **Ready to trade with AI-powered market analysis!**