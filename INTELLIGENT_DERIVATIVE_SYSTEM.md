# Intelligent Derivative Recommendation System

## Overview

The Intelligent Derivative Recommendation System replaces the old 3-minute alert system with a comprehensive AI-powered analysis that only sends notifications when finding the best high-confidence trading opportunities.

## Key Features

### 🧠 Intelligent Analysis
- Fetches data from **ALL** derivative-related APIs
- Analyzes option chains for major indices (NIFTY, BANKNIFTY, FINNIFTY)
- Processes equity snapshots and OI spurts data
- Uses advanced AI scoring algorithms

### 🎯 High-Confidence Filtering
- Only sends alerts for opportunities with **75%+ confidence**
- Requires **70+ AI score** minimum
- Compares with previous analysis to avoid duplicate alerts
- Focuses on new or significantly improved opportunities

### ⏰ Smart Timing
- Runs every **15 minutes** during market hours only (9:00 AM - 3:30 PM)
- Skips weekends automatically
- No spam - only quality recommendations

### 📊 Comprehensive Data Sources
1. **Market Snapshot** - Overall derivatives market data
2. **Active Underlyings** - Most active derivative instruments
3. **OI Spurts** - Open interest surge analysis
4. **Option Chains** - Real-time option chain data for major indices
5. **Equity Snapshot** - Derivative equity contracts
6. **Futures Data** - Futures market analysis

## System Components

### 1. IntelligentDerivativeAnalyzer
**File:** `intelligent_derivative_analyzer.py`

Main analysis engine that:
- Fetches comprehensive market data
- Extracts trading opportunities
- Calculates AI scores and confidence levels
- Compares with historical data
- Sends intelligent notifications

### 2. Updated Monitor Scheduler
**File:** `monitor_scheduler.py`

Enhanced monitoring system:
- Runs intelligent analysis every 15 minutes
- Maintains stock alerts every 30 minutes
- System health checks every 5 minutes
- Market hours awareness

### 3. Intelligent WebSocket
**File:** `derivative_websocket.py`

WebSocket integration:
- Triggers analysis on market data updates
- Background analysis thread
- 15-minute intelligent intervals
- Backward compatibility maintained

### 4. API Endpoints
**File:** `main.py`

New endpoints added:
- `/api/v1/ai/intelligent-derivative-analysis` - Manual trigger
- `/api/v1/ai/derivative-opportunities` - View opportunities
- `/api/v1/ai/test-intelligent-analyzer` - Test components

## Configuration

### Minimum Thresholds
```python
min_confidence_threshold = 75.0    # 75% minimum confidence
min_ai_score_threshold = 70.0      # 70+ AI score required
analysis_interval = 900            # 15 minutes between analyses
```

### AI Scoring Algorithm
```python
def calculate_ai_score(volume, price_change, current_price):
    volume_score = min((volume / 1000) * 10, 40)      # Max 40 points
    price_score = min(abs(price_change) * 1.5, 30)    # Max 30 points  
    liquidity_score = min((price * volume) / 100000, 20)  # Max 20 points
    base_score = 10                                    # Base 10 points
    
    return max(30, min(total_score, 95))              # Range: 30-95
```

### Confidence Calculation
```python
def calculate_confidence(ai_score, volume, price_change):
    score_factor = (ai_score / 100) * 40
    volume_factor = min((volume / 1000000) * 30, 30)
    momentum_factor = min(price_change / 10 * 30, 30)
    
    return max(50, min(total_confidence, 100))        # Range: 50-100%
```

## Usage

### 1. Automatic Operation
The system runs automatically when the main server starts:
```bash
python main.py
```

### 2. Standalone Analysis
Run intelligent analysis independently:
```bash
python intelligent_derivative_analyzer.py
```

### 3. Monitor with Enhanced Features
Use the updated monitor:
```bash
python monitor_scheduler.py
```

### 4. Test the System
Verify everything works:
```bash
python test_intelligent_analyzer.py
```

### 5. API Testing
Test via API endpoints:
```bash
# Manual trigger
GET /api/v1/ai/intelligent-derivative-analysis

# View current opportunities
GET /api/v1/ai/derivative-opportunities

# Test components
GET /api/v1/ai/test-intelligent-analyzer
```

## Sample Notification

```
🚀 INTELLIGENT DERIVATIVE ALERTS - 14:30

📊 Found 3 HIGH-CONFIDENCE opportunities:

1. NIFTY Call
   Strike: 25000 | Expiry: 28-Nov-2024
   💰 BUY CALL at ₹145.50
   📈 Change: +25.3% | Vol: 2,450,000
   🎯 Target: ₹189.15 | SL: ₹123.68
   🤖 AI Score: 87 | Confidence: 82%
   💡 Exceptional volume activity, Strong price momentum

2. BANKNIFTY Put
   Strike: 51000 | Expiry: 28-Nov-2024
   💰 BUY PUT at ₹320.75
   📈 Change: +18.7% | Vol: 1,890,000
   🎯 Target: ₹416.98 | SL: ₹272.64
   🤖 AI Score: 84 | Confidence: 79%
   💡 High volume activity, Good price movement

⚡ Only BEST opportunities with 75%+ confidence
🔄 Next analysis in 15 minutes
```

## Benefits Over Previous System

### ❌ Old System Issues
- Sent alerts every 3 minutes (spam)
- Basic analysis with limited data
- No intelligence or filtering
- Sent during market closed hours
- No comparison with previous data

### ✅ New System Advantages
- **Smart timing** - Only during market hours, every 15 minutes
- **Comprehensive analysis** - All derivative APIs analyzed
- **High-confidence filtering** - Only 75%+ confidence opportunities
- **No spam** - Compares with previous analysis
- **Detailed recommendations** - Stop loss, targets, risk-reward ratios
- **AI-powered scoring** - Advanced algorithms for opportunity ranking

## File Structure

```
aiadvanceanalysisserver/
├── intelligent_derivative_analyzer.py    # Main analysis engine
├── monitor_scheduler.py                  # Updated monitor (enhanced)
├── derivative_websocket.py              # Intelligent WebSocket
├── test_intelligent_analyzer.py         # Test suite
├── main.py                              # API endpoints (updated)
├── previous_derivative_analysis.json    # Previous analysis cache
├── derivative_analysis_history.json     # Analysis history
└── INTELLIGENT_DERIVATIVE_SYSTEM.md     # This documentation
```

## Monitoring and Logs

The system provides detailed logging:
- 🔄 Data fetching progress
- 📊 Opportunity extraction results  
- 🎯 High-confidence opportunity detection
- ✅ Notification sending status
- ⏰ Market hours and timing checks
- 💾 Analysis saving and history

## Troubleshooting

### Common Issues

1. **No notifications sent**
   - Check if market is open (9:00 AM - 3:30 PM, weekdays)
   - Verify opportunities meet 75% confidence threshold
   - Ensure Telegram bot token is configured

2. **Analysis not running**
   - Check 15-minute interval timing
   - Verify market hours
   - Check for API connectivity issues

3. **Low opportunity count**
   - Market may be stable (normal behavior)
   - Thresholds are intentionally high for quality
   - Check individual API responses

### Testing Commands

```bash
# Test all components
python test_intelligent_analyzer.py

# Test via API
curl http://localhost:8000/api/v1/ai/test-intelligent-analyzer

# Manual analysis trigger
curl http://localhost:8000/api/v1/ai/intelligent-derivative-analysis

# View current opportunities
curl http://localhost:8000/api/v1/ai/derivative-opportunities
```

## Future Enhancements

- Machine learning model training on historical success rates
- Sector-wise derivative analysis
- Options Greeks integration
- Risk management alerts
- Portfolio-based recommendations
- Mobile app notifications

---

**Note:** This system is designed to send fewer but higher-quality alerts. If you're not receiving notifications, it means the market doesn't currently have opportunities meeting our strict quality criteria - which is exactly the intended behavior to avoid spam and focus on the best trades only.