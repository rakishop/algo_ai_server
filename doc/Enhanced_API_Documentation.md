# 🚀 Enhanced AI Trading Analysis API Documentation

## Base URL
```
http://localhost:8000
```

## 🆕 Latest Features (v3.0)
- **Redis Caching**: 5x faster responses with 5-minute cache
- **Real-time WebSocket**: Live market updates
- **Advanced Risk Management**: Position sizing & stop-loss calculations
- **Portfolio Optimization**: AI-powered allocation suggestions
- **Market Regime Detection**: Bull/Bear/Sideways identification
- **Enhanced ML Models**: Ensemble learning with uncertainty quantification
- **Performance Monitoring**: API health & response time metrics
- **Multi-timeframe Analysis**: 4 timeframes consensus
- **Pattern Recognition**: Advanced candlestick analysis
- **Sector Analysis**: Rotation & momentum tracking

---

## 🔥 Enhanced Analysis Endpoints

### 1. Comprehensive Technical Analysis
**POST** `/api/v1/enhanced/comprehensive-analysis`

Complete technical analysis with 15+ indicators and 3 strategies.

**Body:**
```json
{
  "tradingSymbol": "HDFCBANK",
  "chartPeriod": "I",
  "timeInterval": 15
}
```

**Response:**
```json
{
  "symbol": "HDFCBANK-EQ",
  "current_price": 1650.50,
  "price_change_pct": 2.15,
  "technical_indicators": {
    "rsi": 65.4,
    "macd": 0.0234,
    "macd_signal": 0.0198,
    "bollinger_position": 75.2,
    "stochastic_k": 68.5,
    "williams_r": -25.3,
    "atr": 45.2,
    "volatility_pct": 2.8,
    "momentum_score": 15.6
  },
  "strategy_analysis": {
    "momentum": {
      "strategy": "momentum",
      "decision": "BUY",
      "confidence": 82,
      "score": 6,
      "signals": ["RSI bullish momentum", "MACD bullish crossover"],
      "key_levels": {
        "resistance": 1675.80,
        "support": 1620.40
      }
    },
    "mean_reversion": {
      "strategy": "mean_reversion",
      "decision": "HOLD",
      "confidence": 65,
      "score": 1,
      "signals": ["Price near middle band"],
      "bb_position": 55.2,
      "mean_price": 1645.20
    },
    "breakout": {
      "strategy": "breakout",
      "decision": "BUY",
      "confidence": 78,
      "score": 4,
      "signals": ["High volume confirmation: 1.8x"],
      "breakout_levels": {
        "resistance": 1680.00,
        "support": 1610.00
      },
      "volume_ratio": 1.8
    }
  },
  "support_resistance": {
    "pivot_point": 1645.20,
    "resistance_1": 1675.80,
    "support_1": 1620.40,
    "bb_upper": 1685.50,
    "bb_lower": 1605.90
  },
  "risk_management": {
    "risk_score": 25.5,
    "risk_level": "Low",
    "position_size": "Medium (2-3%)",
    "stop_loss": 1567.98,
    "take_profit": 1782.54
  }
}
```

### 2. Multi-Timeframe Analysis
**POST** `/api/v1/enhanced/multi-timeframe`

Analysis across 4 timeframes with consensus decision.

**Body:**
```json
{
  "tradingSymbol": "RELIANCE"
}
```

**Response:**
```json
{
  "symbol": "RELIANCE-EQ",
  "multi_timeframe_analysis": {
    "5min": {
      "momentum": {"decision": "BUY", "confidence": 75},
      "mean_reversion": {"decision": "HOLD", "confidence": 60},
      "breakout": {"decision": "BUY", "confidence": 80}
    },
    "15min": {
      "momentum": {"decision": "BUY", "confidence": 82},
      "mean_reversion": {"decision": "BUY", "confidence": 70},
      "breakout": {"decision": "HOLD", "confidence": 55}
    },
    "1hour": {
      "momentum": {"decision": "BUY", "confidence": 85},
      "mean_reversion": {"decision": "SELL", "confidence": 65},
      "breakout": {"decision": "BUY", "confidence": 78}
    },
    "daily": {
      "momentum": {"decision": "BUY", "confidence": 88},
      "mean_reversion": {"decision": "HOLD", "confidence": 50},
      "breakout": {"decision": "BUY", "confidence": 82}
    }
  },
  "consensus": {
    "consensus_decision": "BUY",
    "consensus_confidence": 78,
    "agreement_percentage": 75.0,
    "decision_breakdown": {
      "BUY": 9,
      "SELL": 1,
      "HOLD": 2
    }
  }
}
```

### 3. Advanced Market Scanner
**GET** `/api/v1/enhanced/market-scanner`

Strategy-specific market scanning with portfolio suggestions.

**Parameters:**
- `strategy`: "momentum", "mean_reversion", "breakout", "all"
- `minConfidence`: 70 (default)
- `maxResults`: 30 (default)
- `chartPeriod`: "I" (default)
- `timeInterval`: 15 (default)

**Example:**
```
GET /api/v1/enhanced/market-scanner?strategy=momentum&minConfidence=75&maxResults=20
```

**Response:**
```json
{
  "strategy_filter": "momentum",
  "total_scanned": 450,
  "signals_found": 25,
  "buy_signals": [
    {
      "symbol": "TCS-EQ",
      "strategy": "momentum",
      "decision": "STRONG_BUY",
      "confidence": 88,
      "score": 6,
      "signals": ["Strong upward momentum: 4.2%", "MACD bullish crossover"],
      "current_price": 3450.75,
      "price_change_pct": 4.2,
      "volume": 2500000,
      "rsi": 68.5,
      "key_levels": {
        "resistance": 3520.00,
        "support": 3380.00
      }
    }
  ],
  "sell_signals": [
    {
      "symbol": "WIPRO-EQ",
      "strategy": "momentum",
      "decision": "SELL",
      "confidence": 76,
      "score": -4,
      "signals": ["Strong downward momentum: -3.8%", "RSI overbought"],
      "current_price": 425.30,
      "price_change_pct": -3.8,
      "volume": 1800000,
      "rsi": 78.2
    }
  ],
  "summary": {
    "strong_buy": 8,
    "buy": 12,
    "strong_sell": 3,
    "sell": 2,
    "avg_confidence": 79.5
  },
  "portfolio_suggestions": {
    "portfolio_suggestions": [
      {
        "symbol": "TCS-EQ",
        "allocation_pct": 15,
        "confidence": 88,
        "entry_price": 3450.75,
        "rationale": "Strong momentum signals"
      },
      {
        "symbol": "INFY-EQ",
        "allocation_pct": 15,
        "confidence": 85,
        "entry_price": 1543.90,
        "rationale": "Breakout above resistance"
      }
    ],
    "cash_allocation": 25,
    "total_stocks": 8,
    "diversification_score": 80
  }
}
```

### 4. Pattern Recognition Analysis
**POST** `/api/v1/enhanced/pattern-recognition`

Advanced candlestick pattern and price action analysis.

**Body:**
```json
{
  "tradingSymbol": "INFY",
  "chartPeriod": "I",
  "timeInterval": 30
}
```

**Response:**
```json
{
  "symbol": "INFY-EQ",
  "current_price": 1543.90,
  "candlestick_patterns": {
    "hammer": true,
    "doji": false,
    "engulfing_bullish": true,
    "morning_star": false,
    "hanging_man": false,
    "engulfing_bearish": false,
    "evening_star": false,
    "shooting_star": false
  },
  "price_action": {
    "body_pct": 65.5,
    "upper_shadow_pct": 15.2,
    "lower_shadow_pct": 19.3,
    "candle_type": "bullish",
    "gap": "up"
  },
  "pattern_signal": "STRONG_BULLISH",
  "confidence": 85,
  "pattern_summary": {
    "bullish_patterns_count": 2,
    "bearish_patterns_count": 0,
    "neutral_patterns": false
  }
}
```

### 5. Sector Analysis
**GET** `/api/v1/enhanced/sector-analysis`

Sector momentum, rotation, and leadership analysis.

**Response:**
```json
{
  "sector_analysis": {
    "IT": {
      "stock_count": 15,
      "avg_price_change": 2.8,
      "positive_stocks_pct": 73.3,
      "total_volume": 45000000,
      "momentum_score": 8.5,
      "sector_signal": "BULLISH",
      "top_performers": [
        {"symbol": "TCS", "price_change": 4.2},
        {"symbol": "INFY", "price_change": 3.8},
        {"symbol": "WIPRO", "price_change": 2.1}
      ]
    },
    "Banking": {
      "stock_count": 12,
      "avg_price_change": 1.5,
      "positive_stocks_pct": 58.3,
      "total_volume": 38000000,
      "momentum_score": 3.2,
      "sector_signal": "NEUTRAL",
      "top_performers": [
        {"symbol": "HDFCBANK", "price_change": 2.1},
        {"symbol": "ICICIBANK", "price_change": 1.8}
      ]
    }
  },
  "market_leaders": ["IT", "Pharma", "Auto"],
  "market_laggards": ["Metals", "Energy"],
  "overall_market_sentiment": {
    "bullish_sectors": 4,
    "bearish_sectors": 2,
    "neutral_sectors": 0
  },
  "market_regime": "Bull Market"
}
```

---

## 🔄 Real-time Features

### WebSocket Live Updates
**WS** `/ws/live-analysis`

Real-time market analysis updates.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/live-analysis');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Live Analysis Update:', data);
    
    // Handle different update types
    if (data.type === 'market_scanner_update') {
        updateMarketScanner(data.signals);
    } else if (data.type === 'sector_rotation') {
        updateSectorAnalysis(data.sectors);
    } else if (data.type === 'pattern_alert') {
        showPatternAlert(data.pattern);
    }
};

ws.onopen = function() {
    // Subscribe to specific updates
    ws.send(JSON.stringify({
        action: 'subscribe',
        channels: ['market_scanner', 'sector_analysis', 'pattern_alerts']
    }));
};
```

**Live Update Types:**
```json
{
  "type": "market_scanner_update",
  "timestamp": "2025-01-21T15:30:00",
  "new_signals": [
    {
      "symbol": "RELIANCE-EQ",
      "decision": "STRONG_BUY",
      "confidence": 92,
      "strategy": "breakout",
      "alert_reason": "Breakout above 20-day high with high volume"
    }
  ]
}
```

### Performance Monitoring
**GET** `/api/v1/system/performance`

API performance and system health metrics.

**Response:**
```json
{
  "api_performance": {
    "/api/v1/enhanced/comprehensive-analysis": {
      "success_rate": 98.5,
      "avg_response_time": 1.24,
      "total_calls": 1250
    },
    "/api/v1/enhanced/market-scanner": {
      "success_rate": 99.2,
      "avg_response_time": 2.15,
      "total_calls": 850
    }
  },
  "system_metrics": {
    "cache_hit_rate": 85.2,
    "active_websocket_connections": 15,
    "redis_status": "Connected",
    "ml_model_status": "Loaded",
    "nse_api_status": "Healthy"
  },
  "performance_summary": {
    "overall_health": "Excellent",
    "avg_response_time": 1.68,
    "uptime_hours": 72.5,
    "total_requests_today": 5420
  }
}
```

---

## 🎯 Core AI Endpoints (Enhanced)

### 1. All Stocks Analysis (Optimized)
**GET** `/api/v1/ai/all-stocks-analysis`

Optimized bulk analysis with Redis caching and smart batching.

**Parameters:**
- `chartPeriod`: "I", "D", "W", "M" (default: "I")
- `timeInterval`: 1,3,5,15,30,60 for "I", 1 for others (default: 15)
- `minConfidence`: 60-100 (default: 60)
- `maxResults`: 10-200 (default: 50)

**Performance Improvements:**
- **Redis Caching**: 5-minute cache for chart data
- **Pandas Filtering**: 10x faster stock filtering
- **Batch Processing**: 20 stocks per batch with 10 workers
- **Smart Pre-filtering**: Only stocks with ≥0.5% price change

**Response:**
```json
{
  "total_stocks_available": 3130,
  "filtered_stocks": 650,
  "total_analyzed": 50,
  "total_decisions": 42,
  "chart_period": "I",
  "time_interval": 15,
  "min_confidence": 60,
  "buy_recommendations": [
    {
      "symbol": "ADANIPOWER-EQ",
      "decision": "BUY",
      "confidence": 88,
      "entry_price": 716.10,
      "market_price": 716.10,
      "price_change_pct": 13.42,
      "volume": 863.24,
      "rsi": 65.4,
      "volume_ratio": 1.85,
      "trend_strength": 0.156,
      "volatility": 2.8,
      "trend": "Strong",
      "key_reason": "Strong price movement: 13.4%",
      "ml_confidence": 0.882
    }
  ],
  "summary": {
    "buy_count": 28,
    "sell_count": 14,
    "hold_count": 0,
    "avg_confidence": 74.5,
    "high_confidence_count": 18
  },
  "performance": {
    "cache_enabled": true,
    "parallel_processing": true,
    "batch_processing": true,
    "price_change_filter": ">=0.5%",
    "batch_size": 20,
    "max_workers": 10
  }
}
```

### 2. Single Stock Decision (Enhanced)
**POST** `/api/v1/ai/trade-decision`

Enhanced ML-powered decision with risk management.

**Body:**
```json
{
  "tradingSymbol": "HDFCBANK",
  "chartPeriod": "I",
  "timeInterval": 15,
  "lookbackPeriod": 50,
  "analysisDepth": "comprehensive"
}
```

**Response:**
```json
{
  "symbol": "HDFCBANK-EQ",
  "decision": "BUY",
  "confidence": 78,
  "ml_score": 0.685,
  "current_price": 1650.50,
  "price_change_pct": 2.15,
  "technical_indicators": {
    "sma_5": 1645.20,
    "sma_20": 1620.80,
    "rsi": 65.4,
    "volatility_pct": 1.8
  },
  "volume_analysis": {
    "current_volume": 2500000,
    "avg_volume": 2100000,
    "volume_ratio": 1.19
  },
  "time_analysis": {
    "short_term_trend": "UP",
    "medium_term_trend": "UP",
    "trend_strength": "Strong",
    "time_recommendation": "Suitable for swing trading"
  },
  "risk_management": {
    "risk_score": 28.5,
    "risk_level": "Low",
    "position_size": "Medium (2-3%)",
    "stop_loss": 1567.98,
    "take_profit": 1782.54,
    "risk_reward_ratio": 2.1
  },
  "reasons": ["Strong upward trend", "High volume activity"],
  "ml_uncertainty": 0.156,
  "ensemble_prediction": {
    "random_forest": 0.72,
    "svm": 0.68,
    "neural_network": 0.71
  }
}
```

---

## 📊 Chart Periods & Intervals

### Intraday (I)
- **1 min**: Ultra-short scalping (Redis cached)
- **3 min**: Short-term scalping
- **5 min**: Scalping/day trading
- **15 min**: Swing trading (default)
- **30 min**: Position trading
- **60 min**: Intraday position

### Daily (D)
- **1 day**: Daily position trading

### Weekly (W)
- **1 week**: Weekly swing trading

### Monthly (M)
- **1 month**: Long-term analysis

---

## 🔧 Performance Optimizations

### Redis Caching
```python
# Automatic caching for 5 minutes
cache_key = f"{symbol}_{chartPeriod}_{timeInterval}"
cached_data = redis_client.get(cache_key)
```

### Smart Batching
```python
# Intelligent batching by volatility
batches = intelligent_batching(stocks, max_batch_size=20)
```

### Async Processing
```python
# Concurrent analysis
results = await async_stock_analysis(symbols, ml_analysis)
```

### Pandas Optimization
```python
# Vectorized filtering (10x faster)
df_filtered = df[
    (df['symbol'].notna()) & 
    (df['pchange'].abs() >= 0.5)
]
```

---

## 🎯 Decision Types & Confidence

### Decision Types
- **STRONG_BUY**: Very high confidence bullish (85-95%)
- **BUY**: High confidence bullish (70-84%)
- **HOLD**: Neutral/unclear signals (50-69%)
- **SELL**: High confidence bearish (70-84%)
- **STRONG_SELL**: Very high confidence bearish (85-95%)

### Confidence Levels
- **90-100**: Extremely High (institutional grade)
- **80-89**: Very High (strong conviction)
- **70-79**: High (good probability)
- **60-69**: Medium (moderate confidence)
- **Below 60**: Low (filtered out)

### Risk Levels
- **Low Risk**: Score 0-30 (Large position 3-5%)
- **Medium Risk**: Score 31-60 (Medium position 2-3%)
- **High Risk**: Score 61-100 (Small position <1%)

---

## 🚀 Usage Examples

### Python Client (Enhanced)
```python
import requests
import websocket
import json

# Enhanced comprehensive analysis
response = requests.post(
    "http://localhost:8000/api/v1/enhanced/comprehensive-analysis",
    json={
        "tradingSymbol": "HDFCBANK",
        "chartPeriod": "I",
        "timeInterval": 15
    }
)
analysis = response.json()
print(f"Decision: {analysis['strategy_analysis']['momentum']['decision']}")
print(f"Risk Level: {analysis['risk_management']['risk_level']}")

# Multi-timeframe consensus
mtf_response = requests.post(
    "http://localhost:8000/api/v1/enhanced/multi-timeframe",
    json={"tradingSymbol": "RELIANCE"}
)
consensus = mtf_response.json()
print(f"Consensus: {consensus['consensus']['consensus_decision']}")
print(f"Agreement: {consensus['consensus']['agreement_percentage']}%")

# Advanced market scanner with portfolio
scanner_response = requests.get(
    "http://localhost:8000/api/v1/enhanced/market-scanner",
    params={
        "strategy": "momentum",
        "minConfidence": 75,
        "maxResults": 20
    }
)
scanner = scanner_response.json()
print(f"Buy Signals: {len(scanner['buy_signals'])}")
print(f"Portfolio Suggestions: {scanner['portfolio_suggestions']}")

# Pattern recognition
pattern_response = requests.post(
    "http://localhost:8000/api/v1/enhanced/pattern-recognition",
    json={
        "tradingSymbol": "INFY",
        "chartPeriod": "I",
        "timeInterval": 30
    }
)
patterns = pattern_response.json()
print(f"Pattern Signal: {patterns['pattern_signal']}")

# Sector analysis
sector_response = requests.get(
    "http://localhost:8000/api/v1/enhanced/sector-analysis"
)
sectors = sector_response.json()
print(f"Market Leaders: {sectors['market_leaders']}")
print(f"Market Regime: {sectors['market_regime']}")

# Performance monitoring
perf_response = requests.get(
    "http://localhost:8000/api/v1/system/performance"
)
performance = perf_response.json()
print(f"System Health: {performance['performance_summary']['overall_health']}")
print(f"Cache Hit Rate: {performance['system_metrics']['cache_hit_rate']}%")
```

### JavaScript Client (WebSocket)
```javascript
// Enhanced WebSocket connection
const ws = new WebSocket('ws://localhost:8000/ws/live-analysis');

ws.onopen = function() {
    console.log('Connected to live analysis feed');
    
    // Subscribe to specific channels
    ws.send(JSON.stringify({
        action: 'subscribe',
        channels: ['market_scanner', 'sector_analysis', 'pattern_alerts', 'risk_alerts']
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'market_scanner_update':
            handleNewSignals(data.new_signals);
            break;
        case 'sector_rotation':
            updateSectorDashboard(data.sectors);
            break;
        case 'pattern_alert':
            showPatternNotification(data.pattern);
            break;
        case 'risk_alert':
            handleRiskAlert(data.risk_event);
            break;
    }
};

// Enhanced market scanner
async function getMarketOpportunities() {
    const response = await fetch(
        'http://localhost:8000/api/v1/enhanced/market-scanner?strategy=all&minConfidence=80'
    );
    const data = await response.json();
    
    // Display buy signals
    data.buy_signals.forEach(signal => {
        console.log(`${signal.symbol}: ${signal.decision} (${signal.confidence}%)`);
        console.log(`Strategy: ${signal.strategy}, Reason: ${signal.signals[0]}`);
    });
    
    // Show portfolio suggestions
    if (data.portfolio_suggestions) {
        console.log('Suggested Portfolio:');
        data.portfolio_suggestions.portfolio_suggestions.forEach(stock => {
            console.log(`${stock.symbol}: ${stock.allocation_pct}% allocation`);
        });
    }
}

// Multi-timeframe analysis
async function getConsensusAnalysis(symbol) {
    const response = await fetch(
        'http://localhost:8000/api/v1/enhanced/multi-timeframe',
        {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({tradingSymbol: symbol})
        }
    );
    
    const data = await response.json();
    console.log(`Consensus for ${symbol}: ${data.consensus.consensus_decision}`);
    console.log(`Agreement: ${data.consensus.agreement_percentage}%`);
    
    return data.consensus;
}

// Real-time performance monitoring
setInterval(async () => {
    const response = await fetch('http://localhost:8000/api/v1/system/performance');
    const perf = await response.json();
    
    document.getElementById('system-health').textContent = 
        perf.performance_summary.overall_health;
    document.getElementById('cache-hit-rate').textContent = 
        perf.system_metrics.cache_hit_rate + '%';
    document.getElementById('active-connections').textContent = 
        perf.system_metrics.active_websocket_connections;
}, 30000); // Update every 30 seconds
```

---

## 🔒 System Requirements

### Performance Specifications
- **Response Time**: <2 seconds (with Redis cache <0.5s)
- **Concurrent Users**: 100+ simultaneous connections
- **Throughput**: 1000+ requests/minute
- **Cache Hit Rate**: 85%+ for optimal performance
- **WebSocket Connections**: 50+ concurrent live feeds

### Dependencies
```bash
pip install fastapi uvicorn pandas numpy scikit-learn redis websockets talib
```

### Redis Setup (Optional but Recommended)
```bash
# Install Redis
sudo apt-get install redis-server

# Start Redis
redis-server

# Test connection
redis-cli ping
```

---

## 📈 API Versioning & Updates

- **Current Version**: v3.0 (Enhanced)
- **Previous Version**: v2.0 (AI-Powered)
- **Legacy Version**: v1.0 (Basic)

### Backward Compatibility
All v2.0 endpoints remain functional with v3.0 enhancements.

### Future Roadmap
- **v3.1**: Options analysis integration
- **v3.2**: Crypto market support
- **v3.3**: Advanced portfolio backtesting
- **v4.0**: Real-time order execution integration

---

## 🆘 Support & Troubleshooting

### Common Issues
1. **Slow Response**: Enable Redis caching
2. **WebSocket Disconnects**: Check network stability
3. **Low Confidence Scores**: Adjust minConfidence parameter
4. **Cache Misses**: Verify Redis connection

### Performance Tuning
```python
# Optimize for your use case
params = {
    "maxResults": 30,        # Reduce for faster response
    "minConfidence": 70,     # Higher for quality signals
    "timeInterval": 15,      # Balance between speed and accuracy
    "strategy": "momentum"   # Focus on specific strategy
}
```

### Contact
- **API Issues**: Technical support team
- **Feature Requests**: Product development team
- **Performance Issues**: Infrastructure team

---

**🎯 The Enhanced AI Trading Analysis API provides institutional-grade technical analysis with real-time performance, advanced risk management, and comprehensive market insights for professional trading applications.**