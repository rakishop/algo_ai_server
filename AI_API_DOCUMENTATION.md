# AI-Powered NSE Market Data API Documentation

## Base URL
```
http://localhost:8000
```

## New AI & ML Endpoints

### 1. Smart Stock Picks (AI Recommendations)
**GET** `/api/v1/ai/smart-picks`

AI-powered stock recommendations based on risk profile and investment amount.

**Parameters:**
- `risk_level` (string): "low", "medium", "high" (default: "medium")
- `investment_amount` (float): Investment amount in INR (default: 100000)
- `limit` (int): Number of recommendations (1-50, default: 15)

**Response:**
```json
{
  "recommendations": [
    {
      "symbol": "ADANIPOWER",
      "price": 716.1,
      "quantity": 93,
      "investment_amount": 66597.3,
      "weight_percentage": 10.0,
      "expected_return": 13.42,
      "risk_score": 8.5
    }
  ],
  "risk_level": "medium",
  "total_investment": 100000,
  "diversification_count": 15,
  "ai_insights": {...}
}
```

### 2. Market Anomaly Detection
**GET** `/api/v1/ai/anomaly-detection`

Detect unusual market movements using ML algorithms.

**Parameters:**
- `limit` (int): Number of anomalies to return (1-100, default: 20)

**Response:**
```json
{
  "anomalies": [
    {
      "symbol": "INTENTECH",
      "perChange": 20.0,
      "is_anomaly": true,
      "cluster": 2,
      "anomaly_reason": ["Extreme price change: 20.0%"]
    }
  ],
  "total_anomalies": 15,
  "detection_method": "Isolation Forest ML Algorithm"
}
```

### 3. Similar Stock Finder
**GET** `/api/v1/ai/similar-stocks/{symbol}`

Find stocks similar to a given symbol using ML clustering.

**Parameters:**
- `symbol` (path): Stock symbol (e.g., "RELIANCE")
- `limit` (int): Number of similar stocks (1-30, default: 10)

**Response:**
```json
{
  "target_symbol": "RELIANCE",
  "similar_stocks": [
    {
      "symbol": "TCS",
      "cluster": 1,
      "similarity_score": 0.85,
      "perChange": 2.5
    }
  ],
  "similarity_method": "ML Clustering (K-Means)",
  "count": 10
}
```

### 4. Market Sentiment Analysis
**GET** `/api/v1/ai/market-sentiment`

Analyze overall market sentiment using AI algorithms.

**Response:**
```json
{
  "sentiment_score": 25.5,
  "sentiment_label": "Bullish",
  "market_breadth": {
    "advance": {...},
    "decline": {...}
  },
  "ai_insights": {
    "total_stocks": 1500,
    "avg_change": 1.2,
    "volatility_stats": {...},
    "cluster_distribution": {...}
  }
}
```

### 5. Portfolio Optimizer
**GET** `/api/v1/ai/portfolio-optimizer`

AI-powered portfolio optimization for given stocks.

**Parameters:**
- `symbols` (string): Comma-separated stock symbols (required)
- `investment_amount` (float): Total investment amount (default: 100000)
- `risk_tolerance` (string): "low", "medium", "high" (default: "medium")

**Response:**
```json
{
  "optimized_portfolio": [
    {
      "symbol": "RELIANCE",
      "allocation_percentage": 25.5,
      "investment_amount": 25500,
      "quantity": 10,
      "price": 2550,
      "risk_score": 3.2
    }
  ],
  "total_investment": 100000,
  "risk_tolerance": "medium",
  "optimization_method": "Risk-adjusted allocation"
}
```

### 6. Momentum Analysis (NEW)
**GET** `/api/v1/ai/momentum-analysis`

AI-powered momentum stock identification for trend trading.

**Parameters:**
- `timeframe` (string): "intraday", "daily", "weekly" (default: "daily")
- `min_volume` (int): Minimum volume threshold (default: 1000000)
- `limit` (int): Number of results (1-50, default: 20)

**Response:**
```json
{
  "momentum_stocks": [
    {
      "symbol": "ADANIPOWER",
      "perChange": 13.42,
      "momentum_score": 85.6,
      "trend_strength": "Strong Bullish",
      "trade_quantity": 15000000,
      "price_volatility": 8.5
    }
  ],
  "timeframe": "daily",
  "min_volume_filter": 1000000,
  "analysis_method": "AI Momentum Scoring",
  "total_candidates": 25
}
```

### 7. Scalping Analysis (NEW)
**GET** `/api/v1/ai/scalping-analysis`

AI-powered scalping opportunities for intraday trading.

**Parameters:**
- `volatility_threshold` (float): Minimum volatility (1.0-10.0, default: 2.0)
- `volume_threshold` (int): Minimum volume (default: 5000000)
- `limit` (int): Number of results (1-30, default: 15)

**Response:**
```json
{
  "scalping_opportunities": [
    {
      "symbol": "RELIANCE",
      "scalping_score": 92.3,
      "liquidity_rating": "Excellent",
      "price_volatility": 4.2,
      "trade_quantity": 25000000,
      "ltp": 2550.75
    }
  ],
  "volatility_threshold": 2.0,
  "volume_threshold": 5000000,
  "analysis_method": "AI Scalping Analysis",
  "total_opportunities": 12
}
```

### 8. Comprehensive Options Analysis (UPDATED)
**GET** `/api/v1/ai/options-analysis`

Unified AI-powered options analysis with comprehensive market insights.

**Parameters:**
- `symbol` (string): Stock symbol for option chain analysis (default: "NIFTY")
- `expiry` (string): Expiry date (optional, uses first available)
- `analysis_type` (string): "comprehensive", "option_chain", "market_wide" (default: "comprehensive")
- `strategy_type` (string): "all", "bullish", "bearish", "neutral" (default: "all")
- `limit` (int): Number of strategies (1-50, default: 25)

**Response:**
```json
{
  "option_chain_analysis": {
    "symbol": "NIFTY",
    "expiry": "2024-01-25",
    "underlying_value": 25400,
    "option_chain_metrics": {
      "total_strikes": 45,
      "call_volume": 29551513,
      "put_volume": 34466892,
      "volume_pcr": 1.17,
      "call_oi": 15000000,
      "put_oi": 18500000,
      "oi_pcr": 1.23,
      "max_pain_strike": 25400
    },
    "available_expiries": ["2024-01-25", "2024-02-01"]
  },
  "market_wide_analysis": {
    "market_regime_analysis": {
      "regime": "High Volatility",
      "fear_greed_index": 0,
      "volatility_percentile": 88.01,
      "liquidity_score": 100
    },
    "advanced_features": {
      "pcr": 1.17,
      "implied_volatility_estimate": 44.01,
      "volume_ratio": 64.02,
      "avg_price_change": -20.62,
      "moneyness_ratio": 0.3,
      "volatility_skew": 79.19
    },
    "traditional_metrics": {
      "pcr": 1.17,
      "call_volume": 29551513,
      "put_volume": 34466892,
      "avg_call_change": -60.22,
      "avg_put_change": 18.97
    },
    "total_strikes_analyzed": 20
  },
  "ai_recommended_strategies": [
    {
      "name": "AI Long Straddle",
      "type": "Volatility",
      "confidence_score": 95,
      "risk_reward": 2.2,
      "description": "AI predicts high volatility (48.9). Long straddle profits from big moves in either direction.",
      "reasoning": [
        "High price uncertainty (-20.6%)",
        "Extreme volatility skew (79.2)",
        "Elevated PCR (1.17) shows fear"
      ],
      "strategy_explanation": "Buy Call + Put at same strike. Profits if price moves significantly up OR down. Ideal for volatile but directionless markets.",
      "strike_recommendation": {
        "recommended_strike": 25400,
        "spot_estimate": 25400,
        "strike_selection_reason": "ATM strike 25400 selected for maximum gamma exposure"
      },
      "ai_insights": {
        "predicted_volatility": 48.86,
        "market_regime": "High Volatility",
        "volatility_percentile": 88.01
      }
    }
  ],
  "strategy_filter": "all",
  "analysis_method": "Comprehensive AI Options Analysis",
  "analysis_type": "comprehensive"
}
```

### 8.1. Options Spike Detection (NEW)
**GET** `/api/v1/ai/options-spike-detection`

Real-time spike detection with historical comparison over time periods.

**Parameters:**
- `symbol` (string): Stock symbol (default: "NIFTY")
- `time_period` (int): Minutes to compare (default: 5)
- `spike_threshold` (float): Price change % threshold (default: 20.0)
- `auto_store` (bool): Auto-store current data (default: true)
- `limit` (int): Max results (default: 20)

**Response:**
```json
{
  "symbol": "NIFTY",
  "expiry": "23-Sep-2025",
  "time_period_minutes": 5,
  "period_spikes": [
    {
      "strike": 25400,
      "option_type": "Call",
      "price_change_pct": 25.3,
      "price_change_absolute": 15.50,
      "volume_change_pct": 150.5,
      "volume_change_absolute": 5000,
      "value_change_pct": 45.8,
      "value_change_absolute": 2500000,
      "current_price": 76.80,
      "previous_price": 61.30,
      "current_volume": 8320,
      "previous_volume": 3320,
      "current_value": 7950000,
      "previous_value": 5450000,
      "bid_price": 76.50,
      "ask_price": 77.00,
      "open_interest": 15000,
      "oi_change": 500,
      "implied_volatility": 18.5,
      "change_summary": "₹15.50 (+25.3%) | Value: ₹25,00,000 (+45.8%) in 5min",
      "spike_type": "Price"
    }
  ],
  "current_spikes": [
    {
      "strike": 25050,
      "option_type": "Call",
      "price_change": 30.78,
      "volume": 19041,
      "last_price": 335.45,
      "bid_price": 335.00,
      "ask_price": 336.00,
      "open_interest": 1287,
      "oi_change": -660,
      "implied_volatility": 7.9,
      "spike_type": "Current Price Spike"
    }
  ],
  "total_period_spikes": 3,
  "total_current_spikes": 5,
  "analysis_method": "Time-based Spike Detection",
  "refresh_note": "Call this endpoint every minute to track spikes over time"
}
```

### 8.2. Manual Data Refresh
**GET** `/api/v1/ai/refresh-spike-data`

Manually refresh and store current options data for spike tracking.

**Parameters:**
- `symbol` (string): Stock symbol (default: "NIFTY")

**Response:**
```json
{
  "status": "success",
  "message": "Data refreshed for NIFTY",
  "timestamp": "2025-01-21T10:30:00",
  "options_stored": 45,
  "underlying_value": 25400.50
}
```

### 8.3. Charting Data (NEW)
**POST** `/api/v1/charting/data`

Get historical OHLCV data similar to NSE charting API.

**Parameters:**
- `tradingSymbol` (string): Trading symbol (e.g., "HDFCBANK-EQ")
- `chartPeriod` (string): "I" for Intraday, "D" for Daily (default: "I")
- `timeInterval` (int): Minutes for intraday data (default: 3)
- `chartStart` (int): Start timestamp (default: 0)
- `fromDate` (int): From timestamp (default: 0)
- `toDate` (int): To timestamp (default: current time)

**Response:**
```json
{
  "s": "Ok",
  "t": [1755767871, 1755768059, 1755768233],
  "o": [997.5, 993.0, 993.75],
  "h": [997.5, 993.9, 994.55],
  "l": [992.55, 992.65, 993.65],
  "c": [993.05, 993.75, 994.25],
  "v": [493612, 278028, 83628]
}
```

### 8.4. AI Trade Decision (NEW)
**POST** `/api/v1/ai/trade-decision`

AI-powered trade decision analysis - provides BUY/SELL/HOLD recommendations.

**Parameters:**
- `tradingSymbol` (string): Trading symbol (e.g., "HDFCBANK-EQ")
- `chartPeriod` (string): "I" for Intraday, "D" for Daily (default: "I")
- `timeInterval` (int): Minutes for analysis (default: 3)

**Response:**
```json
{
  "symbol": "HDFCBANK-EQ",
  "decision": "BUY",
  "confidence": 85,
  "current_price": 1002.50,
  "price_change_pct": 2.45,
  "trend": "UP",
  "volatility": 3.2,
  "volume_spike": true,
  "reasons": [
    "Strong upward momentum",
    "Volume spike detected",
    "Positive trend"
  ],
  "analysis_time": "2025-01-21T15:30:00",
  "recommendation": "BUY with 85% confidence"
}
```

### 8.6. Available Symbols
**GET** `/api/v1/charting/symbols`

Get list of available symbols for charting.

**Response:**
```json
{
  "status": "success",
  "symbols": [
    {
      "symbol": "HDFCBANK",
      "name": "HDFC Bank Limited",
      "tradingSymbol": "HDFCBANK-EQ"
    }
  ],
  "total": 50
}
```

### 8.5. Legacy Option Chain Analysis
**GET** `/api/v1/ai/option-chain-analysis`

Backward compatible endpoint for option chain specific analysis.

**Parameters:**
- `symbol` (string): Stock symbol (default: "NIFTY")
- `expiry` (string): Expiry date (optional)

**Response:**
```json
{
  "symbol": "NIFTY",
  "expiry": "2024-01-25",
  "underlying_value": 25400,
  "option_chain_metrics": {
    "volume_pcr": 1.17,
    "oi_pcr": 1.23,
    "max_pain_strike": 25400
  },
  "ai_strategies": [...],
  "analysis_method": "AI Option Chain Analysis"
}
```

---

## Enhanced Filtered Endpoints

### 1. Top Gainers (Enhanced)
**GET** `/api/v1/filtered/top-gainers`

**Parameters:**
- `limit` (int): Number of results (1-50, default: 10)

### 2. High Volume Stocks
**GET** `/api/v1/filtered/high-volume-stocks`

**Parameters:**
- `min_volume` (int): Minimum volume threshold (default: 1000000)
- `limit` (int): Number of results (1-100, default: 20)

### 3. Price Range Filter
**GET** `/api/v1/filtered/price-range-stocks`

**Parameters:**
- `min_price` (float): Minimum price (default: 0)
- `max_price` (float): Maximum price (default: 10000)
- `limit` (int): Number of results (1-200, default: 50)

### 4. Momentum Stocks
**GET** `/api/v1/filtered/momentum-stocks`

**Parameters:**
- `min_change` (float): Minimum percentage change (default: 2.0)
- `min_volume` (int): Minimum volume (default: 100000)
- `limit` (int): Number of results (1-100, default: 25)

### 5. Market Summary
**GET** `/api/v1/filtered/market-summary`

Comprehensive market overview with key metrics.

---

## Machine Learning Features

### Clustering Algorithm
- **Algorithm**: K-Means clustering with 5 clusters
- **Features**: Price, volume, volatility, percentage change
- **Use Case**: Group similar stocks for recommendations

### Anomaly Detection
- **Algorithm**: Isolation Forest
- **Purpose**: Detect unusual market movements
- **Threshold**: 10% contamination rate

### Risk Assessment
- **Low Risk**: Price change ≤ 3%, volatility ≤ 2%
- **Medium Risk**: Price change ≤ 7%, volatility ≤ 5%
- **High Risk**: Price change > 3% or volatility > 3%

### Momentum Scoring
- **Algorithm**: Weighted combination of price change (40%), volume (30%), volatility (30%)
- **Score Range**: 0-100 (higher = stronger momentum)
- **Threshold**: Minimum 60 for momentum classification

### Scalping Suitability
- **Liquidity Score**: Based on trading volume (40% weight)
- **Volatility Score**: Intraday price movement (40% weight)
- **Price Score**: Stock price level (20% weight)
- **Rating**: Excellent > Good > Fair > Poor

### Options Strategy Selection
- **AI-Powered Strategies**: Long Straddle, Iron Condor, Bull Call Spread, Bear Put Spread
- **Market Regime Analysis**: High/Low Volatility, Fear/Greed Index
- **Advanced Features**: PCR, Volatility Skew, Moneyness Ratio, IV Estimation
- **Strike Recommendations**: ATM/ITM/OTM based on AI predictions
- **Confidence Threshold**: Minimum 65% for recommendations
- **Max Pain Calculation**: Automatic strike price where options expire worthless

### Spike Detection Features
- **Time-based Analysis**: Compare prices over 1-60 minute periods
- **Historical Storage**: 24-hour data retention in spike_history.json
- **Multi-metric Tracking**: Price, Volume, Value, and OI changes
- **Real-time Alerts**: Immediate spike detection with configurable thresholds
- **Absolute & Percentage Changes**: Both rupee amounts and percentage changes
- **Auto-refresh**: Automatic data storage on each API call

### Trade Decision AI
- **Decision Types**: BUY, SELL, HOLD with confidence scores
- **Technical Analysis**: Price momentum, volume spikes, trend analysis
- **Volatility Assessment**: Risk evaluation for trade timing
- **Real-time Recommendations**: Instant trade decisions based on current data

### Sentiment Scoring
- **Range**: -100 (Very Bearish) to +100 (Very Bullish)
- **Factors**: Market breadth, gainer/loser strength
- **Labels**: Very Bearish, Bearish, Neutral, Bullish, Very Bullish

---

## Installation & Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Start Server
```bash
python run.py
```

### Train ML Models
The system automatically trains ML models using collected market data on startup.

---

## Usage Examples

### Python Client
```python
import requests

# Get AI stock recommendations
response = requests.get(
    "http://localhost:8000/api/v1/ai/smart-picks",
    params={"risk_level": "medium", "investment_amount": 50000}
)
recommendations = response.json()

# Detect market anomalies
anomalies = requests.get(
    "http://localhost:8000/api/v1/ai/anomaly-detection"
).json()

# Find similar stocks
similar = requests.get(
    "http://localhost:8000/api/v1/ai/similar-stocks/RELIANCE"
).json()

# Get momentum stocks for trend trading
momentum = requests.get(
    "http://localhost:8000/api/v1/ai/momentum-analysis",
    params={"timeframe": "daily", "min_volume": 2000000}
).json()

# Find scalping opportunities
scalping = requests.get(
    "http://localhost:8000/api/v1/ai/scalping-analysis",
    params={"volatility_threshold": 3.0, "volume_threshold": 10000000}
).json()

# Get comprehensive options analysis
options = requests.get(
    "http://localhost:8000/api/v1/ai/options-analysis",
    params={"symbol": "NIFTY", "analysis_type": "comprehensive"}
).json()

# Get specific option chain analysis
chain_analysis = requests.get(
    "http://localhost:8000/api/v1/ai/option-chain-analysis",
    params={"symbol": "BANKNIFTY", "expiry": "2024-01-25"}
).json()

# Detect options spikes over time
spike_detection = requests.get(
    "http://localhost:8000/api/v1/ai/options-spike-detection",
    params={"symbol": "NIFTY", "time_period": 5, "spike_threshold": 20.0}
).json()

# Manually refresh spike data
refresh_data = requests.get(
    "http://localhost:8000/api/v1/ai/refresh-spike-data",
    params={"symbol": "BANKNIFTY"}
).json()

# Get AI trade decision
trade_decision = requests.post(
    "http://localhost:8000/api/v1/ai/trade-decision",
    json={"tradingSymbol": "HDFCBANK-EQ", "timeInterval": 5}
).json()
```

### JavaScript Client
```javascript
// Market sentiment analysis
fetch('http://localhost:8000/api/v1/ai/market-sentiment')
  .then(response => response.json())
  .then(data => {
    console.log('Market Sentiment:', data.sentiment_label);
    console.log('Sentiment Score:', data.sentiment_score);
  });

// Portfolio optimization
const symbols = "RELIANCE,TCS,INFY,HDFC,ICICIBANK";
fetch(`http://localhost:8000/api/v1/ai/portfolio-optimizer?symbols=${symbols}&investment_amount=100000`)
  .then(response => response.json())
  .then(portfolio => console.log(portfolio));

// Momentum analysis for trend trading
fetch('http://localhost:8000/api/v1/ai/momentum-analysis?timeframe=daily&limit=10')
  .then(response => response.json())
  .then(data => {
    console.log('Top Momentum Stocks:', data.momentum_stocks);
  });

// Scalping opportunities
fetch('http://localhost:8000/api/v1/ai/scalping-analysis?volatility_threshold=2.5')
  .then(response => response.json())
  .then(data => {
    console.log('Scalping Opportunities:', data.scalping_opportunities);
  });

// Comprehensive options analysis
fetch('http://localhost:8000/api/v1/ai/options-analysis?symbol=NIFTY&analysis_type=comprehensive')
  .then(response => response.json())
  .then(data => {
    console.log('AI Strategies:', data.ai_recommended_strategies);
    console.log('Market Regime:', data.market_wide_analysis.market_regime_analysis);
    console.log('Option Chain Metrics:', data.option_chain_analysis.option_chain_metrics);
  });

// Legacy option chain analysis
fetch('http://localhost:8000/api/v1/ai/option-chain-analysis?symbol=BANKNIFTY')
  .then(response => response.json())
  .then(data => {
    console.log('Max Pain Strike:', data.option_chain_metrics.max_pain_strike);
  });

// Options spike detection
fetch('http://localhost:8000/api/v1/ai/options-spike-detection?symbol=NIFTY&time_period=5')
  .then(response => response.json())
  .then(data => {
    console.log('Period Spikes:', data.period_spikes);
    console.log('Current Spikes:', data.current_spikes);
  });

// Manual data refresh
fetch('http://localhost:8000/api/v1/ai/refresh-spike-data?symbol=NIFTY')
  .then(response => response.json())
  .then(data => {
    console.log('Data refreshed:', data.message);
  });

// AI Trade Decision
fetch('http://localhost:8000/api/v1/ai/trade-decision', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({tradingSymbol: 'HDFCBANK-EQ', timeInterval: 5})
})
  .then(response => response.json())
  .then(data => {
    console.log('Trade Decision:', data.decision);
    console.log('Confidence:', data.confidence + '%');
    console.log('Reasons:', data.reasons);
  });
```

---

## Performance & Scalability

- **Response Time**: < 2 seconds for most endpoints
- **Concurrent Users**: Supports 100+ concurrent requests
- **Data Freshness**: Real-time NSE data
- **ML Model Training**: Automatic on startup, can be retrained

## Error Handling

All endpoints return structured error responses:

```json
{
  "detail": "Error description",
  "status_code": 500
}
```

## Support & Updates

- API Version: 2.0.0
- ML Models: Automatically updated with new data
- Feature Requests: Contact development team