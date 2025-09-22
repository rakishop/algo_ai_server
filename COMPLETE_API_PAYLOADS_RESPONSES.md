# MyAlgoFax API - Complete Payloads & Responses

## Base URL
```
http://localhost:8000
```

---

## 🤖 AI & Machine Learning Endpoints

### 1. AI Trade Decision
**POST** `/api/v1/ai/trade-decision`

**Payload:**
```json
{
  "tradingSymbol": "RELIANCE",
  "timeInterval": 15,
  "lookbackPeriod": 50,
  "analysisDepth": "comprehensive"
}
```

**Response:**
```json
{
  "symbol": "RELIANCE",
  "decision": "BUY",
  "confidence": 85,
  "ml_score": 0.847,
  "current_price": 2456.75,
  "price_change_pct": 2.3,
  "technical_indicators": {
    "sma_5": 2445.20,
    "sma_20": 2420.15,
    "rsi": 65.4,
    "volatility_pct": 1.8
  },
  "volume_analysis": {
    "current_volume": 1250000,
    "avg_volume": 980000,
    "volume_ratio": 1.28
  },
  "reasons": [
    "Strong upward trend - price above both SMAs",
    "High volume activity (1.3x average)"
  ],
  "analysis_time": "2024-01-15T10:30:00"
}
```

### 2. Multi-Stock Analysis
**POST** `/api/v1/ai/multi-stock-decision`

**Payload:**
```json
{
  "symbols": ["RELIANCE", "TCS", "HDFC"],
  "timeInterval": 15,
  "topN": 5
}
```

**Response:**
```json
{
  "analysis_summary": {
    "total_analyzed": 3,
    "buy_opportunities": 2,
    "sell_opportunities": 0,
    "hold_recommendations": 1
  },
  "top_buy_recommendations": [
    {
      "symbol": "RELIANCE",
      "decision": "BUY",
      "confidence": 87,
      "current_price": 2456.75,
      "price_change_pct": 2.3,
      "rsi": 65.4
    }
  ]
}
```

### 3. Enhanced Market Analysis
**GET** `/api/v1/ai/enhanced-market-analysis?analysis_type=comprehensive&limit=20`

**Response:**
```json
{
  "analysis_type": "comprehensive",
  "total_stocks_analyzed": 20,
  "top_opportunities": [
    {
      "symbol": "RELIANCE",
      "ltp": 2456.75,
      "perChange": 2.3,
      "scalping_score": 85.5,
      "price_volatility": 1.8,
      "volume_ratio": 1.28
    }
  ],
  "risk_assessment": {
    "market_volatility": 2.45,
    "risk_level": "MEDIUM",
    "market_sentiment": "BULLISH"
  }
}
```

---

## 📊 Market Data Endpoints

### 4. Market Gainers & Losers
**GET** `/api/v1/market/daily-movers`

**Response:**
```json
{
  "gainers": {
    "NIFTY": {
      "data": [
        {
          "symbol": "ADANIENT",
          "series": "EQ",
          "open_price": 2495,
          "high_price": 2579,
          "low_price": 2466.1,
          "ltp": 2528,
          "prev_price": 2402,
          "perChange": 5.25,
          "trade_quantity": 7965135,
          "turnover": 200722.19851349998
        }
      ],
      "timestamp": "19-Sep-2025 16:00:00"
    }
  },
  "losers": {
    "NIFTY": {
      "data": [
        {
          "symbol": "HCLTECH",
          "ltp": 1470,
          "prev_price": 1493.8,
          "perChange": -1.59,
          "trade_quantity": 3197957
        }
      ]
    }
  }
}
```

### 5. 52-Week Extremes
**GET** `/api/v1/market/52-week-extremes`

**Response:**
```json
{
  "52_week_high": {
    "high": 67,
    "data": [
      {
        "symbol": "AARADHYA",
        "series": "SM",
        "comapnyName": "Aaradhya Disposal Industries Limited",
        "new52WHL": 127.9,
        "prev52WHL": 124.9,
        "ltp": 122.5,
        "change": -2.15,
        "pChange": -1.7248295226634576
      }
    ],
    "timestamp": "19-Sep-2025 16:00:00"
  },
  "52_week_low": {
    "data": [
      {
        "symbol": "ATCENERGY",
        "new52WHL": 58,
        "prev52WHL": 60.85,
        "ltp": 58.45,
        "change": -4.35,
        "pChange": -6.926751592356688
      }
    ],
    "low": 31
  }
}
```

### 6. Market Activity Summary
**GET** `/api/v1/market/activity-summary`

**Response:**
```json
{
  "most_active_securities": {
    "data": [
      {
        "symbol": "ADANIPOWER",
        "lastPrice": 716.1,
        "pChange": 13.42,
        "quantityTraded": 86324006,
        "totalTradedValue": 60305950591.6,
        "yearHigh": 723,
        "yearLow": 432,
        "dayHigh": 723,
        "dayLow": 665.35
      }
    ]
  },
  "volume_gainers": {
    "data": [
      {
        "symbol": "BHARATGEAR",
        "volume": 2699382,
        "week1AvgVolume": 558729,
        "week1volChange": 4.831285115376025,
        "ltp": 106.98,
        "pChange": 11.85
      }
    ]
  }
}
```

---

## 📈 Portfolio Management

### 7. Create Portfolio
**POST** `/api/v1/portfolio/create`

**Payload:**
```json
{
  "portfolio_id": "portfolio_001",
  "name": "Growth Portfolio",
  "capital": 1000000,
  "risk_tolerance": "MEDIUM"
}
```

**Response:**
```json
{
  "status": "success",
  "portfolio": {
    "portfolio_id": "portfolio_001",
    "name": "Growth Portfolio",
    "total_capital": 1000000,
    "available_cash": 1000000,
    "positions": [],
    "created_date": "2024-01-15T10:30:00",
    "risk_tolerance": "MEDIUM"
  }
}
```

### 8. Add Position
**POST** `/api/v1/portfolio/{portfolio_id}/add-position`

**Payload:**
```json
{
  "symbol": "RELIANCE",
  "quantity": 100,
  "entry_price": 2456.75,
  "position_type": "LONG",
  "stop_loss": 2300.00,
  "target_price": 2600.00
}
```

**Response:**
```json
{
  "status": "success",
  "position_added": {
    "symbol": "RELIANCE",
    "quantity": 100,
    "entry_price": 2456.75,
    "invested_amount": 245675,
    "position_type": "LONG"
  },
  "portfolio_summary": {
    "available_cash": 754325,
    "total_invested": 245675
  }
}
```

### 9. Portfolio Performance
**GET** `/api/v1/portfolio/{portfolio_id}/performance`

**Response:**
```json
{
  "portfolio_id": "portfolio_001",
  "total_capital": 1000000,
  "available_cash": 754325,
  "total_invested": 245675,
  "current_portfolio_value": 1025430,
  "total_pnl": 25430,
  "total_pnl_percentage": 2.54,
  "positions": [
    {
      "symbol": "RELIANCE",
      "quantity": 100,
      "entry_price": 2456.75,
      "current_price": 2510.30,
      "pnl": 5355,
      "pnl_percentage": 2.18,
      "days_held": 5
    }
  ]
}
```

---

## ⚠️ Risk Management

### 10. Calculate Position Size
**POST** `/api/v1/risk/calculate-position-size`

**Payload:**
```json
{
  "portfolio_value": 1000000,
  "risk_per_trade": 0.02,
  "entry_price": 2456.75,
  "stop_loss": 2300.00,
  "risk_tolerance": "MEDIUM"
}
```

**Response:**
```json
{
  "recommended_shares": 127,
  "position_value": 312007.25,
  "risk_amount": 19912.25,
  "risk_percentage": 1.99,
  "position_percentage": 31.20,
  "stop_loss_price": 2300.00
}
```

### 11. Market Risk Assessment
**GET** `/api/v1/risk/market-assessment`

**Response:**
```json
{
  "market_volatility": 2.45,
  "risk_level": "MEDIUM",
  "sentiment_ratio": 0.65,
  "market_sentiment": "BULLISH",
  "avg_volume": 15000000,
  "high_volume_stocks": 12,
  "total_stocks_analyzed": 50,
  "recommendation": "Moderate volatility with bullish sentiment. Normal position sizing recommended."
}
```

---

## 🔍 Market Scanner

### 12. Comprehensive Market Scan
**GET** `/api/v1/scanner/comprehensive`

**Response:**
```json
{
  "scan_timestamp": "2024-01-15T10:30:00",
  "market_overview": {
    "total_gainers": 45,
    "total_losers": 25,
    "market_sentiment": "BULLISH",
    "gainer_loser_ratio": 1.8
  },
  "opportunities": {
    "breakouts": [
      {
        "symbol": "ADANIPORTS",
        "ltp": 1234.50,
        "perChange": 4.25,
        "volume_ratio": 2.1,
        "breakout_score": 85.5,
        "breakout_type": "BULLISH"
      }
    ],
    "momentum": [
      {
        "symbol": "RELIANCE",
        "rsi": 65.4,
        "momentum_score": 78.2,
        "trend": "UPWARD"
      }
    ]
  },
  "summary": {
    "total_opportunities_found": 18,
    "scan_types_completed": 4
  }
}
```

### 13. Breakout Scanner
**GET** `/api/v1/scanner/breakouts?volume_threshold=1.5&price_change_threshold=3.0`

**Response:**
```json
{
  "scan_type": "Breakout Scanner",
  "total_scanned": 150,
  "breakout_opportunities": [
    {
      "symbol": "ADANIPORTS",
      "ltp": 1234.50,
      "perChange": 4.25,
      "trade_quantity": 2500000,
      "breakout_score": 85.5,
      "volume_ratio": 2.1,
      "breakout_type": "BULLISH"
    }
  ],
  "criteria": {
    "min_volume_ratio": 1.5,
    "min_price_change": 3.0
  }
}
```

---

## 📊 Technical Analysis

### 14. Technical Indicators
**POST** `/api/v1/technical/indicators`

**Payload:**
```json
{
  "data": [
    {"o": 2450, "h": 2480, "l": 2440, "c": 2470, "v": 1500000},
    {"o": 2470, "h": 2490, "l": 2460, "c": 2485, "v": 1600000}
  ]
}
```

**Response:**
```json
{
  "bollinger_bands": {
    "upper": [2520.5, 2525.2],
    "middle": [2470.0, 2475.0],
    "lower": [2419.5, 2424.8]
  },
  "macd": {
    "macd": [2.5, 3.1],
    "signal": [2.2, 2.8],
    "histogram": [0.3, 0.3]
  },
  "rsi": [65.5, 68.2],
  "signals": {
    "overall_signal": "BULLISH",
    "strength": 75,
    "bullish_signals": ["Price above Bollinger middle line"],
    "bearish_signals": []
  }
}
```

### 15. Fibonacci Levels
**GET** `/api/v1/technical/fibonacci/RELIANCE?period_days=20`

**Response:**
```json
{
  "symbol": "RELIANCE",
  "period_high": 2490,
  "period_low": 2440,
  "fibonacci_levels": {
    "0%": 2490,
    "23.6%": 2478.2,
    "38.2%": 2470.9,
    "50%": 2465.0,
    "61.8%": 2459.1,
    "78.6%": 2451.4,
    "100%": 2440
  },
  "current_price": 2456.75
}
```

---

## 📈 Charting Data

### 16. OHLCV Data
**POST** `/api/v1/charting/data`

**Payload:**
```json
{
  "tradingSymbol": "RELIANCE",
  "timeInterval": 5,
  "dataPoints": 100
}
```

**Response:**
```json
{
  "s": "Ok",
  "t": [1642234800, 1642234860, 1642234920],
  "o": [1000.50, 1001.20, 1002.10],
  "h": [1005.75, 1006.30, 1007.25],
  "l": [997.25, 998.80, 999.50],
  "c": [1002.30, 1003.15, 1004.80],
  "v": [125000, 130000, 128000],
  "meta": {
    "symbol": "RELIANCE",
    "interval_minutes": 5,
    "data_points": 100
  }
}
```

### 17. Available Symbols
**GET** `/api/v1/charting/symbols`

**Response:**
```json
{
  "status": "success",
  "symbols": [
    {
      "symbol": "RELIANCE",
      "name": "Reliance Industries Limited",
      "current_price": 2456.75,
      "change_pct": 2.3
    }
  ],
  "total": 50,
  "supported_intervals": [1, 3, 5, 15, 30, 60, 240, 1440]
}
```

---

## 📊 Derivatives & Options

### 18. Option Chain
**GET** `/api/v1/option-chain/NIFTY?expiry=2024-01-25`

**Response:**
```json
{
  "symbol": "NIFTY",
  "expiry": "2024-01-25",
  "underlying_value": 21500,
  "calls": [
    {
      "strikePrice": 21000,
      "lastPrice": 520.30,
      "change": 15.20,
      "pChange": 3.01,
      "volume": 125000,
      "openInterest": 2500000,
      "impliedVolatility": 18.5
    }
  ],
  "puts": [
    {
      "strikePrice": 21000,
      "lastPrice": 25.50,
      "change": -2.10,
      "pChange": -7.61,
      "volume": 85000,
      "openInterest": 1800000,
      "impliedVolatility": 19.2
    }
  ]
}
```

### 19. Derivatives Market Snapshot
**GET** `/api/v1/derivatives/market-snapshot`

**Response:**
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "market_summary": {
    "total_contracts": 15000,
    "total_volume": 125000000,
    "total_turnover": 850000000000
  },
  "top_contracts": [
    {
      "symbol": "NIFTY",
      "expiry": "2024-01-25",
      "strike": 21500,
      "type": "CE",
      "ltp": 125.50,
      "volume": 2500000,
      "oi": 15000000
    }
  ]
}
```

---

## 📊 Indices Data

### 20. Live Indices
**GET** `/api/v1/indices/live-data`

**Response:**
```json
{
  "data": [
    {
      "index": "NIFTY 50",
      "last": 25327.05,
      "variation": -96.55,
      "percentChange": -0.38,
      "open": 25410.2,
      "high": 25428.75,
      "low": 25286.3,
      "previousClose": 25423.6,
      "yearHigh": 26277.35,
      "yearLow": 21743.65
    }
  ],
  "timestamp": "19-Sep-2025 15:30"
}
```

---

## 🌐 WebSocket Real-time

### 21. WebSocket Connection
**WS** `/ws`

**Subscribe Message:**
```json
{
  "action": "subscribe",
  "symbol": "RELIANCE"
}
```

**Real-time Update:**
```json
{
  "type": "symbol_update",
  "symbol": "RELIANCE",
  "ltp": 2456.75,
  "change": 15.20,
  "pChange": 0.62,
  "volume": 1250000,
  "timestamp": "2024-01-15T10:30:15"
}
```

### 22. WebSocket Info
**GET** `/api/v1/websocket/info`

**Response:**
```json
{
  "websocket_url": "ws://localhost:8000/ws",
  "status": "active",
  "active_connections": 5,
  "total_subscriptions": 12,
  "subscribe_example": {
    "action": "subscribe",
    "symbol": "RELIANCE"
  },
  "response_types": [
    "subscription_confirmed",
    "market_update",
    "symbol_update"
  ]
}
```

---

## ❌ Error Responses

### Standard Error Format
```json
{
  "error": "Portfolio not found",
  "code": "PORTFOLIO_NOT_FOUND",
  "timestamp": "2024-01-15T10:30:00Z",
  "suggestion": "Create portfolio first using /api/v1/portfolio/create"
}
```

### Common Error Codes
- `INSUFFICIENT_DATA` - Need more historical data
- `INVALID_SYMBOL` - Stock symbol not found
- `PORTFOLIO_NOT_FOUND` - Portfolio ID does not exist
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `ML_ERROR` - Machine learning calculation failed

---

## 📊 Rate Limits

- **AI endpoints**: 50 requests/minute
- **Market data**: 200 requests/minute
- **Portfolio**: 100 requests/minute
- **WebSocket**: 10 connections/IP
- **Scanner**: 50 requests/minute

---

## 🔧 Common Parameters

- `symbol` - Stock symbol (e.g., "RELIANCE", "TCS")
- `timeInterval` - Time interval in minutes (1, 3, 5, 15, 30, 60)
- `limit` - Number of results to return
- `portfolio_id` - Portfolio identifier
- `risk_tolerance` - "LOW", "MEDIUM", "HIGH"
- `confidence` - 0-100 (higher = more confident)
- `decision` - "BUY", "SELL", "HOLD"

---

*Complete API reference with payloads and responses for MyAlgoFax NSE Market Data API v3.0*