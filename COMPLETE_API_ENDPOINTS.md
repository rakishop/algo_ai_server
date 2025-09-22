# MyAlgoFax API - Complete Endpoints Reference

## Base URL
```
http://localhost:8000
```

## 🔥 Core Endpoints

### Root & Status
- **GET** `/` - Welcome message and API info
- **GET** `/test` - Test endpoint status

---

## 🤖 AI & Machine Learning Endpoints

### AI Trade Analysis
- **POST** `/api/v1/ai/trade-decision` - Get AI trade recommendations
- **POST** `/api/v1/ai/multi-stock-decision` - Multi-stock AI analysis
- **GET** `/api/v1/ai/time-based-opportunities` - Time-based trading opportunities
- **GET** `/api/v1/ai/enhanced-market-analysis` - Enhanced market analysis with risk assessment

### AI Scalping & Options
- **GET** `/api/v1/ai/scalping-test` - AI scalping analysis
- **GET** `/api/v1/ai/options-analysis` - Comprehensive options analysis
- **GET** `/api/v1/ai/options-spike-detection` - Options spike detection
- **GET** `/api/v1/ai/refresh-spike-data` - Refresh spike tracking data
- **GET** `/api/v1/ai/option-chain-analysis` - Legacy option chain analysis

---

## 📊 Market Data Endpoints

### Market Overview
- **GET** `/api/v1/market/52-week-extremes` - 52-week high/low stocks
- **GET** `/api/v1/market/daily-movers` - Gainers and losers
- **GET** `/api/v1/market/activity-summary` - Market activity summary
- **GET** `/api/v1/market/price-band-hits` - Price band hitters
- **GET** `/api/v1/market/volume-leaders` - Volume gainers
- **GET** `/api/v1/market/breadth-indicators` - Advance/decline data
- **GET** `/api/v1/market/trading-statistics` - Trading statistics
- **GET** `/api/v1/market/block-deals` - Large deals data

### Legacy Market Endpoints
- **GET** `/api/v1/market/gainers` - Top gainers
- **GET** `/api/v1/market/losers` - Top losers
- **GET** `/api/v1/market/most-active` - Most active securities

---

## 📈 Derivatives & Options

### Derivatives Data
- **GET** `/api/v1/derivatives/market-snapshot` - Derivatives market snapshot
- **GET** `/api/v1/derivatives/active-underlyings` - Most active underlyings
- **GET** `/api/v1/derivatives/open-interest-spurts` - OI spurts data

### Option Chain
- **GET** `/api/option-chain-contract-info` - Option chain contract info
- **GET** `/api/v1/option-chain/{symbol}` - Option chain data
- **GET** `/api/v1/option-chain-info/{symbol}` - Option chain info

---

## 📊 Indices Data

### Indices
- **GET** `/api/v1/indices/live-data` - All indices live data

---

## 📊 Charting & Technical Analysis

### Charting Data
- **POST** `/api/v1/charting/data` - OHLCV charting data
- **GET** `/api/v1/charting/symbols` - Available symbols for charting

### Technical Indicators
- **POST** `/api/v1/technical/indicators` - Calculate technical indicators
- **GET** `/api/v1/technical/fibonacci/{symbol}` - Fibonacci retracement levels
- **GET** `/api/v1/technical/pivot-points/{symbol}` - Pivot points and S/R levels

---

## 🔍 Market Scanner

### Scanner Endpoints
- **GET** `/api/v1/scanner/breakouts` - Breakout stocks scanner
- **GET** `/api/v1/scanner/momentum` - Momentum stocks scanner
- **GET** `/api/v1/scanner/reversals` - Reversal candidates scanner
- **GET** `/api/v1/scanner/gaps` - Gap opportunities scanner
- **GET** `/api/v1/scanner/comprehensive` - Comprehensive market scan

---

## 💼 Portfolio Management

### Portfolio Operations
- **POST** `/api/v1/portfolio/create` - Create new portfolio
- **POST** `/api/v1/portfolio/{portfolio_id}/add-position` - Add position to portfolio
- **GET** `/api/v1/portfolio/{portfolio_id}/performance` - Get portfolio performance
- **GET** `/api/v1/portfolio/{portfolio_id}/recommendations` - Get AI recommendations

---

## ⚠️ Risk Management

### Risk Assessment
- **POST** `/api/v1/risk/calculate-position-size` - Calculate optimal position size
- **GET** `/api/v1/risk/market-assessment` - Market risk assessment

---

## 🌐 WebSocket & Real-time

### WebSocket
- **WS** `/ws` - Real-time WebSocket connection
- **GET** `/api/v1/websocket/info` - WebSocket connection info
- **GET** `/api/v1/websocket/test` - Test WebSocket connection

---

## 📋 Complete Endpoint List (Alphabetical)

### A-C
- **GET** `/api/option-chain-contract-info`
- **POST** `/api/v1/ai/multi-stock-decision`
- **GET** `/api/v1/ai/enhanced-market-analysis`
- **GET** `/api/v1/ai/option-chain-analysis`
- **GET** `/api/v1/ai/options-analysis`
- **GET** `/api/v1/ai/options-spike-detection`
- **GET** `/api/v1/ai/refresh-spike-data`
- **GET** `/api/v1/ai/scalping-test`
- **GET** `/api/v1/ai/time-based-opportunities`
- **POST** `/api/v1/ai/trade-decision`
- **POST** `/api/v1/charting/data`
- **GET** `/api/v1/charting/symbols`

### D-M
- **GET** `/api/v1/derivatives/active-underlyings`
- **GET** `/api/v1/derivatives/market-snapshot`
- **GET** `/api/v1/derivatives/open-interest-spurts`
- **GET** `/api/v1/indices/live-data`
- **GET** `/api/v1/market/52-week-extremes`
- **GET** `/api/v1/market/activity-summary`
- **GET** `/api/v1/market/block-deals`
- **GET** `/api/v1/market/breadth-indicators`
- **GET** `/api/v1/market/daily-movers`
- **GET** `/api/v1/market/gainers`
- **GET** `/api/v1/market/losers`
- **GET** `/api/v1/market/most-active`
- **GET** `/api/v1/market/price-band-hits`
- **GET** `/api/v1/market/trading-statistics`
- **GET** `/api/v1/market/volume-leaders`

### O-S
- **GET** `/api/v1/option-chain/{symbol}`
- **GET** `/api/v1/option-chain-info/{symbol}`
- **POST** `/api/v1/portfolio/create`
- **POST** `/api/v1/portfolio/{portfolio_id}/add-position`
- **GET** `/api/v1/portfolio/{portfolio_id}/performance`
- **GET** `/api/v1/portfolio/{portfolio_id}/recommendations`
- **POST** `/api/v1/risk/calculate-position-size`
- **GET** `/api/v1/risk/market-assessment`
- **GET** `/api/v1/scanner/breakouts`
- **GET** `/api/v1/scanner/comprehensive`
- **GET** `/api/v1/scanner/gaps`
- **GET** `/api/v1/scanner/momentum`
- **GET** `/api/v1/scanner/reversals`

### T-W
- **POST** `/api/v1/technical/indicators`
- **GET** `/api/v1/technical/fibonacci/{symbol}`
- **GET** `/api/v1/technical/pivot-points/{symbol}`
- **GET** `/api/v1/websocket/info`
- **GET** `/api/v1/websocket/test`
- **GET** `/test`
- **WS** `/ws`

---

## 📝 Request/Response Examples

### AI Trade Decision
```bash
POST /api/v1/ai/trade-decision
{
  "tradingSymbol": "RELIANCE",
  "timeInterval": 15,
  "lookbackPeriod": 50
}
```

### Portfolio Creation
```bash
POST /api/v1/portfolio/create
{
  "portfolio_id": "port_001",
  "name": "AI Portfolio",
  "capital": 1000000,
  "risk_tolerance": "MEDIUM"
}
```

### WebSocket Connection
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.send(JSON.stringify({
  "action": "subscribe",
  "symbol": "RELIANCE"
}));
```

### Market Scanner
```bash
GET /api/v1/scanner/comprehensive
```

### Technical Analysis
```bash
POST /api/v1/technical/indicators
{
  "data": [
    {"o": 2450, "h": 2480, "l": 2440, "c": 2470, "v": 1500000}
  ]
}
```

---

## 🔧 Parameters & Options

### Common Parameters
- `symbol` - Stock symbol (e.g., "RELIANCE", "TCS")
- `timeInterval` - Time interval in minutes (1, 3, 5, 15, 30, 60)
- `limit` - Number of results to return
- `portfolio_id` - Portfolio identifier
- `risk_tolerance` - "LOW", "MEDIUM", "HIGH"

### Response Formats
All endpoints return JSON with consistent error handling:
```json
{
  "error": "Error message",
  "code": "ERROR_CODE",
  "suggestion": "How to fix"
}
```

---

## 📊 Total Endpoints: **50+**

### By Category:
- **AI/ML Endpoints**: 8
- **Market Data**: 11
- **Derivatives/Options**: 6
- **Technical Analysis**: 3
- **Portfolio Management**: 4
- **Risk Management**: 2
- **Scanner**: 5
- **WebSocket**: 3
- **Charting**: 2
- **Indices**: 1
- **Utility**: 5

---

## 🚀 Quick Start Guide

1. **Get Market Overview**: `GET /api/v1/market/daily-movers`
2. **AI Analysis**: `POST /api/v1/ai/trade-decision`
3. **Real-time Data**: Connect to `ws://localhost:8000/ws`
4. **Portfolio Tracking**: `POST /api/v1/portfolio/create`
5. **Market Scanning**: `GET /api/v1/scanner/comprehensive`

---

*Complete API reference for MyAlgoFax NSE Market Data API v3.0*