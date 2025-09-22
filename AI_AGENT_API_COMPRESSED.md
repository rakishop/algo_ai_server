# MyAlgoFax API - AI Agent Quick Reference

## Base URL
```
http://localhost:8000
```

## Core Endpoints for AI Agents

### 1. AI Trade Decision
**POST** `/api/v1/ai/trade-decision`
```json
{
  "tradingSymbol": "RELIANCE",
  "timeInterval": 15,
  "lookbackPeriod": 50
}
```
**Returns:** `{decision: "BUY/SELL/HOLD", confidence: 85, ml_score: 0.847, current_price: 2456.75}`

### 2. Multi-Stock Analysis
**POST** `/api/v1/ai/multi-stock-decision`
```json
{
  "symbols": ["RELIANCE", "TCS", "HDFC"],
  "timeInterval": 15,
  "topN": 5
}
```
**Returns:** Ranked buy/sell recommendations with confidence scores

### 3. Market Scanner
**GET** `/api/v1/scanner/comprehensive`
**Returns:** Breakouts, momentum, reversals, gaps opportunities

### 4. Portfolio Management
**POST** `/api/v1/portfolio/create`
```json
{
  "portfolio_id": "port_001",
  "name": "AI Portfolio",
  "capital": 1000000,
  "risk_tolerance": "MEDIUM"
}
```

**GET** `/api/v1/portfolio/{portfolio_id}/performance`
**Returns:** Portfolio PnL, positions, recommendations

### 5. Real-time WebSocket
**WS** `/ws`
```javascript
ws.send(JSON.stringify({action: "subscribe", symbol: "RELIANCE"}));
```

### 6. Market Data
**GET** `/api/v1/market/gainers` - Top gainers
**GET** `/api/v1/market/losers` - Top losers  
**GET** `/api/v1/market/most-active` - High volume stocks

### 7. Technical Analysis
**POST** `/api/v1/technical/indicators`
```json
{
  "data": [{"o": 2450, "h": 2480, "l": 2440, "c": 2470, "v": 1500000}]
}
```
**Returns:** RSI, MACD, Bollinger Bands, signals

### 8. Risk Management
**POST** `/api/v1/risk/calculate-position-size`
```json
{
  "portfolio_value": 1000000,
  "risk_per_trade": 0.02,
  "entry_price": 2456.75,
  "stop_loss": 2300.00
}
```

## Quick Response Formats

### Trade Decision Response
```json
{
  "symbol": "RELIANCE",
  "decision": "BUY",
  "confidence": 85,
  "current_price": 2456.75,
  "reasons": ["Strong upward trend", "High volume activity"]
}
```

### Market Data Response
```json
{
  "symbol": "RELIANCE",
  "ltp": 2456.75,
  "perChange": 2.3,
  "volume": 1250000,
  "high": 2480.0,
  "low": 2440.0
}
```

### Portfolio Response
```json
{
  "total_pnl": 25430,
  "total_pnl_percentage": 2.54,
  "positions": [
    {
      "symbol": "RELIANCE",
      "pnl": 5355,
      "pnl_percentage": 2.18
    }
  ]
}
```

## AI Agent Usage Patterns

### 1. Get Trading Signals
```bash
curl -X POST "http://localhost:8000/api/v1/ai/trade-decision" \
  -H "Content-Type: application/json" \
  -d '{"tradingSymbol": "RELIANCE", "timeInterval": 15}'
```

### 2. Scan Market Opportunities
```bash
curl "http://localhost:8000/api/v1/scanner/comprehensive"
```

### 3. Monitor Portfolio
```bash
curl "http://localhost:8000/api/v1/portfolio/port_001/performance"
```

### 4. Real-time Updates
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.send(JSON.stringify({action: "subscribe", symbol: "RELIANCE"}));
```

## Error Handling
All endpoints return standard format:
```json
{
  "error": "Error message",
  "code": "ERROR_CODE",
  "suggestion": "How to fix"
}
```

## Rate Limits
- AI endpoints: 50 req/min
- Market data: 200 req/min
- WebSocket: 10 connections/IP

## Key Parameters
- `timeInterval`: 1,3,5,15,30,60 (minutes)
- `confidence`: 0-100 (higher = more confident)
- `risk_tolerance`: LOW, MEDIUM, HIGH
- `decision`: BUY, SELL, HOLD

## Status Codes
- 200: Success
- 400: Bad request
- 429: Rate limit exceeded
- 500: Server error