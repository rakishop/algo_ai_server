# MyAlgoFax NSE Market Data API v3.0

## Overview

Enterprise-grade NSE market data API with AI-powered analysis, real-time WebSocket streaming, advanced portfolio management, comprehensive risk assessment, and professional-grade market scanning tools.

### New in v3.0
- **Real-time WebSocket Streaming**: Live market data updates
- **Advanced Portfolio Management**: Multi-portfolio tracking with performance analytics
- **Comprehensive Risk Management**: VaR, position sizing, and market risk assessment
- **Professional Market Scanner**: Breakout, momentum, reversal, and gap scanning
- **Advanced Technical Indicators**: 15+ professional indicators including Ichimoku, MACD, Bollinger Bands
- **Enhanced AI Analysis**: Improved ML models with better prediction accuracy

## Base URL
```
http://localhost:8000
```

## WebSocket Streaming

### Endpoint: `/ws`
**Protocol:** WebSocket

**Description:** Real-time market data streaming with symbol subscriptions.

**Connection Example:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

// Subscribe to a symbol
ws.send(JSON.stringify({
    "action": "subscribe",
    "symbol": "RELIANCE"
}));

// Receive real-time updates
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Market update:', data);
};
```

**Message Types:**
- `market_update`: General market data updates (every 30 seconds)
- `subscription_confirmed`: Confirmation of symbol subscription
- `symbol_update`: Specific symbol price updates

## Portfolio Management

### Create Portfolio
**Endpoint:** `POST /api/v1/portfolio/create`

**Request Body:**
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

### Add Position
**Endpoint:** `POST /api/v1/portfolio/{portfolio_id}/add-position`

**Request Body:**
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

### Get Portfolio Performance
**Endpoint:** `GET /api/v1/portfolio/{portfolio_id}/performance`

**Response:**
```json
{
    "portfolio_id": "portfolio_001",
    "portfolio_name": "Growth Portfolio",
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
            "invested_amount": 245675,
            "current_value": 251030,
            "pnl": 5355,
            "pnl_percentage": 2.18,
            "position_type": "LONG",
            "days_held": 5
        }
    ],
    "risk_tolerance": "MEDIUM"
}
```

### Get Portfolio Recommendations
**Endpoint:** `GET /api/v1/portfolio/{portfolio_id}/recommendations`

**Response:**
```json
{
    "portfolio_id": "portfolio_001",
    "recommendations": [
        {
            "symbol": "TCS",
            "action": "BUY",
            "recommended_quantity": 50,
            "current_price": 3456.20,
            "confidence": 78.5,
            "reason": "High ML score: 78.5",
            "risk_level": "MEDIUM"
        }
    ],
    "available_cash": 754325,
    "max_position_size": 50000
}
```

## Risk Management

### Calculate Position Size
**Endpoint:** `POST /api/v1/risk/calculate-position-size`

**Request Body:**
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

### Market Risk Assessment
**Endpoint:** `GET /api/v1/risk/market-assessment`

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

## Market Scanner

### Breakout Scanner
**Endpoint:** `GET /api/v1/scanner/breakouts`

**Parameters:**
- `volume_threshold` (float, optional): Minimum volume ratio (default: 1.5)
- `price_change_threshold` (float, optional): Minimum price change % (default: 3.0)

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
    },
    "scan_time": "2024-01-15T10:30:00"
}
```

### Momentum Scanner
**Endpoint:** `GET /api/v1/scanner/momentum`

**Parameters:**
- `rsi_min` (float, optional): Minimum RSI (default: 30)
- `rsi_max` (float, optional): Maximum RSI (default: 70)
- `min_volume` (int, optional): Minimum volume (default: 1000000)

### Reversal Scanner
**Endpoint:** `GET /api/v1/scanner/reversals`

**Parameters:**
- `oversold_threshold` (float, optional): Maximum price change for oversold (default: -5.0)
- `volume_spike` (float, optional): Volume spike multiplier (default: 2.0)

### Gap Scanner
**Endpoint:** `GET /api/v1/scanner/gaps`

**Parameters:**
- `min_gap_percent` (float, optional): Minimum gap percentage (default: 2.0)

### Comprehensive Market Scan
**Endpoint:** `GET /api/v1/scanner/comprehensive`

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
        "breakouts": [...],
        "momentum": [...],
        "reversals": [...],
        "gaps": [...]
    },
    "summary": {
        "total_opportunities_found": 18,
        "scan_types_completed": 4,
        "recommendation": "Focus on breakout and momentum stocks in current market conditions"
    }
}
```

## Indian Stock Indices Analysis

### Get Index Constituents with 52-Week Data
**Endpoint:** `GET /api/v1/indices/constituents/{index_name}`

**Description:** Get all constituents of an Indian index with their current price and 52-week high/low data. This endpoint dynamically fetches data from NSE.

**Supported Indices:** Any valid NSE index symbol (e.g., NIFTY 50, NIFTY BANK, NIFTY IT, etc.)

**Example:** `GET /api/v1/indices/constituents/NIFTY%2050`

**Response:**
```json
{
    "index_name": "NIFTY 50",
    "total_constituents": 50,
    "constituents_data": [
        {
            "symbol": "RELIANCE",
            "ltp": 2456.75,
            "perChange": 2.15,
            "high_52w": 2725.00,
            "low_52w": 1850.00,
            "volume": 15000000,
            "value": 36850000000
        }
    ],
    "stocks_at_52w_high": [
        {
            "symbol": "RELIANCE",
            "ltp": 2456.75,
            "perChange": 2.15,
            "high_52w": 2456.75,
            "low_52w": 1850.00,
            "volume": 15000000,
            "value": 36850000000
        }
    ],
    "stocks_at_52w_low": [],
    "high_count": 1,
    "low_count": 0,
    "analysis_time": "real-time"
}
```

### Get Available Indices
**Endpoint:** `GET /api/v1/indices/available`

**Description:** Get list of all available NSE indices with their categories.

**Response:**
```json
{
    "available_indices": [
        {
            "index_name": "NIFTY 50",
            "index_symbol": "NIFTY 50",
            "category": "INDICES ELIGIBLE IN DERIVATIVES"
        },
        {
            "index_name": "NIFTY BANK",
            "index_symbol": "NIFTY BANK",
            "category": "INDICES ELIGIBLE IN DERIVATIVES"
        }
    ],
    "description": "Use index_symbol with /constituents/{index_name} endpoint"
}
```

## Technical Analysis

### Calculate All Indicators
**Endpoint:** `POST /api/v1/technical/indicators`

**Request Body:**
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
    "stochastic": {
        "k": [65.5, 68.2],
        "d": [62.1, 66.8]
    },
    "williams_r": [-34.5, -31.8],
    "atr": [25.5, 26.2],
    "fibonacci": {
        "0%": 2490,
        "23.6%": 2478.2,
        "38.2%": 2470.9,
        "50%": 2465.0,
        "61.8%": 2459.1,
        "78.6%": 2451.4,
        "100%": 2440
    },
    "pivot_points": {
        "pivot": 2463.33,
        "r1": 2486.66,
        "r2": 2503.33,
        "r3": 2526.66,
        "s1": 2446.66,
        "s2": 2423.33,
        "s3": 2406.66
    },
    "signals": {
        "overall_signal": "BULLISH",
        "strength": 75,
        "bullish_signals": ["Price above Bollinger middle line", "MACD bullish crossover"],
        "bearish_signals": []
    }
}
```

### Fibonacci Levels
**Endpoint:** `GET /api/v1/technical/fibonacci/{symbol}`

**Parameters:**
- `period_days` (int, optional): Period for high/low calculation (default: 20)

### Pivot Points
**Endpoint:** `GET /api/v1/technical/pivot-points/{symbol}`

## Derivatives Trading

### Get Derivatives Equity Snapshot
**Endpoint:** `GET /api/v1/derivatives/equity-snapshot`

**Parameters:**
- `limit` (int, optional): Number of contracts to fetch (default: 20)

**Response:**
```json
{
  "volume": {
    "data": [
      {
        "identifier": "OPTIDXNIFTY30-09-2025PE24800.00",
        "instrumentType": "OPTIDX",
        "instrument": "Index Options",
        "underlying": "NIFTY",
        "expiryDate": "30-Sep-2025",
        "optionType": "Put",
        "strikePrice": 24800,
        "lastPrice": 145.45,
        "numberOfContractsTraded": 7562946,
        "totalTurnover": 483952.91454,
        "premiumTurnover": 141154748.51454,
        "openInterest": 69853,
        "underlyingValue": 24654.7,
        "pChange": 282.763157894737
      }
    ]
  }
}
```

### Get AI Derivatives Trading Calls
**Endpoint:** `GET /api/v1/derivatives/ai-trading-calls`

**Parameters:**
- `limit` (int, optional): Number of contracts to analyze (default: 20)

**Response:**
```json
{
  "analysis_timestamp": "2024-01-15T10:30:00",
  "total_contracts_analyzed": 20,
  "market_sentiment": "BULLISH",
  "market_analysis": {
    "total_calls": 12,
    "total_puts": 8,
    "bullish_signals": 15,
    "bearish_signals": 5,
    "call_put_ratio": 1.5
  },
  "top_ai_recommendations": [
    {
      "identifier": "OPTIDXNIFTY30-09-2025PE24800.00",
      "underlying": "NIFTY",
      "option_type": "Put",
      "strike_price": 24800,
      "expiry_date": "30-Sep-2025",
      "last_price": 145.45,
      "price_change_percent": 282.76,
      "volume": 7562946,
      "open_interest": 69853,
      "underlying_value": 24654.7,
      "ai_score": 85.5,
      "recommendation": "BUY",
      "trend": "BEARISH",
      "risk_level": "LOW",
      "signals": ["High Volume", "Strong Price Movement", "High Open Interest"]
    }
  ],
  "high_volume_opportunities": [...],
  "momentum_plays": [...],
  "trading_strategy": {
    "primary_focus": "High volume contracts with strong price movement",
    "risk_management": "Monitor open interest and underlying price movement",
    "market_outlook": "Current sentiment is bullish"
  }
}
```

## Enhanced AI Analysis

## Usage Examples

### Get Nifty 50 constituents with 52-week data:
```
GET /api/v1/indices/constituents/NIFTY%2050
```

### Get Bank Nifty constituents with 52-week data:
```
GET /api/v1/indices/constituents/NIFTY%20BANK
```

### Get all available indices:
```
GET /api/v1/indices/available
```

### Enhanced Market Analysis
**Endpoint:** `GET /api/v1/ai/enhanced-market-analysis`

**Parameters:**
- `analysis_type` (string, optional): "comprehensive" or "basic" (default: "comprehensive")
- `risk_assessment` (bool, optional): Include risk assessment (default: true)
- `limit` (int, optional): Number of stocks to analyze (default: 20)cks to analyze (default: 20)

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
    "market_insights": {
        "total_stocks": 20,
        "avg_change": 1.25,
        "volatility_stats": {
            "high_volatility_count": 5,
            "avg_volatility": 2.1
        },
        "volume_stats": {
            "high_volume_count": 8,
            "avg_volume": 12500000
        }
    },
    "risk_assessment": {
        "market_volatility": 2.45,
        "risk_level": "MEDIUM",
        "market_sentiment": "BULLISH",
        "recommendation": "Moderate volatility with bullish sentiment. Normal position sizing recommended."
    }
}
```

## Usage Examples

### WebSocket Connection
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = function() {
    // Subscribe to multiple symbols
    ws.send(JSON.stringify({action: "subscribe", symbol: "RELIANCE"}));
    ws.send(JSON.stringify({action: "subscribe", symbol: "TCS"}));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'market_update') {
        updateMarketData(data);
    }
};
```

### Create and Manage Portfolio
```bash
# Create portfolio
curl -X POST "http://localhost:8000/api/v1/portfolio/create" \
  -H "Content-Type: application/json" \
  -d '{"portfolio_id": "growth_001", "name": "Growth Portfolio", "capital": 1000000, "risk_tolerance": "MEDIUM"}'

# Add position
curl -X POST "http://localhost:8000/api/v1/portfolio/growth_001/add-position" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "RELIANCE", "quantity": 100, "entry_price": 2456.75, "stop_loss": 2300}'

# Get performance
curl "http://localhost:8000/api/v1/portfolio/growth_001/performance"
```

### Market Scanning
```bash
# Comprehensive market scan
curl "http://localhost:8000/api/v1/scanner/comprehensive"

# Breakout scanner
curl "http://localhost:8000/api/v1/scanner/breakouts?volume_threshold=2.0&price_change_threshold=4.0"

# Technical analysis
curl "http://localhost:8000/api/v1/technical/fibonacci/RELIANCE"
```

## API Rate Limits

- **WebSocket connections**: 10 concurrent connections per IP
- **Portfolio endpoints**: 100 requests per minute
- **Scanner endpoints**: 50 requests per minute
- **Technical analysis**: 200 requests per minute
- **Risk management**: 100 requests per minute

## Error Handling

### Standard Error Response
```json
{
    "error": "Portfolio not found",
    "code": "PORTFOLIO_NOT_FOUND",
    "timestamp": "2024-01-15T10:30:00Z",
    "suggestion": "Create portfolio first using /api/v1/portfolio/create"
}
```

### WebSocket Error Messages
```json
{
    "type": "error",
    "message": "Invalid symbol for subscription",
    "code": "INVALID_SYMBOL"
}
```

## Security & Best Practices

### API Security
- Use HTTPS in production
- Implement API key authentication
- Rate limiting per user/IP
- Input validation and sanitization

### Trading Best Practices
1. **Risk Management**: Never risk more than 2% per trade
2. **Position Sizing**: Use the risk management endpoints for optimal sizing
3. **Diversification**: Monitor correlation risk in portfolios
4. **Stop Losses**: Always set stop losses for positions
5. **Market Conditions**: Adjust strategies based on market risk assessment

### Performance Optimization
- Use WebSocket for real-time data instead of polling
- Cache frequently accessed portfolio data
- Batch multiple symbol requests when possible
- Use appropriate scan parameters to limit response size

## Support & Documentation

- **API Status**: http://localhost:8000/
- **Interactive Docs**: http://localhost:8000/docs
- **Technical Support**: Available for enterprise users
- **GitHub**: Repository with examples and SDKs

---

*MyAlgoFax NSE Market Data API v3.0 - Professional trading tools powered by AI and machine learning. Built for traders, by traders.*

© 2024 MyAlgoFax. All rights reserved.