# NSE Market Data API Documentation

## Base URL
```
http://localhost:8000
```

## Authentication
No authentication required for current endpoints.

---

## Market Data Endpoints

### 1. 52-Week Extremes
**GET** `/api/v1/market/52-week-extremes`

Returns stocks hitting 52-week highs and lows along with analysis data.

**Response:**
```json
{
  "52_week_high": {...},
  "52_week_low": {...},
  "analysis_api": {...}
}
```

### 2. Daily Market Movers
**GET** `/api/v1/market/daily-movers`

Returns top gainers and losers for the day.

**Response:**
```json
{
  "gainers": {...},
  "losers": {...}
}
```

### 3. Market Activity Summary
**GET** `/api/v1/market/activity-summary`

Returns comprehensive market activity including most active securities, SME, security gainers, and volume gainers.

**Response:**
```json
{
  "most_active_securities": {...},
  "most_active_sme": {...},
  "security_gainers": {...},
  "volume_gainers": {...}
}
```

### 4. Price Band Hits
**GET** `/api/v1/market/price-band-hits`

Returns stocks that hit upper or lower price bands.

### 5. Volume Leaders
**GET** `/api/v1/market/volume-leaders`

Returns stocks with highest volume gains.

### 6. Market Breadth Indicators
**GET** `/api/v1/market/breadth-indicators`

Returns advance, decline, and unchanged stock counts.

**Response:**
```json
{
  "advance": {...},
  "decline": {...},
  "unchanged": {...}
}
```

### 7. Trading Statistics
**GET** `/api/v1/market/trading-statistics`

Returns overall market trading statistics.

### 8. Block Deals
**GET** `/api/v1/market/block-deals`

Returns large block deals executed in the market.

---

## Derivatives Endpoints

### 1. Market Snapshot
**GET** `/api/v1/derivatives/market-snapshot`

Returns derivatives market snapshot with top 20 contracts.

### 2. Active Underlyings
**GET** `/api/v1/derivatives/active-underlyings`

Returns most active underlying assets in derivatives market.

### 3. Open Interest Spurts
**GET** `/api/v1/derivatives/open-interest-spurts`

Returns open interest spurts for underlyings and contracts.

**Response:**
```json
{
  "underlyings": {...},
  "contracts": {...}
}
```

---

## AI Trading Analysis Endpoints

### 1. All Stocks Analysis
**GET** `/api/v1/ai/all-stocks-analysis`

Analyzes all NSE traded stocks with AI-powered chart analysis for trading decisions.

**Parameters:**
- `chartPeriod` (string): "I" (Intraday), "D" (Daily), "W" (Weekly), "M" (Monthly). Default: "I"
- `timeInterval` (int): Minutes for intraday (1,3,5,15,30,60) or 1 for D/W/M. Default: 15
- `minConfidence` (int): Minimum confidence threshold (0-100). Default: 60
- `maxResults` (int): Maximum stocks to analyze. Default: 100

**Example:**
```
GET /api/v1/ai/all-stocks-analysis?chartPeriod=I&timeInterval=5&minConfidence=70&maxResults=150
```

**Response:**
```json
{
  "total_stocks_available": 3130,
  "total_analyzed": 100,
  "total_decisions": 85,
  "chart_period": "I",
  "time_interval": 15,
  "min_confidence": 60,
  "buy_recommendations": [...],
  "sell_recommendations": [...],
  "summary": {
    "buy_count": 45,
    "sell_count": 25,
    "avg_confidence": 72.5
  }
}
```

### 2. Single Stock Decision
**POST** `/api/v1/ai/trade-decision`

AI analysis for individual stock trading decision.

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

### 3. Multi-Stock Analysis
**POST** `/api/v1/ai/multi-stock-decision`

Analyze multiple stocks and rank by confidence.

### 4. Daily Movers Analysis
**GET** `/api/v1/ai/daily-movers-analysis`

Analyzes gainers, losers, and most active stocks with AI.

### 5. Time-Based Opportunities
**GET** `/api/v1/ai/time-based-opportunities`

Find trading opportunities based on time intervals.

### 6. Chart Data
**POST** `/api/v1/charting/data`

Get NSE charting data for technical analysis.

### 7. Available Symbols
**GET** `/api/v1/charting/symbols`

Get list of available symbols for analysis.

---

## Indices Endpoints

### 1. Live Indices Data
**GET** `/api/v1/indices/live-data`

Returns live data for all NSE indices.

---

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "error": "Error description",
  "status_code": 500,
  "data": null
}
```

## Rate Limiting
- No rate limiting currently implemented
- Recommended: Max 100 requests per minute per IP

## Data Freshness
- All data is fetched live from NSE APIs
- Data refresh frequency depends on NSE's update cycle
- Market hours: 9:15 AM to 3:30 PM IST (Monday to Friday)

## Usage Examples

### cURL
```bash
# Get 52-week extremes
curl -X GET "http://localhost:8000/api/v1/market/52-week-extremes"

# Get daily movers
curl -X GET "http://localhost:8000/api/v1/market/daily-movers"
```

### Python
```python
import requests

# Get market activity summary
response = requests.get("http://localhost:8000/api/v1/market/activity-summary")
data = response.json()
```

### JavaScript
```javascript
// Get live indices data
fetch('http://localhost:8000/api/v1/indices/live-data')
  .then(response => response.json())
  .then(data => console.log(data));
```

## Support
For technical support or feature requests, contact the development team.