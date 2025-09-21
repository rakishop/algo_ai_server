# MyAlgoFax NSE Market Data API v2.1

## Overview

Advanced NSE market data API with AI-powered historical analysis, time-based decision systems, and multi-stock portfolio analysis. This API provides comprehensive trading tools with configurable time periods, intervals, and ML-driven insights for optimal entry/exit timing across different trading styles.

### New in v2.1
- **Time-Based Decision System**: Historical analysis with specific time periods and intervals
- **Multi-Stock Analysis**: Simultaneous analysis of multiple stocks with confidence ranking  
- **Flexible Time Intervals**: Support for scalping (1-5 min), swing (15-60 min), and position trading (4+ hours)
- **Enhanced ML Models**: Pandas DataFrame processing with advanced technical indicators
- **Market Timing Intelligence**: Automatic trading style recommendations based on time intervals

## Base URL
```
http://localhost:8000
```

## Enhanced Trade Decision Analysis

### Endpoint: `/api/v1/ai/trade-decision`
**Method:** POST

**Description:** AI-powered stock decision system with comprehensive historical analysis using specific time periods and intervals.

**Parameters:**
- `tradingSymbol` (string): Stock symbol to analyze
- `chartPeriod` (string, optional): Chart period (default: "I")
- `timeInterval` (int, optional): Time interval in minutes (default: 5)
- `lookbackPeriod` (int, optional): Historical data points to analyze (default: 50)
- `analysisDepth` (string, optional): Analysis depth - "comprehensive" or "basic" (default: "comprehensive")

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
  "time_analysis": {
    "short_term_trend": "UP",
    "medium_term_trend": "UP",
    "time_recommendation": "Suitable for swing trading",
    "trend_strength": "Strong"
  },
  "reasons": [
    "Strong upward trend - price above both SMAs",
    "High volume activity (1.3x average)",
    "RSI in healthy bullish zone"
  ],
  "analysis_time": "2024-01-15T10:30:00",
  "lookback_period": 50,
  "interval_minutes": 5,
  "recommendation": "BUY with 85% confidence based on comprehensive analysis"
}
```

### Endpoint: `/api/v1/ai/multi-stock-decision`
**Method:** POST

**Description:** Analyze multiple stocks simultaneously and rank by decision confidence.

**Parameters:**
- `symbols` (array): List of stock symbols to analyze
- `timeInterval` (int, optional): Time interval in minutes (default: 15)
- `lookbackPeriod` (int, optional): Historical data points (default: 50)
- `topN` (int, optional): Number of top recommendations to return (default: 10)

**Response:**
```json
{
  "analysis_summary": {
    "total_analyzed": 15,
    "buy_opportunities": 6,
    "sell_opportunities": 3,
    "hold_recommendations": 6
  },
  "top_buy_recommendations": [
    {
      "symbol": "RELIANCE",
      "decision": "BUY",
      "confidence": 87,
      "ml_score": 0.847,
      "current_price": 2456.75,
      "price_change_pct": 2.3,
      "rsi": 65.4,
      "volume_ratio": 1.28,
      "trend_strength": "Strong"
    }
  ],
  "top_sell_recommendations": [
    {
      "symbol": "HDFC",
      "decision": "SELL",
      "confidence": 82,
      "ml_score": 0.234,
      "current_price": 1567.30,
      "price_change_pct": -1.8,
      "rsi": 28.5,
      "volume_ratio": 1.45,
      "trend_strength": "Strong"
    }
  ],
  "market_overview": {
    "bullish_sentiment": 40.0,
    "bearish_sentiment": 20.0
  }
}
```

### Endpoint: `/api/v1/ai/time-based-opportunities`
**Method:** GET

**Description:** Find trading opportunities based on specific time intervals with confidence filtering.

**Parameters:**
- `timeInterval` (int, optional): Time interval in minutes (default: 15)
- `minConfidence` (int, optional): Minimum confidence threshold (default: 70)
- `maxSymbols` (int, optional): Maximum symbols to return (default: 15)

**Response:**
```json
{
  "time_interval_minutes": 15,
  "min_confidence_threshold": 70,
  "total_opportunities": 8,
  "high_confidence_trades": [
    {
      "symbol": "RELIANCE",
      "action": "BUY",
      "confidence": 87,
      "entry_price": 2456.75,
      "price_change": 2.3,
      "rsi": 65.4,
      "volume_strength": 1.28,
      "trend": "Strong",
      "time_frame": "15 minutes",
      "key_reason": "Strong upward trend - price above both SMAs"
    }
  ],
  "market_timing_analysis": {
    "optimal_for_scalping": false,
    "optimal_for_swing": true,
    "optimal_for_position": false,
    "analysis_timestamp": "2024-01-15T10:30:00"
  },
  "risk_assessment": {
    "high_confidence_count": 3,
    "medium_confidence_count": 5,
    "avg_confidence": 78.5
  }
}
```

## Enhanced Charting Data

### Endpoint: `/api/v1/charting/data`
**Method:** POST

**Description:** Get comprehensive OHLCV charting data with metadata for specific time periods and intervals.

**Parameters:**
- `tradingSymbol` (string): Stock symbol
- `chartPeriod` (string, optional): Chart period (default: "I")
- `timeInterval` (int, optional): Time interval in minutes (default: 3)
- `chartStart` (int, optional): Chart start timestamp (default: 0)
- `fromDate` (int, optional): From date timestamp (default: auto-calculated)
- `toDate` (int, optional): To date timestamp (default: current time)
- `dataPoints` (int, optional): Number of data points to return (default: 200)

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
    "interval_minutes": 3,
    "period": "I",
    "data_points": 200,
    "time_range": {
      "from": 1642200000,
      "to": 1642234920,
      "from_date": "2024-01-15T00:00:00",
      "to_date": "2024-01-15T10:30:00"
    }
  }
}
```

### Endpoint: `/api/v1/charting/symbols`
**Method:** GET

**Description:** Get list of available symbols for charting with current market data.

**Response:**
```json
{
  "status": "success",
  "symbols": [
    {
      "symbol": "RELIANCE",
      "name": "Reliance Industries Limited",
      "tradingSymbol": "RELIANCE-EQ",
      "current_price": 2456.75,
      "change_pct": 2.3
    }
  ],
  "total": 50,
  "supported_intervals": [1, 3, 5, 15, 30, 60, 240, 1440],
  "supported_periods": ["I", "D", "W", "M"]
}
```

## Usage Examples

### Get Enhanced Trade Decision
```bash
curl -X POST "http://localhost:8000/api/v1/ai/trade-decision" \
  -H "Content-Type: application/json" \
  -d '{"tradingSymbol": "RELIANCE", "timeInterval": 15, "lookbackPeriod": 100, "analysisDepth": "comprehensive"}'
```

### Analyze Multiple Stocks
```bash
curl -X POST "http://localhost:8000/api/v1/ai/multi-stock-decision" \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["RELIANCE", "TCS", "HDFC", "INFY"], "timeInterval": 30, "topN": 5}'
```

### Find Time-Based Opportunities
```bash
curl "http://localhost:8000/api/v1/ai/time-based-opportunities?timeInterval=15&minConfidence=75&maxSymbols=10"
```

### Get Historical Chart Data
```bash
curl -X POST "http://localhost:8000/api/v1/charting/data" \
  -H "Content-Type: application/json" \
  -d '{"tradingSymbol": "RELIANCE", "timeInterval": 5, "dataPoints": 100}'
```

## Time Intervals & Trading Strategies

### Scalping (1-5 minutes)
- **1 minute**: Ultra-high frequency trading
- **3 minutes**: Quick momentum captures
- **5 minutes**: Short-term price movements
- **Best for**: Day traders, high-volume strategies

### Swing Trading (15-60 minutes)
- **15 minutes**: Intraday swing setups
- **30 minutes**: Multi-session positions
- **60 minutes**: Daily trend following
- **Best for**: Part-time traders, trend followers

### Position Trading (4+ hours)
- **4 hours**: Weekly trend analysis
- **1 day**: Long-term investment decisions
- **1 week**: Monthly position sizing
- **Best for**: Long-term investors, low-frequency trading

### Lookback Periods
- **20-50 data points**: Short-term analysis
- **50-100 data points**: Medium-term trends
- **100-200 data points**: Long-term patterns
- **200+ data points**: Comprehensive historical analysis

## Technical Indicators & ML Features

### Built-in Technical Indicators
- **Moving Averages**: SMA5, SMA20 with ratio analysis
- **Momentum**: RSI (14-period) with overbought/oversold detection
- **Volatility**: Rolling standard deviation with percentage calculation
- **Volume**: Volume ratio analysis with spike detection
- **Trend**: Multi-timeframe trend strength analysis
- **Price Action**: Daily range analysis and price change patterns

### Machine Learning Features
- **Feature Engineering**: 8-dimensional feature vectors for ML models
- **Normalization**: StandardScaler for consistent feature scaling
- **Decision Scoring**: Sigmoid-based probability scoring (0-1 range)
- **Confidence Calculation**: ML score to confidence percentage conversion
- **Pattern Recognition**: Historical pattern matching with pandas DataFrame
- **Risk Assessment**: Volatility-adjusted confidence scoring

### Time-Based Analysis
- **Short-term Trends**: 5-period price movement analysis
- **Medium-term Trends**: 10-period trend confirmation
- **Lookback Flexibility**: 20-200 data points for historical context
- **Interval Optimization**: Automatic trading style recommendation based on time intervals

## Time-Based Trading Recommendations

### For Day Traders (1-15 minute intervals)
- Use `/api/v1/ai/time-based-opportunities` with `timeInterval=5`
- Set `minConfidence=75` for high-probability setups
- Monitor `/api/v1/ai/trade-decision` every 5-10 minutes
- Focus on volume spikes and RSI divergences

### For Swing Traders (15-60 minute intervals)
- Use `/api/v1/ai/multi-stock-decision` with `timeInterval=30`
- Set `lookbackPeriod=100` for trend confirmation
- Analyze multiple stocks simultaneously for diversification
- Focus on trend strength and SMA crossovers

### For Position Traders (4+ hour intervals)
- Use `/api/v1/ai/trade-decision` with `timeInterval=240`
- Set `lookbackPeriod=200` for comprehensive analysis
- Focus on long-term trend analysis and fundamental alignment
- Use higher confidence thresholds (80%+) for position sizing

## API Endpoints Summary

| Endpoint | Method | Description | Key Features |
|----------|--------|-------------|-------------|
| `/api/v1/ai/trade-decision` | POST | Enhanced ML trade decisions | Historical analysis with time periods |
| `/api/v1/ai/multi-stock-decision` | POST | Multi-stock analysis & ranking | Batch processing with confidence ranking |
| `/api/v1/ai/time-based-opportunities` | GET | Time-interval opportunity finder | Confidence filtering & market timing |
| `/api/v1/charting/data` | POST | Enhanced OHLCV data | Configurable time periods & intervals |
| `/api/v1/charting/symbols` | GET | Available symbols with prices | Real-time market data included |

## Performance Optimization

- **Caching**: Historical data cached for 1 minute
- **Batch Processing**: Multi-stock analysis optimized for parallel processing
- **Data Compression**: OHLCV data compressed for faster transmission
- **Smart Filtering**: Pre-filtering based on confidence thresholds to reduce response size

## Rate Limits

- **Standard endpoints**: 100 requests per minute
- **AI analysis endpoints**: 50 requests per minute
- **Multi-stock analysis**: 20 requests per minute (max 20 symbols per request)
- **Real-time charting**: 200 requests per minute
- **Time-based opportunities**: 30 requests per minute

## Error Handling

### Common Error Responses
```json
{
  "error": "Insufficient historical data for analysis",
  "code": "INSUFFICIENT_DATA",
  "timestamp": "2024-01-15T10:30:00Z",
  "details": "Need minimum 20 data points, got 8",
  "suggestion": "Reduce lookbackPeriod or increase timeInterval"
}
```

### ML Analysis Errors
```json
{
  "error": "ML model calculation failed",
  "code": "ML_ERROR",
  "fallback_decision": "HOLD",
  "confidence": 50,
  "reason": "Using basic technical analysis as fallback"
}
```

## Support & Best Practices

### For Technical Support
- Email: support@myalgofax.com
- Documentation: https://api.myalgofax.com/docs
- Status Page: https://status.myalgofax.com

### Best Practices for Time-Based Analysis
1. **Choose Appropriate Intervals**: Match time intervals to your trading style
2. **Use Sufficient Lookback**: Minimum 50 data points for reliable ML analysis
3. **Monitor Confidence Levels**: Use 70%+ confidence for high-probability trades
4. **Combine Multiple Timeframes**: Cross-verify decisions across different intervals
5. **Risk Management**: Never risk more than 2% per trade regardless of confidence
6. **Market Hours**: Best results during active trading hours (9:15 AM - 3:30 PM IST)

### Performance Tips
- Cache frequently accessed symbols for faster response times
- Use batch analysis (`multi-stock-decision`) for portfolio-wide decisions
- Set appropriate confidence thresholds to filter noise
- Monitor API rate limits for high-frequency applications

## License & Disclaimer

### License
This API is proprietary software. Unauthorized use, reproduction, or distribution is prohibited.

### Trading Disclaimer
**IMPORTANT**: This API provides analytical tools and should not be considered as financial advice. 
- All trading decisions should be made based on your own research and risk tolerance
- Past performance does not guarantee future results
- Machine learning predictions are probabilistic and may be incorrect
- Always use proper risk management and position sizing
- Consider consulting with a financial advisor for investment decisions

### Data Accuracy
- Real-time data may have delays of up to 1 minute
- Historical data is provided for analysis purposes only
- Technical indicators are calculated using standard formulas but may vary from other platforms
- ML models are continuously updated but may have periods of reduced accuracy

---

*This API provides comprehensive market analysis tools powered by artificial intelligence, machine learning algorithms, and time-based historical analysis. All data is sourced from NSE and processed with pandas DataFrame analysis for optimal trading decisions across multiple time horizons.*

© 2024 MyAlgoFax. All rights reserved.