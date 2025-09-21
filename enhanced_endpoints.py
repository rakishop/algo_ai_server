from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from websocket_streaming import manager
from portfolio_manager import PortfolioManager
from risk_manager import RiskManager
from nse_client import NSEClient
from ml_analyzer import MLStockAnalyzer
from market_scanner import MarketScanner
from advanced_indicators import AdvancedTechnicalIndicators
import json
import asyncio

router = APIRouter()
portfolio_manager = PortfolioManager()
risk_manager = RiskManager()
nse_client = NSEClient()
ml_analyzer = MLStockAnalyzer()
market_scanner = MarketScanner()
indicators = AdvancedTechnicalIndicators()

# WebSocket endpoint for real-time streaming
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("action") == "subscribe":
                symbol = message.get("symbol")
                if symbol:
                    await manager.subscribe_symbol(websocket, symbol)
                    await websocket.send_text(json.dumps({
                        "type": "subscription_confirmed",
                        "symbol": symbol
                    }))
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Portfolio Management Endpoints
@router.post("/api/v1/portfolio/create")
def create_portfolio(portfolio_data: dict):
    """Create a new portfolio"""
    return portfolio_manager.create_portfolio(
        portfolio_data.get("portfolio_id"),
        portfolio_data.get("name"),
        portfolio_data.get("capital"),
        portfolio_data.get("risk_tolerance", "MEDIUM")
    )

@router.post("/api/v1/portfolio/{portfolio_id}/add-position")
def add_position(portfolio_id: str, position_data: dict):
    """Add a position to portfolio"""
    return portfolio_manager.add_position(
        portfolio_id,
        position_data.get("symbol"),
        position_data.get("quantity"),
        position_data.get("entry_price"),
        position_data.get("position_type", "LONG"),
        position_data.get("stop_loss"),
        position_data.get("target_price")
    )

@router.get("/api/v1/portfolio/{portfolio_id}/performance")
def get_portfolio_performance(portfolio_id: str):
    """Get portfolio performance with current market prices"""
    try:
        # Get current market data for price updates
        active_data = nse_client.get_most_active_securities()
        current_prices = {}
        
        if "data" in active_data:
            for stock in active_data["data"]:
                symbol = stock.get("symbol")
                price = stock.get("ltp") or stock.get("lastPrice")
                if symbol and price:
                    current_prices[symbol] = float(price)
        
        return portfolio_manager.get_portfolio_performance(portfolio_id, current_prices)
    except Exception as e:
        return {"error": str(e)}

@router.get("/api/v1/portfolio/{portfolio_id}/recommendations")
def get_portfolio_recommendations(portfolio_id: str):
    """Get AI-powered portfolio recommendations"""
    try:
        # Get market data for recommendations
        active_data = nse_client.get_most_active_securities()
        market_stocks = []
        
        if "data" in active_data:
            for stock in active_data["data"]:
                features = ml_analyzer.extract_features(stock)
                market_stocks.append({**stock, **features})
        
        return portfolio_manager.get_portfolio_recommendations(portfolio_id, market_stocks)
    except Exception as e:
        return {"error": str(e)}

# Risk Management Endpoints
@router.post("/api/v1/risk/calculate-position-size")
def calculate_position_size(risk_data: dict):
    """Calculate optimal position size based on risk management"""
    return risk_manager.calculate_position_size(
        risk_data.get("portfolio_value"),
        risk_data.get("risk_per_trade", 0.02),
        risk_data.get("entry_price"),
        risk_data.get("stop_loss"),
        risk_data.get("risk_tolerance", "MEDIUM")
    )

@router.get("/api/v1/risk/market-assessment")
def assess_market_risk():
    """Assess current market risk conditions"""
    try:
        # Get comprehensive market data
        active_data = nse_client.get_most_active_securities()
        gainers_data = nse_client.get_gainers_data()
        losers_data = nse_client.get_losers_data()
        
        all_market_data = []
        for data_source in [active_data, gainers_data, losers_data]:
            if "data" in data_source:
                all_market_data.extend(data_source["data"])
        
        return risk_manager.assess_market_risk(all_market_data)
    except Exception as e:
        return {"error": str(e)}

@router.get("/api/v1/ai/enhanced-market-analysis")
def enhanced_market_analysis(
    analysis_type: str = "comprehensive",
    risk_assessment: bool = True,
    limit: int = 20
):
    """Enhanced market analysis with risk assessment"""
    try:
        # Get multiple data sources
        active_data = nse_client.get_most_active_securities()
        gainers_data = nse_client.get_gainers_data()
        volume_data = nse_client.get_volume_gainers()
        
        all_stocks = []
        for data_source in [active_data, gainers_data, volume_data]:
            if "data" in data_source:
                for stock in data_source["data"]:
                    features = ml_analyzer.extract_features(stock)
                    all_stocks.append({**stock, **features})
        
        # Remove duplicates
        unique_stocks = {}
        for stock in all_stocks:
            symbol = stock.get('symbol')
            if symbol and symbol not in unique_stocks:
                unique_stocks[symbol] = stock
        
        stocks_list = list(unique_stocks.values())[:limit]
        
        result = {
            "analysis_type": analysis_type,
            "total_stocks_analyzed": len(stocks_list),
            "top_opportunities": sorted(stocks_list, key=lambda x: x.get('scalping_score', 0), reverse=True)[:10],
            "market_insights": ml_analyzer.get_market_insights(stocks_list)
        }
        
        if risk_assessment:
            result["risk_assessment"] = risk_manager.assess_market_risk(stocks_list)
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

# Market Scanner Endpoints
@router.get("/api/v1/scanner/breakouts")
def scan_breakout_stocks(
    volume_threshold: float = 1.5,
    price_change_threshold: float = 3.0
):
    """Scan for breakout stocks with high volume and price movement"""
    return market_scanner.scan_breakout_stocks(volume_threshold, price_change_threshold)

@router.get("/api/v1/scanner/momentum")
def scan_momentum_stocks(
    rsi_min: float = 30,
    rsi_max: float = 70,
    min_volume: int = 1000000
):
    """Scan for momentum stocks with good RSI levels"""
    return market_scanner.scan_momentum_stocks((rsi_min, rsi_max), min_volume)

@router.get("/api/v1/scanner/reversals")
def scan_reversal_candidates(
    oversold_threshold: float = -5.0,
    volume_spike: float = 2.0
):
    """Scan for potential reversal candidates"""
    return market_scanner.scan_reversal_candidates(oversold_threshold, volume_spike)

@router.get("/api/v1/scanner/gaps")
def scan_gap_opportunities(min_gap_percent: float = 2.0):
    """Scan for gap up/down opportunities"""
    return market_scanner.scan_gap_opportunities(min_gap_percent)

@router.get("/api/v1/scanner/comprehensive")
def comprehensive_market_scan():
    """Perform comprehensive market scan with multiple strategies"""
    return market_scanner.comprehensive_market_scan()

# Technical Analysis Endpoints
@router.post("/api/v1/technical/indicators")
def calculate_technical_indicators(ohlcv_data: dict):
    """Calculate comprehensive technical indicators for OHLCV data"""
    try:
        data = ohlcv_data.get("data", [])
        if not data:
            return {"error": "No OHLCV data provided"}
        
        return indicators.calculate_all_indicators(data)
    except Exception as e:
        return {"error": str(e)}

@router.get("/api/v1/technical/fibonacci/{symbol}")
def get_fibonacci_levels(
    symbol: str,
    period_days: int = 20
):
    """Get Fibonacci retracement levels for a symbol"""
    try:
        # Get recent market data (simplified - in real implementation, get historical data)
        active_data = nse_client.get_most_active_securities()
        
        if "data" in active_data:
            for stock in active_data["data"]:
                if stock.get("symbol") == symbol.upper():
                    # Use day's high/low for Fibonacci calculation
                    high = stock.get("dayHigh") or stock.get("high_price", 0)
                    low = stock.get("dayLow") or stock.get("low_price", 0)
                    
                    if high and low:
                        fib_levels = indicators.fibonacci_retracement(float(high), float(low))
                        return {
                            "symbol": symbol.upper(),
                            "period_high": high,
                            "period_low": low,
                            "fibonacci_levels": fib_levels,
                            "current_price": stock.get("ltp") or stock.get("lastPrice", 0)
                        }
        
        return {"error": f"Symbol {symbol} not found in active data"}
        
    except Exception as e:
        return {"error": str(e)}

@router.get("/api/v1/technical/pivot-points/{symbol}")
def get_pivot_points(symbol: str):
    """Get pivot points and support/resistance levels for a symbol"""
    try:
        active_data = nse_client.get_most_active_securities()
        
        if "data" in active_data:
            for stock in active_data["data"]:
                if stock.get("symbol") == symbol.upper():
                    high = stock.get("dayHigh") or stock.get("high_price", 0)
                    low = stock.get("dayLow") or stock.get("low_price", 0)
                    close = stock.get("ltp") or stock.get("lastPrice", 0)
                    
                    if high and low and close:
                        pivot_data = indicators.pivot_points(float(high), float(low), float(close))
                        return {
                            "symbol": symbol.upper(),
                            "current_price": close,
                            "pivot_points": pivot_data,
                            "day_range": {
                                "high": high,
                                "low": low,
                                "range_percent": ((float(high) - float(low)) / float(low)) * 100
                            }
                        }
        
        return {"error": f"Symbol {symbol} not found in active data"}
        
    except Exception as e:
        return {"error": str(e)}