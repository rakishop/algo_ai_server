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

# WebSocket Info Endpoint (shows in docs)
@router.get("/api/v1/websocket/info")
def websocket_info():
    """Get WebSocket connection information and usage guide"""
    return {
        "websocket_url": "ws://localhost:8000/ws",
        "status": "active",
        "connection_guide": {
            "url": "ws://localhost:8000/ws",
            "protocol": "WebSocket",
            "message_format": "JSON"
        },
        "subscribe_example": {
            "action": "subscribe",
            "symbol": "RELIANCE"
        },
        "response_types": [
            "subscription_confirmed",
            "market_update",
            "symbol_update"
        ],
        "javascript_example": "const ws = new WebSocket('ws://localhost:8000/ws'); ws.send(JSON.stringify({action: 'subscribe', symbol: 'RELIANCE'}));",
        "active_connections": len(manager.active_connections),
        "subscriptions": len(manager.subscriptions)
    }

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
    """Assess market risk with technical analysis for key stocks"""
    try:
        # Get scanner results for better risk assessment
        scanner_results = market_scanner.comprehensive_market_scan()
        all_stocks = []
        for category in scanner_results.get('opportunities', {}).values():
            all_stocks.extend(category)
        
        # Add technical analysis to high-risk stocks
        high_risk_stocks = [s for s in all_stocks if abs(s.get('perChange', 0)) > 3][:5]
        for stock in high_risk_stocks:
            high = stock.get('high_price', stock.get('ltp', 0))
            low = stock.get('low_price', stock.get('ltp', 0))
            current = stock.get('ltp', 0)
            
            if high and low and current and float(high) > float(low):
                stock['fibonacci_levels'] = indicators.fibonacci_retracement(float(high), float(low))
                stock['pivot_points'] = indicators.pivot_points(float(high), float(low), float(current))
        
        risk_assessment = risk_manager.assess_market_risk(all_stocks)
        risk_assessment['high_risk_stocks_with_technicals'] = high_risk_stocks
        
        return risk_assessment
    except Exception as e:
        return {"error": str(e)}

@router.get("/api/v1/ai/enhanced-market-analysis")
def enhanced_market_analysis(
    analysis_type: str = "comprehensive",
    risk_assessment: bool = True,
    limit: int = 20
):
    """Enhanced market analysis with technical analysis and risk assessment"""
    try:
        # Get scanner results with technical analysis
        scanner_results = market_scanner.comprehensive_market_scan()
        all_stocks = []
        for category in scanner_results.get('opportunities', {}).values():
            all_stocks.extend(category)
        
        # Add technical analysis to top stocks
        top_stocks = sorted(all_stocks, key=lambda x: x.get('breakout_score', x.get('momentum_score', x.get('reversal_score', 0))), reverse=True)[:limit]
        
        for stock in top_stocks:
            high = stock.get('high_price', stock.get('ltp', 0))
            low = stock.get('low_price', stock.get('ltp', 0))
            current = stock.get('ltp', 0)
            
            if high and low and current and float(high) > float(low):
                stock['fibonacci_levels'] = indicators.fibonacci_retracement(float(high), float(low))
                stock['pivot_points'] = indicators.pivot_points(float(high), float(low), float(current))
        
        result = {
            "analysis_type": analysis_type,
            "total_stocks_analyzed": len(all_stocks),
            "top_opportunities": top_stocks[:10],
            "market_insights": scanner_results.get('market_overview', {})
        }
        
        if risk_assessment:
            result["risk_assessment"] = risk_manager.assess_market_risk(all_stocks)
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

# Market Scanner Endpoints
@router.get("/api/v1/scanner/breakouts")
def scan_breakout_stocks(
    volume_threshold: float = 1.5,
    price_change_threshold: float = 3.0
):
    """Scan for breakout stocks with technical analysis included"""
    result = market_scanner.scan_breakout_stocks(volume_threshold, price_change_threshold)
    
    # Add technical analysis to each stock
    for stock in result.get('breakout_opportunities', []):
        high = stock.get('high_price', stock.get('dayHigh', stock.get('ltp', 0)))
        low = stock.get('low_price', stock.get('dayLow', stock.get('ltp', 0)))
        current = stock.get('ltp', 0)
        
        # If OHLC data is missing or invalid, use current price with small range
        if not high or not low or not current or float(high) == 0 or float(low) == 0 or float(high) <= float(low):
            current_price = float(current) if current else 100
            high = current_price * 1.02  # 2% above current
            low = current_price * 0.98   # 2% below current
        
        if high and low and current:
            stock['fibonacci_levels'] = indicators.fibonacci_retracement(float(high), float(low))
            stock['pivot_points'] = indicators.pivot_points(float(high), float(low), float(current))
    
    return result

@router.get("/api/v1/scanner/momentum")
def scan_momentum_stocks(
    rsi_min: float = 30,
    rsi_max: float = 70,
    min_volume: int = 1000000
):
    """Scan for momentum stocks with technical analysis included"""
    result = market_scanner.scan_momentum_stocks((rsi_min, rsi_max), min_volume)
    
    # Add technical analysis to each stock
    for stock in result.get('momentum_opportunities', []):
        high = stock.get('high_price', stock.get('ltp', 0))
        low = stock.get('low_price', stock.get('ltp', 0))
        current = stock.get('ltp', 0)
        
        if high and low and current and float(high) > float(low):
            stock['fibonacci_levels'] = indicators.fibonacci_retracement(float(high), float(low))
            stock['pivot_points'] = indicators.pivot_points(float(high), float(low), float(current))
    
    return result

@router.get("/api/v1/scanner/reversals")
def scan_reversal_candidates(
    oversold_threshold: float = -5.0,
    volume_spike: float = 2.0
):
    """Scan for reversal candidates with technical analysis included"""
    result = market_scanner.scan_reversal_candidates(oversold_threshold, volume_spike)
    
    # Add technical analysis to each stock
    for stock in result.get('reversal_candidates', []):
        high = stock.get('high_price', stock.get('ltp', 0))
        low = stock.get('low_price', stock.get('ltp', 0))
        current = stock.get('ltp', 0)
        
        if high and low and current and float(high) > float(low):
            stock['fibonacci_levels'] = indicators.fibonacci_retracement(float(high), float(low))
            stock['pivot_points'] = indicators.pivot_points(float(high), float(low), float(current))
    
    return result

@router.get("/api/v1/scanner/gaps")
def scan_gap_opportunities(min_gap_percent: float = 2.0):
    """Scan for gap opportunities with technical analysis included"""
    result = market_scanner.scan_gap_opportunities(min_gap_percent)
    
    # Add technical analysis to each stock
    for stock in result.get('gap_opportunities', []):
        high = stock.get('high_price', stock.get('ltp', 0))
        low = stock.get('low_price', stock.get('ltp', 0))
        current = stock.get('ltp', 0)
        
        if high and low and current and float(high) > float(low):
            stock['fibonacci_levels'] = indicators.fibonacci_retracement(float(high), float(low))
            stock['pivot_points'] = indicators.pivot_points(float(high), float(low), float(current))
    
    return result

@router.get("/api/v1/scanner/comprehensive")
def comprehensive_market_scan():
    """Perform comprehensive market scan with technical analysis for each stock"""
    try:
        base_results = market_scanner.comprehensive_market_scan()
        
        # Add technical analysis for each stock
        for category_name, stocks in base_results.get('opportunities', {}).items():
            for stock in stocks:
                symbol = stock.get('symbol')
                high = stock.get('high_price', stock.get('dayHigh', stock.get('ltp', 0)))
                low = stock.get('low_price', stock.get('dayLow', stock.get('ltp', 0)))
                current = stock.get('ltp', 0)
                
                # If OHLC data is missing or invalid, use current price with small range
                if not high or not low or not current or float(high) == 0 or float(low) == 0 or float(high) <= float(low):
                    current_price = float(current) if current else 100
                    high = current_price * 1.02  # 2% above current
                    low = current_price * 0.98   # 2% below current
                
                if high and low and current and symbol:
                    # Add fibonacci levels
                    fib_levels = indicators.fibonacci_retracement(float(high), float(low))
                    stock['fibonacci_levels'] = fib_levels
                    
                    # Add pivot points
                    pivot_data = indicators.pivot_points(float(high), float(low), float(current))
                    stock['pivot_points'] = pivot_data
                    
                    # Add support/resistance levels
                    stock['technical_levels'] = {
                        'support_1': pivot_data.get('S1', 0),
                        'support_2': pivot_data.get('S2', 0),
                        'resistance_1': pivot_data.get('R1', 0),
                        'resistance_2': pivot_data.get('R2', 0),
                        'pivot': pivot_data.get('PP', 0)
                    }
        
        base_results['enhanced_with_technical_analysis'] = True
        return base_results
        
    except Exception as e:
        return {"error": str(e)}

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
        symbol = symbol.upper()
        
        # If symbol is 'AI' or 'SCANNER', return fibonacci for all AI recommended stocks
        if symbol in ['AI', 'SCANNER']:
            scanner_results = market_scanner.comprehensive_market_scan()
            all_stocks = []
            for category in scanner_results.get('opportunities', {}).values():
                all_stocks.extend(category)
            
            fibonacci_results = []
            for stock in all_stocks[:10]:  # Limit to top 10
                stock_symbol = stock.get('symbol')
                if stock_symbol:
                    high = stock.get('high_price', stock.get('ltp', 0))
                    low = stock.get('low_price', stock.get('ltp', 0))
                    current = stock.get('ltp', 0)
                    
                    if high and low and current:
                        fib_levels = indicators.fibonacci_retracement(float(high), float(low))
                        fibonacci_results.append({
                            "symbol": stock_symbol,
                            "fibonacci_levels": fib_levels,
                            "current_price": float(current),
                            "scanner_type": stock.get('reversal_signal', stock.get('breakout_type', stock.get('momentum_type', 'AI_RECOMMENDED')))
                        })
            
            return {
                "analysis_type": "AI Scanner Fibonacci Analysis",
                "total_stocks": len(fibonacci_results),
                "fibonacci_analysis": fibonacci_results
            }
        
        # For indices, get real data from NSE
        if symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY']:
            indices_data = nse_client.get_all_indices()
            if "data" in indices_data:
                for index in indices_data["data"]:
                    index_name = index.get("index", "")
                    if (symbol == "NIFTY" and index_name == "NIFTY 50") or \
                       (symbol == "BANKNIFTY" and index_name == "NIFTY BANK") or \
                       (symbol == "FINNIFTY" and index_name == "NIFTY FINANCIAL SERVICES") or \
                       index_name == symbol:
                        current = float(index.get("last", 0))
                        high = float(index.get("high", current))
                        low = float(index.get("low", current))
                        
                        if high > low:
                            fib_levels = indicators.fibonacci_retracement(high, low)
                            return {
                                "symbol": symbol,
                                "period_high": high,
                                "period_low": low,
                                "fibonacci_levels": fib_levels,
                                "current_price": current,
                                "change": float(index.get("variation", 0)),
                                "percent_change": float(index.get("percentChange", 0))
                            }
            return {"error": f"Real-time data not available for {symbol}"}
        
        # For stocks, get from scanner results first, then fallback to market data
        scanner_results = market_scanner.comprehensive_market_scan()
        all_scanner_stocks = []
        for category in scanner_results.get('opportunities', {}).values():
            all_scanner_stocks.extend(category)
        
        # Check scanner results first
        for stock in all_scanner_stocks:
            if stock.get('symbol') == symbol:
                high = stock.get('high_price', stock.get('ltp', 0))
                low = stock.get('low_price', stock.get('ltp', 0))
                current = stock.get('ltp', 0)
                
                if high and low and float(high) > float(low):
                    fib_levels = indicators.fibonacci_retracement(float(high), float(low))
                    return {
                        "symbol": symbol,
                        "period_high": float(high),
                        "period_low": float(low),
                        "fibonacci_levels": fib_levels,
                        "current_price": float(current),
                        "price_change": float(stock.get("net_price", 0)),
                        "price_change_percent": float(stock.get("perChange", 0)),
                        "scanner_type": stock.get('reversal_signal', stock.get('breakout_type', stock.get('momentum_type', 'SCANNER_STOCK')))
                    }
        
        # Fallback to market data sources
        data_sources = [nse_client.get_most_active_securities(), nse_client.get_gainers_data(), nse_client.get_losers_data()]
        for data_source in data_sources:
            if "data" in data_source:
                for stock in data_source["data"]:
                    if stock.get("symbol") == symbol:
                        high = stock.get("dayHigh") or stock.get("high_price") or stock.get("ltp", 0)
                        low = stock.get("dayLow") or stock.get("low_price") or stock.get("ltp", 0)
                        current = stock.get("ltp") or stock.get("lastPrice", 0)
                        
                        if high and low and float(high) > float(low):
                            fib_levels = indicators.fibonacci_retracement(float(high), float(low))
                            return {
                                "symbol": symbol,
                                "period_high": float(high),
                                "period_low": float(low),
                                "fibonacci_levels": fib_levels,
                                "current_price": float(current),
                                "price_change": float(stock.get("change", 0)),
                                "price_change_percent": float(stock.get("perChange", stock.get("pChange", 0)))
                            }
        
        # If symbol not found, return available symbols for debugging
        available_symbols = []
        for data_source in data_sources:
            if "data" in data_source:
                available_symbols.extend([s.get("symbol", "") for s in data_source["data"][:5]])
        
        return {
            "error": f"Symbol {symbol} not found in market data",
            "available_symbols": list(set(available_symbols)),
            "suggestion": "Try using RELIANCE, TCS, HDFCBANK, or check if markets are open"
        }
        
    except Exception as e:
        return {"error": str(e)}

@router.get("/api/v1/technical/pivot-points/{symbol}")
def get_pivot_points(symbol: str):
    """Get pivot points and support/resistance levels for a symbol"""
    try:
        symbol = symbol.upper()
        
        # If symbol is 'AI' or 'SCANNER', return pivot points for all AI recommended stocks
        if symbol in ['AI', 'SCANNER']:
            scanner_results = market_scanner.comprehensive_market_scan()
            all_stocks = []
            for category in scanner_results.get('opportunities', {}).values():
                all_stocks.extend(category)
            
            pivot_results = []
            for stock in all_stocks[:10]:  # Limit to top 10
                stock_symbol = stock.get('symbol')
                if stock_symbol:
                    high = stock.get('high_price', stock.get('ltp', 0))
                    low = stock.get('low_price', stock.get('ltp', 0))
                    current = stock.get('ltp', 0)
                    
                    if high and low and current:
                        pivot_data = indicators.pivot_points(float(high), float(low), float(current))
                        pivot_results.append({
                            "symbol": stock_symbol,
                            "current_price": float(current),
                            "pivot_points": pivot_data,
                            "scanner_type": stock.get('reversal_signal', stock.get('breakout_type', stock.get('momentum_type', 'AI_RECOMMENDED'))),
                            "day_range": {
                                "high": float(high),
                                "low": float(low),
                                "range_percent": ((float(high) - float(low)) / float(low)) * 100
                            }
                        })
            
            return {
                "analysis_type": "AI Scanner Pivot Points Analysis",
                "total_stocks": len(pivot_results),
                "pivot_analysis": pivot_results
            }
        
        # For indices, get real data from NSE
        if symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY']:
            indices_data = nse_client.get_all_indices()
            if "data" in indices_data:
                for index in indices_data["data"]:
                    index_name = index.get("index", "")
                    if (symbol == "NIFTY" and index_name == "NIFTY 50") or \
                       (symbol == "BANKNIFTY" and index_name == "NIFTY BANK") or \
                       (symbol == "FINNIFTY" and index_name == "NIFTY FINANCIAL SERVICES") or \
                       index_name == symbol:
                        current = float(index.get("last", 0))
                        high = float(index.get("high", current))
                        low = float(index.get("low", current))
                        
                        if high > low:
                            pivot_data = indicators.pivot_points(high, low, current)
                            return {
                                "symbol": symbol,
                                "current_price": current,
                                "pivot_points": pivot_data,
                                "day_range": {
                                    "high": high,
                                    "low": low,
                                    "range_percent": ((high - low) / low) * 100
                                },
                                "change": float(index.get("variation", 0)),
                                "percent_change": float(index.get("percentChange", 0))
                            }
            return {"error": f"Real-time data not available for {symbol}"}
        
        # For stocks, get from scanner results first, then fallback to market data
        scanner_results = market_scanner.comprehensive_market_scan()
        all_scanner_stocks = []
        for category in scanner_results.get('opportunities', {}).values():
            all_scanner_stocks.extend(category)
        
        # Check scanner results first
        for stock in all_scanner_stocks:
            if stock.get('symbol') == symbol:
                high = stock.get('high_price', stock.get('ltp', 0))
                low = stock.get('low_price', stock.get('ltp', 0))
                current = stock.get('ltp', 0)
                
                if high and low and current and float(high) > float(low):
                    pivot_data = indicators.pivot_points(float(high), float(low), float(current))
                    return {
                        "symbol": symbol,
                        "current_price": float(current),
                        "pivot_points": pivot_data,
                        "day_range": {
                            "high": float(high),
                            "low": float(low),
                            "range_percent": ((float(high) - float(low)) / float(low)) * 100
                        },
                        "price_change": float(stock.get("net_price", 0)),
                        "price_change_percent": float(stock.get("perChange", 0)),
                        "scanner_type": stock.get('reversal_signal', stock.get('breakout_type', stock.get('momentum_type', 'SCANNER_STOCK')))
                    }
        
        # Fallback to market data sources
        data_sources = [nse_client.get_most_active_securities(), nse_client.get_gainers_data(), nse_client.get_losers_data()]
        for data_source in data_sources:
            if "data" in data_source:
                for stock in data_source["data"]:
                    if stock.get("symbol") == symbol:
                        high = stock.get("dayHigh") or stock.get("high_price") or stock.get("ltp", 0)
                        low = stock.get("dayLow") or stock.get("low_price") or stock.get("ltp", 0)
                        current = stock.get("ltp") or stock.get("lastPrice", 0)
                        
                        if high and low and current and float(high) > float(low):
                            pivot_data = indicators.pivot_points(float(high), float(low), float(current))
                            return {
                                "symbol": symbol,
                                "current_price": float(current),
                                "pivot_points": pivot_data,
                                "day_range": {
                                    "high": float(high),
                                    "low": float(low),
                                    "range_percent": ((float(high) - float(low)) / float(low)) * 100
                                },
                                "price_change": float(stock.get("change", 0)),
                                "price_change_percent": float(stock.get("perChange", stock.get("pChange", 0)))
                            }
        
        return {"error": f"Symbol {symbol} not found in market data"}
        
    except Exception as e:
        return {"error": str(e)}

# WebSocket Test Endpoint
@router.get("/api/v1/websocket/test")
def test_websocket_connection():
    """Test WebSocket connection status"""
    return {
        "websocket_endpoint": "/ws",
        "status": "available",
        "active_connections": len(manager.active_connections),
        "total_subscriptions": len(manager.subscriptions),
        "test_instructions": {
            "step1": "Connect to ws://localhost:8000/ws",
            "step2": "Send: {\"action\": \"subscribe\", \"symbol\": \"RELIANCE\"}",
            "step3": "Receive real-time market updates"
        }
    }