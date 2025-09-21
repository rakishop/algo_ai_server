from fastapi import FastAPI
from nse_client import NSEClient
from filtered_endpoints import FilteredEndpoints
from ai_endpoints import AIEndpoints
from routes.market_routes import create_market_routes
from routes.derivatives_routes import create_derivatives_routes
from routes.indices_routes import create_indices_routes
from option_chain_endpoint import create_option_chain_routes
from charting_endpoint import create_charting_routes

app = FastAPI(title="NSE Market Data API", version="2.0.0", description="AI-Powered NSE market data with ML analysis and smart filtering")
nse_client = NSEClient()

# Create shared instances to avoid multiple instantiation
from data_processor import DataProcessor
from ml_analyzer import MLStockAnalyzer
from datetime import datetime
processor = DataProcessor()
ml_analyzer = MLStockAnalyzer()

# Initialize endpoints
filtered_endpoints = FilteredEndpoints(app, nse_client)
ai_endpoints = AIEndpoints(app, nse_client)

# Include route modules
app.include_router(create_market_routes(nse_client))
app.include_router(create_derivatives_routes(nse_client))
app.include_router(create_indices_routes(nse_client))
app.include_router(create_charting_routes(nse_client))
# Option chain routes integrated into main comprehensive analysis endpoint

@app.get("/")
def welcome():
    return {"message": "Hi, Welcome to Enterprise API"}

@app.get("/test")
def test_endpoint():
    return {"status": "working", "endpoints": ["scalping-analysis", "options-strategies"]}

@app.get("/api/v1/ai/scalping-test")
def get_scalping_analysis(
    volatility_threshold: float = 2.0,
    volume_threshold: int = 5000000,
    limit: int = 15
):
    try:
        active_data = nse_client.get_most_active_securities()
        volume_data = nse_client.get_volume_gainers()
        
        # Use shared instances
        all_stocks = []
        all_stocks.extend(processor.extract_stock_data(active_data))
        all_stocks.extend(processor.extract_stock_data(volume_data))
        
        scalping_stocks = []
        for stock in all_stocks:
            features = ml_analyzer.extract_features(stock)
            volume = stock.get('trade_quantity', 0)
            volatility = features.get('price_volatility', 0)
            
            # More lenient criteria
            liquidity_score = min(volume/1000000, 20) * 0.4
            volatility_score = min(volatility, 10) * 0.4
            price_score = min(stock.get('ltp', 0)/100, 10) * 0.2
            scalping_score = min((liquidity_score + volatility_score + price_score) * 5, 100)
            
            if volume >= volume_threshold/2 and volatility >= volatility_threshold/2 and scalping_score > 30:
                    scalping_stocks.append({
                        **stock,
                        **features,
                        'scalping_score': scalping_score,
                        'liquidity_rating': "Excellent" if volume > 20000000 else "Good" if volume > 10000000 else "Fair"
                    })
        
        scalping_stocks.sort(key=lambda x: x['scalping_score'], reverse=True)
        
        return {
            "scalping_opportunities": scalping_stocks[:limit],
            "volatility_threshold": volatility_threshold,
            "volume_threshold": volume_threshold,
            "analysis_method": "AI Scalping Analysis",
            "total_opportunities": len(scalping_stocks)
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/ai/options-analysis")
def analyze_options_comprehensive(
    symbol: str = "NIFTY",
    expiry: str = None,
    analysis_type: str = "comprehensive",
    strategy_type: str = "all",
    limit: int = 25
):
    try:
        from options_ai_analyzer import OptionsAIAnalyzer
        ai_analyzer = OptionsAIAnalyzer()
        
        # Normalize parameters (case-insensitive)
        symbol = symbol.upper()
        analysis_type = analysis_type.lower()
        strategy_type = strategy_type.lower()
        
        # Get both option chain and derivatives data
        result = {}
        
        # Option Chain Analysis (specific symbol/expiry)
        if analysis_type in ["comprehensive", "option_chain"]:
            chain_info = nse_client.get_option_chain_info(symbol)
            if "error" not in chain_info:
                if not expiry and "expiryDates" in chain_info:
                    expiry = chain_info["expiryDates"][0]
                
                chain_data = nse_client.get_option_chain(symbol, expiry)
                if "error" not in chain_data:
                    processed_chain = processor.extract_option_chain_data(chain_data)
                    chain_options = processed_chain["calls"] + processed_chain["puts"]
                    
                    # Calculate option chain specific metrics
                    calls = processed_chain["calls"]
                    puts = processed_chain["puts"]
                    
                    call_volume = sum(opt.get('totalTradedVolume', 0) for opt in calls)
                    put_volume = sum(opt.get('totalTradedVolume', 0) for opt in puts)
                    call_oi = sum(opt.get('openInterest', 0) for opt in calls)
                    put_oi = sum(opt.get('openInterest', 0) for opt in puts)
                    
                    # Max Pain calculation
                    strikes = sorted(set(opt.get('strikePrice', 0) for opt in chain_options))
                    max_pain_strike = 0
                    min_pain = float('inf')
                    
                    for strike in strikes:
                        call_pain = sum(max(0, strike - opt.get('strikePrice', 0)) * opt.get('openInterest', 0) 
                                      for opt in calls if opt.get('strikePrice', 0) < strike)
                        put_pain = sum(max(0, opt.get('strikePrice', 0) - strike) * opt.get('openInterest', 0) 
                                     for opt in puts if opt.get('strikePrice', 0) > strike)
                        total_pain = call_pain + put_pain
                        
                        if total_pain < min_pain:
                            min_pain = total_pain
                            max_pain_strike = strike
                    
                    result["option_chain_analysis"] = {
                        "symbol": symbol,
                        "expiry": expiry,
                        "underlying_value": processed_chain["underlying_value"],
                        "option_chain_metrics": {
                            "total_strikes": processed_chain["total_strikes"],
                            "call_volume": call_volume,
                            "put_volume": put_volume,
                            "volume_pcr": put_volume / call_volume if call_volume > 0 else 1,
                            "call_oi": call_oi,
                            "put_oi": put_oi,
                            "oi_pcr": put_oi / call_oi if call_oi > 0 else 1,
                            "max_pain_strike": max_pain_strike
                        },
                        "available_expiries": chain_info.get("expiryDates", [])[:5]
                    }
        
        # Market-wide derivatives analysis
        if analysis_type in ["comprehensive", "market_wide"]:
            derivatives_data = nse_client.get_derivatives_snapshot()
            options_data = processor.extract_derivatives_data(derivatives_data)
            
            # Extract advanced features for analysis
            features = ai_analyzer.extract_advanced_features(options_data)
            market_regime = ai_analyzer.analyze_market_regime(features)
            
            # Calculate traditional metrics
            calls = [opt for opt in options_data if opt.get('optionType') == 'Call']
            puts = [opt for opt in options_data if opt.get('optionType') == 'Put']
            
            call_volume = sum(opt.get('numberOfContractsTraded', 0) for opt in calls)
            put_volume = sum(opt.get('numberOfContractsTraded', 0) for opt in puts)
            pcr = put_volume / call_volume if call_volume > 0 else 1
            
            avg_call_change = sum(opt.get('pChange', 0) for opt in calls) / len(calls) if calls else 0
            avg_put_change = sum(opt.get('pChange', 0) for opt in puts) / len(puts) if puts else 0
            
            result["market_wide_analysis"] = {
                "market_regime_analysis": market_regime,
                "advanced_features": {
                    "pcr": features[0],
                    "implied_volatility_estimate": features[1],
                    "volume_ratio": features[2],
                    "avg_price_change": features[3],
                    "moneyness_ratio": features[5],
                    "volatility_skew": features[7]
                },
                "traditional_metrics": {
                    "pcr": pcr,
                    "call_volume": call_volume,
                    "put_volume": put_volume,
                    "avg_call_change": avg_call_change,
                    "avg_put_change": avg_put_change
                },
                "total_strikes_analyzed": len(options_data)
            }
        
        # AI Strategy Recommendations (use chain data if available, otherwise derivatives)
        options_for_ai = chain_options if 'chain_options' in locals() else options_data
        ai_strategies = ai_analyzer.predict_optimal_strategies(options_for_ai)
        
        result.update({
            "ai_recommended_strategies": ai_strategies,
            "strategy_filter": strategy_type,
            "analysis_method": "Comprehensive AI Options Analysis",
            "analysis_type": analysis_type
        })
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/ai/options-spike-detection")
def detect_options_spikes(
    symbol: str = "NIFTY",
    time_period: int = 5,
    spike_threshold: float = 20.0,
    auto_store: bool = True,
    limit: int = 20
):
    """Detect spikes over time periods with historical comparison"""
    try:
        from spike_tracker import SpikeTracker
        tracker = SpikeTracker()
        symbol = symbol.upper()
        
        # Get current option chain data
        chain_info = nse_client.get_option_chain_info(symbol)
        if "error" in chain_info:
            return chain_info
        
        expiry = chain_info["expiryDates"][0] if "expiryDates" in chain_info else None
        chain_data = nse_client.get_option_chain(symbol, expiry)
        if "error" in chain_data:
            return chain_data
        
        processed_chain = processor.extract_option_chain_data(chain_data)
        all_options = processed_chain["calls"] + processed_chain["puts"]
        
        # Store current snapshot if auto_store enabled
        if auto_store:
            tracker.store_snapshot(symbol, all_options)
        
        # Detect period-based spikes
        period_spikes = tracker.detect_period_spikes(symbol, time_period)
        
        # Current moment spikes (for immediate alerts)
        current_spikes = []
        for option in all_options:
            price_change = abs(option.get('pchange', 0))
            volume = option.get('totalTradedVolume', 0)
            
            if price_change >= spike_threshold:
                current_spikes.append({
                    "strike": option.get('strikePrice', 0),
                    "option_type": option.get('optionType', 'Unknown'),
                    "price_change": price_change,
                    "volume": volume,
                    "last_price": option.get('lastPrice', 0),
                    "bid_price": option.get('bidprice', 0),
                    "ask_price": option.get('askPrice', 0),
                    "open_interest": option.get('openInterest', 0),
                    "oi_change": option.get('changeinOpenInterest', 0),
                    "implied_volatility": option.get('impliedVolatility', 0),
                    "spike_type": "Current Price Spike"
                })
        
        return {
            "symbol": symbol,
            "expiry": expiry,
            "time_period_minutes": time_period,
            "period_spikes": period_spikes[:limit//2],
            "current_spikes": current_spikes[:limit//2],
            "total_period_spikes": len(period_spikes),
            "total_current_spikes": len(current_spikes),
            "spike_threshold": spike_threshold,
            "analysis_method": "Time-based Spike Detection",
            "underlying_value": processed_chain["underlying_value"],
            "refresh_note": "Call this endpoint every minute to track spikes over time"
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/ai/refresh-spike-data")
def refresh_spike_data(symbol: str = "NIFTY"):
    """Manually refresh and store current options data for spike tracking"""
    try:
        from spike_tracker import SpikeTracker
        tracker = SpikeTracker()
        symbol = symbol.upper()
        
        chain_info = nse_client.get_option_chain_info(symbol)
        if "error" in chain_info:
            return chain_info
        
        expiry = chain_info["expiryDates"][0] if "expiryDates" in chain_info else None
        chain_data = nse_client.get_option_chain(symbol, expiry)
        if "error" in chain_data:
            return chain_data
        
        processed_chain = processor.extract_option_chain_data(chain_data)
        all_options = processed_chain["calls"] + processed_chain["puts"]
        
        tracker.store_snapshot(symbol, all_options)
        
        return {
            "status": "success",
            "message": f"Data refreshed for {symbol}",
            "timestamp": datetime.now().isoformat(),
            "options_stored": len(all_options),
            "underlying_value": processed_chain["underlying_value"]
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/option-chain-contract-info")
def get_option_chain_info(symbol: str = "NIFTY"):
    """Get option chain contract info"""
    return nse_client.get_option_chain_info(symbol.upper())

@app.get("/api/v1/ai/option-chain-analysis")
def analyze_option_chain_legacy(
    symbol: str = "NIFTY",
    expiry: str = None
):
    """Legacy endpoint - redirects to comprehensive analysis"""
    result = analyze_options_comprehensive(
        symbol=symbol.upper(), 
        expiry=expiry, 
        analysis_type="option_chain"
    )
    
    # Return in legacy format for backward compatibility
    if "option_chain_analysis" in result:
        chain_data = result["option_chain_analysis"]
        return {
            **chain_data,
            "ai_strategies": result.get("ai_recommended_strategies", []),
            "analysis_method": "AI Option Chain Analysis"
        }
    
    return result