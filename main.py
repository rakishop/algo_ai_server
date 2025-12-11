from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
import json
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

load_dotenv()
from nse_client import NSEClient
from filtered_endpoints import FilteredEndpoints
from ai_endpoints import AIEndpoints
from routes.market_routes import create_market_routes
from routes.derivatives_routes import create_derivatives_routes
from routes.indices_routes import create_indices_routes
from routes.equity_routes import create_equity_routes
from routes.futures_analysis_fixed import create_futures_analysis_routes
from option_chain_endpoint import create_option_chain_routes
from charting_endpoint import create_charting_routes
from ws.websocket_streaming import manager
from portfolio_manager import PortfolioManager
from risk_manager import RiskManager
import asyncio
from typing import List, Dict
import schedule
import threading
import time
from datetime import datetime

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        from train_models import train_from_json_files, setup_training_data
        if not train_from_json_files():
            setup_training_data()
    except Exception as e:
        print(f"Training warning: {e}")
    
    stream_task = asyncio.create_task(manager.start_market_stream())
    
    # Start Telegram alerts scheduler
    def run_telegram_alerts():
        try:
            from auto_stock_alerts import send_stock_alert
            send_stock_alert()
            print(f"Telegram alert sent at {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"Telegram alert failed: {e}")
    
    # Start Intelligent Derivative Analysis scheduler
    def run_intelligent_derivative_analysis():
        try:
            from analyzers.intelligent_derivative_analyzer import IntelligentDerivativeAnalyzer
            analyzer = IntelligentDerivativeAnalyzer()
            if analyzer.is_market_open():
                result = analyzer.run_intelligent_analysis(websocket_manager=manager)
                if result:
                    print(f"Intelligent derivative alert sent at {datetime.now().strftime('%H:%M:%S')}")
                    print(f"WebSocket connections: {len(manager.active_connections)}")
                else:
                    print(f"No new derivative opportunities at {datetime.now().strftime('%H:%M:%S')}")
            else:
                print(f"Market closed - skipping derivative analysis at {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"Intelligent derivative analysis failed: {e}")
    
    # Start Volume alerts scheduler (reduced frequency)
    def run_volume_alerts():
        try:
            from volume_alert_system import VolumeAlertSystem
            alert_system = VolumeAlertSystem()
            if alert_system.is_market_open():
                from volume_alert_system import send_volume_alert
                send_volume_alert()
                print(f"Volume alert checked at {datetime.now().strftime('%H:%M:%S')}")
            else:
                print(f"Market closed - skipping volume check at {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"Volume alert failed: {e}")
    
    schedule.every(30).minutes.do(run_telegram_alerts)
    schedule.every(3).minutes.do(run_intelligent_derivative_analysis)  # Intelligent analysis every 3 minutes
    schedule.every(10).minutes.do(run_volume_alerts)  # Reduced from 3 to 10 minutes
    
    # Smart alerts every 15 minutes
    def run_smart_alerts():
        try:
            from smart_alerts import run_smart_alerts
            run_smart_alerts()
            print(f"Smart alerts checked at {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"Smart alerts failed: {e}")
    
    schedule.every(15).minutes.do(run_smart_alerts)
    
    # News alerts every 60 minutes
    def run_news_alerts():
        try:
            from news_telegram_alert import send_news_to_telegram
            send_news_to_telegram()
            print(f"News alert sent at {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"News alert failed: {e}")
    
    schedule.every(60).minutes.do(run_news_alerts)
    
    # Sector news alerts every 2 hours
    def run_sector_alerts():
        try:
            from sector_news_monitor import monitor_sector_news
            alerts_sent = monitor_sector_news()
            print(f"Sector alerts: {alerts_sent} sent at {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"Sector alerts failed: {e}")
    
    schedule.every(2).hours.do(run_sector_alerts)
    
    # Free Twitter monitoring every 10 minutes
    def run_twitter_monitor():
        try:
            from free_twitter_scraper import scrape_free_tweets
            result = scrape_free_tweets()
            if result:
                print(f"Free Twitter alert sent at {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"Free Twitter scraper failed: {e}")
    
    schedule.every(10).minutes.do(run_twitter_monitor)
    
    # NSE Large Deals monitoring every 5 minutes
    def run_large_deals_monitor():
        try:
            from send_telegram import send_telegram_message
            data = nse_client.get_capital_market_large_deals()
            
            if "error" not in data and 'LARGEDEAL' in data and data['LARGEDEAL']:
                deals = data['LARGEDEAL']
                message = f"🔥 NSE LARGE DEALS - {datetime.now().strftime('%H:%M')}\n\n"
                for i, deal in enumerate(deals[:3], 1):  # Top 3 deals
                    symbol = deal.get('symbol', deal.get('SYMBOL', ''))
                    deal_type = deal.get('dealType', deal.get('DEAL_TYPE', ''))
                    price = deal.get('price', deal.get('PRICE', 0))
                    quantity = deal.get('quantity', deal.get('QUANTITY', 0))
                    value = deal.get('value', deal.get('VALUE', 0))
                    value_cr = value / 10000000
                    message += f"{i}. {symbol} | {deal_type} | ₹{price:.1f} | {quantity:,} | ₹{value_cr:.1f}Cr\n"
                
                message += f"\n💰 Total {len(deals)} large deals"
                send_telegram_message(message)
                print(f"Large deals alert sent at {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"Large deals monitor failed: {e}")
    
    schedule.every(5).minutes.do(run_large_deals_monitor)
    
    def run_scheduler():
        print("Starting schedulers (Telegram: 30min, Derivatives: 3min, Volume: 10min)...")
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)
            except KeyboardInterrupt:
                print("Scheduler interrupted")
                break
            except Exception as e:
                print(f"Scheduler error: {e}")
                time.sleep(60)
    
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    # Start instant news monitoring
    def start_news_monitor():
        from instant_news_monitor import start_instant_news_monitor
        start_instant_news_monitor()
    
    news_monitor_thread = threading.Thread(target=start_news_monitor, daemon=True)
    news_monitor_thread.start()
    
    # Start NSE announcement monitoring for WebSocket broadcasting
    def start_nse_monitor():
        import asyncio
        from nse_announcement_monitor import start_exchange_announcement_monitor
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(start_exchange_announcement_monitor())
    
    nse_monitor_thread = threading.Thread(target=start_nse_monitor, daemon=True)
    nse_monitor_thread.start()
    
    # Start keep-alive service for Render.com
    from keep_alive import KeepAlive
    server_url = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000")
    keep_alive = KeepAlive(server_url)
    keep_alive.start()
    
    print(f"Telegram scheduler started at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Intelligent derivative analysis started (every 15 minutes during market hours)")
    print(f"Volume alert scheduler started (every 10 minutes)")
    print(f"NSE Large Deals monitor started (every 5 minutes)")
    print(f"Instant news monitor started (every 1 minute)")
    print(f"NSE & BSE announcement WebSocket monitor started (every 2 minutes)")
    print(f"Keep-alive service started for {server_url}")
    yield
    # Shutdown
    print("🛑 Shutting down services...")
    
    # Stop all schedulers
    schedule.clear()
    
    # Stop exchange announcement monitor
    try:
        from nse_announcement_monitor import stop_exchange_monitor
        stop_exchange_monitor()
    except Exception as e:
        print(f"Error stopping exchange monitor: {e}")
    
    # Cancel WebSocket stream
    if not stream_task.cancelled():
        stream_task.cancel()
        try:
            await stream_task
        except asyncio.CancelledError:
            print("🛑 WebSocket stream cancelled")
    
    # Stop keep-alive service
    try:
        keep_alive.stop()
    except:
        pass
    
    print("🛑 Server shutdown complete")

app = FastAPI(
    title="NSE Market Data API", 
    version="3.0.0", 
    description="AI-Powered NSE market data with ML analysis, portfolio management, and real-time streaming",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize new managers
portfolio_manager = PortfolioManager()
risk_manager = RiskManager()
nse_client = NSEClient()

# Create shared instances to avoid multiple instantiation
from data_processor import DataProcessor
from ml_analyzer import MLStockAnalyzer
processor = DataProcessor()
ml_analyzer = MLStockAnalyzer()

# Initialize endpoints
filtered_endpoints = FilteredEndpoints(app, nse_client)
ai_endpoints = AIEndpoints(app, nse_client)

# Initialize enhanced AI endpoints
from enhanced_ai_endpoints import EnhancedAIEndpoints
enhanced_ai_endpoints = EnhancedAIEndpoints(app, nse_client)

# Include route modules
app.include_router(create_market_routes(nse_client))
app.include_router(create_derivatives_routes(nse_client))
app.include_router(create_indices_routes(nse_client))
app.include_router(create_equity_routes(nse_client))
app.include_router(create_futures_analysis_routes(nse_client))
app.include_router(create_charting_routes(nse_client))
# Option chain routes integrated into main comprehensive analysis endpoint

# Include enhanced endpoints
from enhanced_endpoints import router as enhanced_router
app.include_router(enhanced_router)



@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/")
def welcome():
    return {
        "message": "Hi, Welcome to Enterprise API v3.0",
        "new_features": [
            "Real-time WebSocket streaming",
            "Advanced portfolio management",
            "Risk management system",
            "Enhanced AI analysis"
        ],
        "version": "3.0.0"
    }

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

@app.get("/api/v1/technical/indicators")
def get_technical_indicators(
    symbol: str = "RELIANCE"
):
    """Get AI-powered technical indicators using pandas and ML models"""
    try:
        from ai_technical_analyzer import AITechnicalAnalyzer
        
        # Get market data
        gainers_data = nse_client.get_gainers_data()
        active_data = nse_client.get_most_active_securities()
        volume_data = nse_client.get_volume_gainers()
        
        # Find symbol in the data
        symbol = symbol.upper()
        stock_data = None
        
        # Search across all data sources
        for data_source in [gainers_data, active_data, volume_data]:
            if data_source.get("data"):
                for stock in data_source["data"]:
                    if stock.get("symbol") == symbol:
                        stock_data = stock
                        break
            if stock_data:
                break
        
        if not stock_data:
            # Get available symbols for debugging
            available_symbols = []
            for data_source in [gainers_data, active_data, volume_data]:
                if data_source and data_source.get("data"):
                    available_symbols.extend([s.get("symbol", "") for s in data_source["data"][:5]])
            
            return {
                "error": f"Symbol {symbol} not found in current market data",
                "available_symbols": list(set(available_symbols)),
                "suggestion": "Try using one of the available symbols or check if markets are open"
            }
        
        # Get market context for AI analysis
        market_context = []
        for data_source in [gainers_data, active_data]:
            if data_source.get("data"):
                market_context.extend(data_source["data"][:10])  # Top 10 from each
        
        # Initialize AI analyzer
        ai_analyzer = AITechnicalAnalyzer()
        
        # Perform AI analysis
        analysis_result = ai_analyzer.analyze_stock(stock_data, market_context)
        
        return analysis_result
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/technical/multi-timeframe")
def get_multi_timeframe_analysis(
    symbol: str = "RELIANCE"
):
    """Get multi-timeframe AI technical analysis"""
    try:
        from ai_technical_analyzer import AITechnicalAnalyzer
        
        # Get current stock data
        gainers_data = nse_client.get_gainers_data()
        active_data = nse_client.get_most_active_securities()
        
        symbol = symbol.upper()
        stock_data = None
        
        for data_source in [gainers_data, active_data]:
            if data_source.get("data"):
                for stock in data_source["data"]:
                    if stock.get("symbol") == symbol:
                        stock_data = stock
                        break
            if stock_data:
                break
        
        if not stock_data:
            # Get available symbols for debugging
            available_symbols = []
            for data_source in [gainers_data, active_data]:
                if data_source and data_source.get("data"):
                    available_symbols.extend([s.get("symbol", "") for s in data_source["data"][:5]])
            
            return {
                "error": f"Symbol {symbol} not found in current market data",
                "available_symbols": list(set(available_symbols)),
                "suggestion": "Try using one of the available symbols or check if markets are open"
            }
        
        ai_analyzer = AITechnicalAnalyzer()
        
        # Simulate different timeframes with varying volatility
        timeframes = {
            "5min": {"volatility_factor": 0.5, "trend_factor": 0.3},
            "15min": {"volatility_factor": 0.7, "trend_factor": 0.5},
            "1hour": {"volatility_factor": 1.0, "trend_factor": 0.8},
            "daily": {"volatility_factor": 1.5, "trend_factor": 1.0}
        }
        
        analysis_results = {}
        consensus_signals = []
        
        for tf_name, tf_params in timeframes.items():
            # Modify stock data for different timeframes
            tf_stock_data = stock_data.copy()
            current_change = float(stock_data.get('perChange', 0))
            tf_stock_data['perChange'] = current_change * tf_params['trend_factor']
            
            # Analyze for this timeframe
            try:
                tf_analysis = ai_analyzer.analyze_stock(tf_stock_data, [])
                analysis_results[tf_name] = tf_analysis
                
                # Collect signals for consensus
                if 'ai_analysis' in tf_analysis and tf_analysis['ai_analysis']:
                    signal = tf_analysis['ai_analysis'].get('predicted_signal', 'HOLD')
                    confidence = tf_analysis['ai_analysis'].get('confidence', 50)
                    consensus_signals.append((signal, confidence, tf_params['trend_factor']))
            except Exception as e:
                analysis_results[tf_name] = {"error": f"Analysis failed: {str(e)}"}
        
        # Calculate consensus
        weighted_signals = {}
        for signal, confidence, weight in consensus_signals:
            if signal not in weighted_signals:
                weighted_signals[signal] = 0
            weighted_signals[signal] += confidence * weight
        
        consensus_signal = max(weighted_signals, key=weighted_signals.get) if weighted_signals else "HOLD"
        consensus_confidence = max(weighted_signals.values()) / sum(weighted_signals.values()) * 100 if weighted_signals else 50
        
        return {
            "symbol": symbol,
            "multi_timeframe_analysis": analysis_results,
            "consensus": {
                "signal": consensus_signal,
                "confidence": round(consensus_confidence, 2),
                "agreement_level": "High" if consensus_confidence > 70 else "Medium" if consensus_confidence > 50 else "Low"
            },
            "analysis_method": "AI Multi-Timeframe Technical Analysis"
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

@app.get("/api/v1/telegram/poll")
async def telegram_poll():
    """Poll for new Telegram messages and respond"""
    try:
        import requests
        from notifications.telegram_handler import TelegramHandler
        from config import settings
        
        handler = TelegramHandler()
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/getUpdates"
        
        response = requests.get(url)
        updates = response.json()
        
        processed = 0
        if updates.get("ok") and updates.get("result"):
            for update in updates["result"][-5:]:  # Process last 5 messages
                if 'message' in update and update['message'].get('text'):
                    handler.handle_message(update)
                    processed += 1
        
        return {"status": "success", "processed_messages": processed}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/ai/price-targets")
async def get_price_targets(symbol: str = "RELIANCE"):
    """Get AI price targets for stock"""
    try:
        from price_targets import analyze_stock_targets
        return analyze_stock_targets(symbol)
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/ai/news-sentiment")
async def get_news_sentiment_endpoint():
    """Get market news sentiment analysis from real sources"""
    try:
        from news_sentiment import get_news_sentiment
        return get_news_sentiment()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/news/all-sources")
async def get_all_news_sources():
    """Get raw news from all sources"""
    try:
        from real_news_fetcher import fetch_real_news
        return fetch_real_news()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/news/dashboard")
async def get_news_for_dashboard():
    """Get formatted news data for dashboard display"""
    try:
        from real_news_fetcher import RealNewsFetcher
        fetcher = RealNewsFetcher()
        
        # Get all news
        all_news_data = fetcher.get_all_news()
        
        # Format for UI
        formatted_news = []
        for news_item in all_news_data.get('news', []):
            formatted_news.append({
                'id': hash(news_item.get('title', '') + news_item.get('source', '')),
                'title': news_item.get('title', 'No Title'),
                'summary': news_item.get('summary', news_item.get('subject', 'No Summary')),
                'source': news_item.get('source', 'Unknown').replace('_', ' ').title(),
                'link': news_item.get('link', ''),
                'published': news_item.get('published', news_item.get('timestamp', '')),
                'images': news_item.get('images', []),
                'has_images': news_item.get('has_images', False),
                'timestamp': news_item.get('timestamp', '')
            })
        
        # Sort by timestamp (newest first)
        formatted_news.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return {
            'success': True,
            'total_news': len(formatted_news),
            'sources_count': {
                'rss_feeds': all_news_data.get('rss_count', 0),
                'nse_announcements': all_news_data.get('nse_count', 0),
                'bse_announcements': all_news_data.get('bse_count', 0),
                'tv_channels': all_news_data.get('tv_count', 0)
            },
            'last_updated': all_news_data.get('last_updated', ''),
            'news': formatted_news  # Return all news items
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'news': [],
            'total_news': 0
        }

@app.get("/api/v1/news/nse-announcements")
async def get_nse_announcements():
    """Get NSE corporate announcements"""
    try:
        from real_news_fetcher import RealNewsFetcher
        fetcher = RealNewsFetcher()
        return {"announcements": fetcher.scrape_nse_announcements()}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/news/bse-announcements")
async def get_bse_announcements():
    """Get BSE announcements"""
    try:
        from real_news_fetcher import RealNewsFetcher
        fetcher = RealNewsFetcher()
        return {"announcements": fetcher.scrape_bse_announcements()}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/feeds/rss")
async def get_rss_feeds():
    """Get RSS feeds manually"""
    try:
        from real_news_fetcher import RealNewsFetcher
        fetcher = RealNewsFetcher()
        return {"rss_news": fetcher.fetch_rss_news()}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/feeds/nse")
async def get_nse_feeds():
    """Get NSE feeds manually"""
    try:
        from real_news_fetcher import RealNewsFetcher
        fetcher = RealNewsFetcher()
        return {"nse_announcements": fetcher.scrape_nse_announcements()}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/feeds/bse")
async def get_bse_feeds():
    """Get BSE feeds manually"""
    try:
        from real_news_fetcher import RealNewsFetcher
        fetcher = RealNewsFetcher()
        return {"bse_announcements": fetcher.scrape_bse_announcements()}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/feeds/tv")
async def get_tv_feeds():
    """Get TV channel feeds manually"""
    try:
        from real_news_fetcher import RealNewsFetcher
        fetcher = RealNewsFetcher()
        return {"tv_news": fetcher.scrape_tv_channel_news()}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/feeds/all")
async def get_all_feeds():
    """Get all feeds manually"""
    try:
        from real_news_fetcher import RealNewsFetcher
        fetcher = RealNewsFetcher()
        return fetcher.get_all_news()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/market/nse-large-deals")
async def get_nse_large_deals():
    """Get NSE large deals from capital market snapshot"""
    try:
        data = nse_client.get_capital_market_large_deals()
        
        if "error" in data:
            return data
        
        # Format the response - include ALL available fields
        formatted_deals = []
        if 'LARGEDEAL' in data:
            for deal in data['LARGEDEAL']:
                formatted_deals.append({
                    **deal,  # Include all original fields from XML feed
                    'timestamp': datetime.now().isoformat(),
                    # Standardized field names for consistency
                    'symbol': deal.get('symbol', deal.get('SYMBOL', '')),
                    'company_name': deal.get('companyName', deal.get('COMPANY_NAME', '')),
                    'deal_type': deal.get('dealType', deal.get('DEAL_TYPE', '')),
                    'quantity': deal.get('quantity', deal.get('QUANTITY', 0)),
                    'price': deal.get('price', deal.get('PRICE', 0)),
                    'value': deal.get('value', deal.get('VALUE', 0)),
                    'client_name': deal.get('clientName', deal.get('CLIENT_NAME', '')),
                    'deal_time': deal.get('dealTime', deal.get('DEAL_TIME', '')),
                    'link': deal.get('link', deal.get('LINK', '')),
                    'image_path': deal.get('imagePath', deal.get('IMAGE_PATH', '')),
                    'xml_feed_data': deal  # Complete XML feed data
                })
        
        return {
            'success': True,
            'total_deals': len(formatted_deals),
            'large_deals': formatted_deals,
            'last_updated': datetime.now().isoformat()
        }
            
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/feeds/large-deals")
async def get_large_deals_feed():
    """Get NSE large deals feed manually"""
    try:
        data = nse_client.get_capital_market_large_deals()
        if "error" in data:
            return data
        
        formatted_deals = []
        if 'LARGEDEAL' in data:
            for deal in data['LARGEDEAL']:
                formatted_deals.append({
                    **deal,
                    'timestamp': datetime.now().isoformat()
                })
        
        return {
            'success': True,
            'total_deals': len(formatted_deals),
            'large_deals': formatted_deals,
            'last_updated': datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/market/nse-large-deals-telegram")
async def send_nse_large_deals_telegram():
    """Get NSE large deals and send Telegram notification"""
    try:
        from send_telegram import send_telegram_message
        
        data = nse_client.get_capital_market_large_deals()
        
        if "error" in data:
            return data
        
        # Format deals for Telegram - include ALL available fields
        formatted_deals = []
        if 'LARGEDEAL' in data:
            for deal in data['LARGEDEAL']:
                formatted_deals.append({
                    **deal,  # Include all original fields
                    'timestamp': datetime.now().isoformat(),
                    # Standardized field names for consistency
                    'symbol': deal.get('symbol', deal.get('SYMBOL', '')),
                    'company_name': deal.get('companyName', deal.get('COMPANY_NAME', '')),
                    'deal_type': deal.get('dealType', deal.get('DEAL_TYPE', '')),
                    'quantity': deal.get('quantity', deal.get('QUANTITY', 0)),
                    'price': deal.get('price', deal.get('PRICE', 0)),
                    'value': deal.get('value', deal.get('VALUE', 0)),
                    'client_name': deal.get('clientName', deal.get('CLIENT_NAME', '')),
                    'deal_time': deal.get('dealTime', deal.get('DEAL_TIME', '')),
                    'link': deal.get('link', deal.get('LINK', '')),
                    'image_path': deal.get('imagePath', deal.get('IMAGE_PATH', '')),
                    'xml_feed_data': deal  # Store complete XML feed data
                })
        
        # Send Telegram notification if deals found
        if formatted_deals:
            message = f"🔥 NSE LARGE DEALS ALERT - {datetime.now().strftime('%H:%M')}\n\n"
            for i, deal in enumerate(formatted_deals[:5], 1):  # Top 5 deals
                value_cr = deal.get('value', 0) / 10000000  # Convert to crores
                message += f"{i}. {deal['symbol']} | {deal['deal_type']} | ₹{deal.get('price', 0):.1f} | {deal.get('quantity', 0):,} qty | ₹{value_cr:.1f}Cr\n"
            
            message += f"\n💰 Total {len(formatted_deals)} large deals detected"
            
            telegram_result = send_telegram_message(message)
            
            return {
                'success': True,
                'total_deals': len(formatted_deals),
                'large_deals': formatted_deals,
                'telegram_sent': telegram_result.get('ok', False),
                'last_updated': datetime.now().isoformat()
            }
        else:
            return {
                'success': True,
                'total_deals': 0,
                'large_deals': [],
                'telegram_sent': False,
                'message': 'No large deals found',
                'last_updated': datetime.now().isoformat()
            }
            
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/feeds/telegram-test")
async def test_telegram_feed():
    """Test Telegram notification with sample data"""
    try:
        from send_telegram import send_telegram_message
        message = f"🧪 TEST FEED - {datetime.now().strftime('%H:%M')}\n\nThis is a test message from feeds endpoint."
        result = send_telegram_message(message)
        return {
            'success': result.get('ok', False),
            'message': 'Test notification sent',
            'telegram_response': result
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/news/sector-alerts")
async def trigger_sector_alerts():
    """Trigger sector-wise news alerts"""
    try:
        from sector_news_monitor import monitor_sector_news
        alerts_sent = monitor_sector_news()
        return {"status": "success", "alerts_sent": alerts_sent}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/twitter/monitor")
async def trigger_twitter_monitor():
    """Trigger free Twitter monitoring"""
    try:
        from free_twitter_scraper import scrape_free_tweets
        result = scrape_free_tweets()
        return {"status": "success" if result else "no_new_tweets"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/news/telegram-alert")
async def send_news_alert():
    """Send news update to Telegram"""
    try:
        from news_telegram_alert import send_news_to_telegram
        result = send_news_to_telegram()
        return {"status": "success" if result else "failed", "message": "News alert sent"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/ai/smart-alerts")
async def trigger_smart_alerts():
    """Manually trigger smart alerts"""
    try:
        from smart_alerts import run_smart_alerts
        run_smart_alerts()
        return {"status": "success", "message": "Smart alerts triggered"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/ai/derivative-opportunities")
async def get_derivative_opportunities():
    """Get current derivative opportunities without sending notifications"""
    try:
        from analyzers.intelligent_derivative_analyzer import IntelligentDerivativeAnalyzer
        
        analyzer = IntelligentDerivativeAnalyzer()
        
        # Fetch and analyze data
        all_data = analyzer.fetch_all_derivative_data()
        opportunities = analyzer.extract_opportunities_from_data(all_data)
        best_opportunities = analyzer.compare_with_previous_analysis(opportunities)
        
        # Convert opportunities to dict format
        opportunities_data = []
        for opp in best_opportunities[:10]:  # Top 10
            opportunities_data.append({
                "symbol": opp.symbol,
                "option_type": opp.option_type,
                "strike_price": opp.strike_price,
                "expiry_date": opp.expiry_date,
                "current_price": opp.current_price,
                "price_change": opp.price_change,
                "volume": opp.volume,
                "open_interest": opp.open_interest,
                "ai_score": opp.ai_score,
                "recommendation": opp.recommendation,
                "confidence": opp.confidence,
                "reasons": opp.reasons,
                "stop_loss": opp.stop_loss,
                "target": opp.target,
                "risk_reward_ratio": opp.risk_reward_ratio
            })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_opportunities_found": len(opportunities),
            "high_confidence_opportunities": len(best_opportunities),
            "market_open": analyzer.is_market_open(),
            "opportunities": opportunities_data,
            "analysis_criteria": {
                "min_confidence_threshold": analyzer.min_confidence_threshold,
                "min_ai_score_threshold": analyzer.min_ai_score_threshold
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/ai/volume-alert")
async def send_volume_alert_endpoint():
    """Manually trigger volume spike alert"""
    try:
        from volume_alert_system import send_volume_alert
        result = send_volume_alert()
        
        return {
            "status": "success" if result else "no_spikes",
            "message": "Volume alert sent" if result else "No volume spikes detected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/ai/intelligent-derivative-analysis")
async def run_intelligent_derivative_analysis():
    """Manually trigger intelligent derivative analysis"""
    try:
        from analyzers.intelligent_derivative_analyzer import IntelligentDerivativeAnalyzer
        
        analyzer = IntelligentDerivativeAnalyzer()
        
        # Force analysis regardless of timing for manual trigger
        analyzer.last_analysis_time = None
        
        result = analyzer.run_intelligent_analysis(websocket_manager=manager)
        
        return {
            "status": "success" if result else "no_opportunities",
            "message": "High-confidence opportunities found and sent" if result else "No new high-confidence opportunities found",
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "intelligent_derivative_analysis",
            "notification_sent": result
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/ai/telegram-alert")
async def send_telegram_alert():
    """Send AI-analyzed top stocks to Telegram"""
    try:
        import requests
        from config import settings
        from datetime import datetime
        from market_scanner import MarketScanner
        from ai_stock_selector import AIStockSelector
        
        # Use comprehensive AI analysis
        scanner = MarketScanner()
        ai_selector = AIStockSelector()
        
        # Get comprehensive market scan
        market_scan = scanner.comprehensive_market_scan()
        
        # Get AI-selected breakout stocks
        breakout_scan = scanner.scan_breakout_stocks()
        ai_breakouts = breakout_scan.get('breakout_opportunities', [])
        
        # Get momentum stocks
        momentum_scan = scanner.scan_momentum_stocks()
        ai_momentum = momentum_scan.get('momentum_opportunities', [])
        
        # Combine and score all opportunities
        all_opportunities = ai_breakouts + ai_momentum
        
        # Separate gainers and losers with AI scoring
        ai_gainers = [stock for stock in all_opportunities if stock.get('perChange', 0) > 0]
        ai_losers = [stock for stock in all_opportunities if stock.get('perChange', 0) < 0]
        
        # Sort by AI scores
        ai_gainers = sorted(ai_gainers, key=lambda x: x.get('breakout_score', x.get('momentum_score', 0)), reverse=True)[:5]
        ai_losers = sorted(ai_losers, key=lambda x: x.get('breakout_score', x.get('momentum_score', 0)), reverse=True)[:4]
        
        # Create AI-powered message
        message = f"🤖 AI MARKET ANALYSIS - {datetime.now().strftime('%H:%M')}\n\n"
        message += "📈 TOP 5 AI BREAKOUT GAINERS\n"
        for i, stock in enumerate(ai_gainers, 1):
            score = stock.get('breakout_score', stock.get('momentum_score', 0))
            signal_type = "BREAKOUT" if 'breakout_score' in stock else "MOMENTUM"
            message += f"{i}. {stock['symbol']} | ₹{stock['ltp']:.1f} | +{stock['perChange']:.1f}% | AI:{score:.0f} {signal_type}\n"
        
        message += "\n📉 TOP 4 AI BREAKOUT LOSERS\n"
        for i, stock in enumerate(ai_losers, 1):
            score = stock.get('breakout_score', stock.get('momentum_score', 0))
            signal_type = "BREAKOUT" if 'breakout_score' in stock else "MOMENTUM"
            message += f"{i}. {stock['symbol']} | ₹{stock['ltp']:.1f} | {stock['perChange']:.1f}% | AI:{score:.0f} {signal_type}\n"
        
        message += f"\n💡 AI analyzed {len(all_opportunities)} opportunities"
        
        # Send to group
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
        data = {"chat_id": settings.telegram_chat_id, "text": message}
        response = requests.post(url, data=data)
        
        return {
            "status": "success" if response.json().get('ok') else "failed",
            "message": "AI analysis sent to Telegram group",
            "ai_gainers": len(ai_gainers),
            "ai_losers": len(ai_losers),
            "total_analyzed": len(all_opportunities),
            "telegram_response": response.json()
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/test-broadcast")
async def test_broadcast(payload: dict):
    """Test endpoint to broadcast message to all WebSocket clients"""
    try:
        success = await manager.broadcast_json(payload)
        return {"status": "success" if success else "no_connections", "connections": len(manager.active_connections), "payload": payload}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/ai/force-derivative-alert")
async def force_derivative_alert():
    """Force derivative analysis and WebSocket broadcast for testing"""
    try:
        from analyzers.intelligent_derivative_analyzer import IntelligentDerivativeAnalyzer
        analyzer = IntelligentDerivativeAnalyzer()
        
        # Force analysis regardless of timing
        analyzer.last_analysis_time = None
        
        # Run analysis with server's manager in force mode
        result = analyzer.run_intelligent_analysis(websocket_manager=manager, force=True)
        
        return {
            "status": "success" if result else "no_opportunities",
            "message": "Analysis completed and broadcast sent" if result else "No opportunities found",
            "websocket_connections": len(manager.active_connections),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/option-chain-contract-info")
def get_option_chain_info(symbol: str = "NIFTY"):
    """Get option chain contract info"""
    return nse_client.get_option_chain_info(symbol.upper())


@app.get("/api/v1/futures/master-quote")
def get_futures_master_quote():
    """Get all future stock symbols from NSE master-quote API"""
    try:
        return nse_client.get_futures_master_quote()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/options/historical")
def get_options_historical(
    symbol: str = "NIFTY",
    option_type: str = "CE", 
    strike_price: float = 25000.0,
    expiry_date: str = "30-Sep-2025",
    days: int = 90
):
    """Get options historical data with AI analysis"""
    try:
        from datetime import datetime
        to_date = datetime.now().strftime("%d-%m-%Y")
        from_date = datetime.now().strftime("%d-%m-%Y")
        
        hist_data = nse_client.get_options_historical_data(symbol, option_type, strike_price, expiry_date, from_date, to_date)
        if "error" in hist_data:
            return hist_data
        
        # AI analysis with pandas
        import pandas as pd
        df = pd.DataFrame(hist_data["data"])
        
        df['close'] = pd.to_numeric(df['FH_CLOSING_PRICE'], errors='coerce')
        df['volume'] = pd.to_numeric(df['FH_TOT_TRADED_QTY'], errors='coerce')
        df['oi'] = pd.to_numeric(df['FH_OPEN_INT'], errors='coerce')
        df['underlying'] = pd.to_numeric(df['FH_UNDERLYING_VALUE'], errors='coerce')
        
        # Technical indicators
        df['iv_estimate'] = (df['close'] / df['underlying']) * 100
        df['moneyness'] = df['underlying'] / strike_price
        df['oi_change_pct'] = df['FH_CHANGE_IN_OI'] / df['oi'] * 100
        
        latest = df.iloc[0]
        
        return {
            "symbol": symbol,
            "option_type": option_type,
            "strike_price": strike_price,
            "expiry_date": expiry_date,
            "current_data": {
                "premium": float(latest['close']),
                "volume": int(latest['volume']),
                "oi": int(latest['oi']),
                "underlying_price": float(latest['underlying']),
                "moneyness": float(latest['moneyness']),
                "iv_estimate": float(latest['iv_estimate'])
            },
            "analysis": {
                "avg_volume": float(df['volume'].mean()),
                "avg_oi": float(df['oi'].mean()),
                "price_volatility": float(df['close'].std()),
                "trend": "BULLISH" if latest['close'] > df['close'].mean() else "BEARISH"
            },
            "total_records": len(df)
        }
    except Exception as e:
        return {"error": str(e)}

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

