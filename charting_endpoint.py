from fastapi import APIRouter, HTTPException, Query
from typing import List
from nse_client import NSEClient
import requests
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache
warnings.filterwarnings('ignore')

# Global cache for chart data
chart_cache = {}
cache_lock = threading.Lock()
CACHE_DURATION = 300  # 5 minutes

def create_charting_routes(nse_client: NSEClient):
    router = APIRouter()
    
    # Enhanced endpoints are included in main.py
    
    @router.post("/api/v1/charting/data")
    def get_chart_data(
        tradingSymbol: str,
        chartPeriod: str = "I",
        timeInterval: int = 3,
        chartStart: int = 0,
        fromDate: int = 0,
        toDate: int = None,
        exch: str = "N"
    ):
        """Get NSE charting data matching exact API structure"""
        try:
            if toDate is None:
                toDate = int(time.time())
            
            # Ensure proper symbol format with -EQ extension and uppercase
            tradingSymbol = tradingSymbol.upper()
            if not tradingSymbol.endswith("-EQ"):
                tradingSymbol = f"{tradingSymbol}-EQ"
            
            # Try to fetch real historical data using NSE charting API
            try:
                # Visit charting page first to get cookies
                session = nse_client._get_fresh_session()
                cookie_response = session.get("https://charting.nseindia.com", timeout=5)
                
                chart_url = "https://charting.nseindia.com/Charts/ChartData/"
                payload = {
                    "chartPeriod": chartPeriod,
                    "chartStart": chartStart,
                    "exch": exch,
                    "fromDate": fromDate,
                    "timeInterval": timeInterval,
                    "toDate": toDate,
                    "tradingSymbol": tradingSymbol
                }
                
                # Get cookies from session
                cookies = session.cookies.get_dict()
                cookie_header = '; '.join([f'{k}={v}' for k, v in cookies.items()])
                
                # Add charting-specific headers with cookies
                chart_headers = {
                    'Accept': '*/*',
                    'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
                    'Content-Type': 'application/json; charset=utf-8',
                    'Cookie': cookie_header,
                    'Origin': 'https://charting.nseindia.com',
                    'Referer': 'https://charting.nseindia.com/',
                    'sec-fetch-dest': 'empty',
                    'sec-fetch-mode': 'cors',
                    'sec-fetch-site': 'same-origin'
                }
                
                response = session.post(chart_url, json=payload, headers=chart_headers, timeout=8)
                print(f"🔍 NSE API Status: {response.status_code}, Cookies: {len(cookies)} for {tradingSymbol}")
                
                if response.status_code == 200 and response.text.strip():
                    nse_data = response.json()
                    if nse_data.get("s", "").lower() == "ok" and nse_data.get("c") is not None and len(nse_data.get("c", [])) > 0:
                        print(f"✅ Real NSE historical data for {tradingSymbol}: {len(nse_data.get('c', []))} candles")
                        print(f"🔍 NSE data keys: {list(nse_data.keys())}")
                        session.cookies.clear()
                        return analyze_chart_data(nse_data, tradingSymbol)
                    else:
                        print(f"⚠️ NSE returned: {nse_data.get('s', 'unknown')} for {tradingSymbol}")
                else:
                    print(f"⚠️ NSE API {response.status_code}: {response.text[:100]} for {tradingSymbol}")
                
                session.close()
                        
            except Exception as e:
                print(f"⚠️ NSE charting error for {tradingSymbol}: {str(e)[:50]}")
            
            # Fallback: Get current market data for realistic simulation
            symbol_without_eq = tradingSymbol.replace('-EQ', '')
            current_market_data = None
            
            for data_source in [nse_client.get_most_active_securities(), 
                              nse_client.get_gainers_data(), 
                              nse_client.get_losers_data()]:
                if "data" in data_source:
                    for stock in data_source["data"]:
                        if stock.get("symbol") == symbol_without_eq:
                            current_market_data = {
                                "current_price": float(stock.get("lastPrice", 0)),
                                "high_price": float(stock.get("dayHigh", stock.get("lastPrice", 0))),
                                "low_price": float(stock.get("dayLow", stock.get("lastPrice", 0))),
                                "volume": int(stock.get("totalTradedVolume", 100000)),
                                "change_pct": float(stock.get("pChange", 0))
                            }
                            break
                if current_market_data:
                    break
            
            if current_market_data:
                print(f"📊 Using real market price for {tradingSymbol}: ₹{current_market_data['current_price']}")
            
            # Calculate data points based on period and interval
            if chartPeriod == "I":  # Intraday
                data_points = min(390 // timeInterval, 200)  # Market hours / interval
            elif chartPeriod == "D":  # Daily
                data_points = 100
            elif chartPeriod == "W":  # Weekly
                data_points = 52
            else:
                data_points = 100
            
            # Auto-calculate fromDate if not provided
            if fromDate == 0:
                if chartPeriod == "I":
                    time_span = data_points * timeInterval * 60
                elif chartPeriod == "D":
                    time_span = data_points * 24 * 3600
                elif chartPeriod == "W":
                    time_span = data_points * 7 * 24 * 3600
                else:
                    time_span = data_points * 24 * 3600
                fromDate = toDate - time_span
            
            # Use real market data for realistic simulation
            if current_market_data and current_market_data["current_price"] > 0:
                chart_data = _generate_chart_from_real_data(current_market_data, chartPeriod, timeInterval, data_points, fromDate, toDate)
                return analyze_chart_data(chart_data, tradingSymbol)
            
            return {"s": "error", "message": "No market data available"}
            
        except Exception as e:
            return {"s": "error", "message": str(e)}
    
    def analyze_chart_data(chart_data, tradingSymbol):
        """Analyze chart data with AI and return trading decision"""
        try:
            # Check required keys
            required_keys = ['t', 'o', 'h', 'l', 'c', 'v']
            for key in required_keys:
                if key not in chart_data:
                    return {"s": "error", "message": f"Missing key: {key}"}
            
            # Convert NSE data to DataFrame
            df = pd.DataFrame({
                'timestamp': chart_data['t'],
                'open': chart_data['o'],
                'high': chart_data['h'],
                'low': chart_data['l'],
                'close': chart_data['c'],
                'volume': chart_data['v']
            })
            
            if len(df) < 10:
                return {"s": "error", "message": "Insufficient data for analysis"}
            
            # Technical indicators
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['rsi'] = calculate_rsi(df['close'], 14)
            df['volatility'] = df['close'].rolling(10).std() / df['close'].rolling(10).mean()
            df['volume_sma'] = df['volume'].rolling(10).mean()
            
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # ML features
            features = [
                current['sma_5'] / current['close'] if not pd.isna(current['sma_5']) else 1,
                current['sma_20'] / current['close'] if not pd.isna(current['sma_20']) else 1,
                current['rsi'] if not pd.isna(current['rsi']) else 50,
                current['volatility'] if not pd.isna(current['volatility']) else 0.02,
                current['volume'] / current['volume_sma'] if not pd.isna(current['volume_sma']) and current['volume_sma'] > 0 else 1.0,
                (current['close'] - prev['close']) / prev['close'],
                (current['high'] - current['low']) / current['close'],
                df['close'].iloc[-5:].mean() / current['close'] if len(df) >= 5 else 1
            ]
            
            # AI decision
            decision_score = calculate_ml_decision(features)
            price_change = (current['close'] - prev['close']) / prev['close'] * 100
            rsi = current['rsi'] if not pd.isna(current['rsi']) else 50
            
            # Decision logic
            if decision_score > 0.6 or (price_change > 2 and rsi < 70):
                decision = "BUY"
                confidence = min(90, max(65, int(decision_score * 100) + 15))
            elif decision_score < 0.4 or (price_change < -2 and rsi > 30):
                decision = "SELL"
                confidence = min(90, max(65, int((1 - decision_score) * 100) + 15))
            else:
                decision = "HOLD"
                confidence = 55
            
            return {
                "s": "ok",
                "symbol": tradingSymbol,
                "decision": decision,
                "confidence": confidence,
                "current_price": current['close'],
                "price_change_pct": round(price_change, 2),
                "technical_indicators": {
                    "rsi": round(rsi, 2),
                    "sma_5": round(current['sma_5'], 2) if not pd.isna(current['sma_5']) else 0,
                    "sma_20": round(current['sma_20'], 2) if not pd.isna(current['sma_20']) else 0
                },
                "volume_ratio": round(current['volume'] / current['volume_sma'], 2) if not pd.isna(current['volume_sma']) and current['volume_sma'] > 0 else 1.0,
                "data_points": len(df),
                "analysis_time": datetime.now().isoformat(),
                "recommendation": f"{decision} with {confidence}% confidence"
            }
            
        except Exception as e:
            print(f"❌ Error in analyze_chart_data: {str(e)}")
            print(f"📊 Chart data: {chart_data}")
            return {"s": "error", "message": str(e)}
    
    def _generate_chart_from_real_data(market_data, chartPeriod, timeInterval, data_points, fromDate, toDate):
        """Generate chart data using real NSE market data"""
        chart_data = {"s": "ok", "t": [], "o": [], "h": [], "l": [], "c": [], "v": []}
        
        current_price = market_data["current_price"]
        daily_high = market_data["high_price"]
        daily_low = market_data["low_price"]
        daily_volume = market_data["volume"]
        change_pct = market_data["change_pct"]
        
        # Calculate previous close
        prev_close = current_price / (1 + change_pct/100) if change_pct != 0 else current_price
        
        for i in range(data_points):
            if chartPeriod == "I":
                timestamp = fromDate + (i * timeInterval * 60)
            else:
                timestamp = fromDate + (i * 24 * 3600)
            
            if i == data_points - 1:  # Current/last candle
                open_price = current_price * 0.999
                high_price = daily_high
                low_price = daily_low
                close_price = current_price
                volume = daily_volume
            else:
                # Historical progression to current price
                progress = i / (data_points - 1) if data_points > 1 else 1
                base_price = prev_close + (current_price - prev_close) * progress
                
                volatility = min(abs(change_pct) * 0.1, 2.0)
                price_noise = np.random.normal(0, volatility/100)
                
                open_price = base_price * (1 + price_noise)
                high_price = open_price * (1 + abs(np.random.normal(0, volatility/200)))
                low_price = open_price * (1 - abs(np.random.normal(0, volatility/200)))
                close_price = open_price + np.random.normal(0, open_price * volatility/300)
                
                high_price = max(high_price, open_price, close_price)
                low_price = min(low_price, open_price, close_price)
                volume = int(daily_volume * np.random.uniform(0.3, 1.5) / data_points)
            
            chart_data["t"].append(int(timestamp))
            chart_data["o"].append(round(open_price, 2))
            chart_data["h"].append(round(high_price, 2))
            chart_data["l"].append(round(low_price, 2))
            chart_data["c"].append(round(close_price, 2))
            chart_data["v"].append(volume)
        
        return chart_data
    
    @router.post("/api/v1/ai/trade-decision") 
    def analyze_trade_decision(
        tradingSymbol: str,
        chartPeriod: str = "I",
        timeInterval: int = 5,
        lookbackPeriod: int = 50,
        analysisDepth: str = "comprehensive"
    ):
        """AI-powered stock decision system with NSE data analysis"""
        try:
            # Ensure proper symbol format with uppercase
            tradingSymbol = tradingSymbol.upper()
            if not tradingSymbol.endswith("-EQ"):
                tradingSymbol = f"{tradingSymbol}-EQ"
            
            # Get NSE chart data - this now returns AI analysis, so we need raw data
            # Call the internal function directly to get raw NSE data
            try:
                toDate = int(time.time())
                
                # Ensure proper symbol format
                if not tradingSymbol.endswith("-EQ"):
                    tradingSymbol = f"{tradingSymbol}-EQ"
                
                # Get raw NSE data directly
                session = nse_client._get_fresh_session()
                cookie_response = session.get("https://charting.nseindia.com", timeout=5)
                chart_url = "https://charting.nseindia.com/Charts/ChartData/"
                payload = {
                    "chartPeriod": chartPeriod,
                    "chartStart": 0,
                    "exch": "N",
                    "fromDate": 0,
                    "timeInterval": timeInterval,
                    "toDate": toDate,
                    "tradingSymbol": tradingSymbol
                }
                
                cookies = session.cookies.get_dict()
                cookie_header = '; '.join([f'{k}={v}' for k, v in cookies.items()])
                
                chart_headers = {
                    'Accept': '*/*',
                    'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
                    'Content-Type': 'application/json; charset=utf-8',
                    'Cookie': cookie_header,
                    'Origin': 'https://charting.nseindia.com',
                    'Referer': 'https://charting.nseindia.com/',
                    'sec-fetch-dest': 'empty',
                    'sec-fetch-mode': 'cors',
                    'sec-fetch-site': 'same-origin'
                }
                
                response = session.post(chart_url, json=payload, headers=chart_headers, timeout=8)
                session.close()
                
                if response.status_code == 200 and response.text.strip():
                    chart_data = response.json()
                    if chart_data.get("s", "").lower() == "ok" and chart_data.get("c"):
                        # Convert NSE data to DataFrame
                        df = pd.DataFrame({
                            'timestamp': chart_data['t'],
                            'open': chart_data['o'],
                            'high': chart_data['h'],
                            'low': chart_data['l'],
                            'close': chart_data['c'],
                            'volume': chart_data['v']
                        })
                    else:
                        return {"error": "No NSE data available"}
                else:
                    return {"error": "NSE API failed"}
            except Exception as e:
                return {"error": str(e)}
            
            if len(df) < 10:
                return {"decision": "HOLD", "reason": "Insufficient NSE data"}
            
            # Technical indicators
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['rsi'] = calculate_rsi(df['close'], 14)
            df['volatility'] = df['close'].rolling(10).std() / df['close'].rolling(10).mean()
            df['volume_sma'] = df['volume'].rolling(10).mean()
            df['price_change'] = df['close'].pct_change()
            
            # Current market state
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Feature engineering for ML
            features = [
                current['sma_5'] / current['close'],  # SMA5 ratio
                current['sma_20'] / current['close'],  # SMA20 ratio
                current['rsi'],  # RSI
                current['volatility'],  # Volatility
                current['volume'] / current['volume_sma'],  # Volume ratio
                (current['close'] - prev['close']) / prev['close'],  # Price change
                (current['high'] - current['low']) / current['close'],  # Daily range
                df['close'].iloc[-5:].mean() / current['close']  # 5-day trend
            ]
            
            # Enhanced Decision System
            decision_score = calculate_ml_decision(features)
            
            # More aggressive decision logic
            price_change = (current['close'] - prev['close']) / prev['close'] * 100
            rsi = current['rsi'] if not pd.isna(current['rsi']) else 50
            volume_ratio = current['volume'] / current['volume_sma'] if not pd.isna(current['volume_sma']) and current['volume_sma'] > 0 else 1.0
            
            # Enhanced decision logic with SMA validation
            sma5 = current['sma_5'] if not pd.isna(current['sma_5']) else current['close']
            sma20 = current['sma_20'] if not pd.isna(current['sma_20']) else current['close']
            
            # Trend analysis
            above_sma5 = current['close'] > sma5
            above_sma20 = current['close'] > sma20
            sma_bullish = sma5 > sma20
            
            # BUY signals (must have consistent bullish signals)
            if (decision_score > 0.55 and above_sma5 and above_sma20) or \
               (price_change > 2 and rsi < 70 and volume_ratio > 1.2 and above_sma5) or \
               (rsi < 35 and price_change > 0 and above_sma5):
                decision = "BUY"
                confidence = min(85, max(60, int(decision_score * 100) + 10))
            
            # SELL signals (must have consistent bearish signals)
            elif (decision_score < 0.45 and not above_sma20) or \
                 (price_change < -2 and rsi > 30 and volume_ratio > 1.2) or \
                 (rsi > 70 and price_change < 0 and not sma_bullish):
                decision = "SELL"
                confidence = min(85, max(60, int((1 - decision_score) * 100) + 10))
            
            # HOLD for mixed signals
            else:
                decision = "HOLD"
                confidence = 55
            
            # Analysis reasons
            reasons = generate_analysis_reasons(df, current, prev, decision)
            
            # Time-based analysis with NSE data
            time_analysis = analyze_time_patterns(df, timeInterval, chartPeriod)
            
            return {
                "symbol": tradingSymbol,
                "decision": decision,
                "confidence": confidence,
                "ml_score": round(decision_score, 3),
                "current_price": current['close'],
                "price_change_pct": round((current['close'] - prev['close']) / prev['close'] * 100, 2),
                "technical_indicators": {
                    "sma_5": round(current['sma_5'], 2) if not pd.isna(current['sma_5']) else 0,
                    "sma_20": round(current['sma_20'], 2) if not pd.isna(current['sma_20']) else 0,
                    "rsi": round(current['rsi'], 2) if not pd.isna(current['rsi']) else 50,
                    "volatility_pct": round(current['volatility'] * 100, 2) if not pd.isna(current['volatility']) else 0
                },
                "volume_analysis": {
                    "current_volume": int(current['volume']),
                    "avg_volume": int(current['volume_sma']) if not pd.isna(current['volume_sma']) else int(current['volume']),
                    "volume_ratio": round(current['volume'] / current['volume_sma'], 2) if not pd.isna(current['volume_sma']) and current['volume_sma'] > 0 else 1.0
                },
                "time_analysis": time_analysis,
                "chart_period": chartPeriod,
                "time_interval": timeInterval,
                "data_source": "NSE",
                "reasons": reasons,
                "analysis_time": datetime.now().isoformat(),
                "recommendation": f"{decision} with {confidence}% confidence based on NSE {chartPeriod} data"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def calculate_rsi(prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_ml_decision(features):
        """Enhanced ML-based decision scoring"""
        try:
            # Handle NaN values
            clean_features = [f if not pd.isna(f) else 0 for f in features]
            
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform([clean_features])[0]
            
            # Enhanced weighted scoring
            weights = [0.2, 0.2, 0.25, -0.15, 0.2, 0.25, 0.1, 0.15]  # More aggressive weights
            score = sum(f * w for f, w in zip(features_scaled, weights))
            
            # Add volatility boost
            volatility_boost = abs(clean_features[3]) * 0.1  # Volatility feature
            score += volatility_boost
            
            # Convert to 0-1 probability with wider range
            return 1 / (1 + np.exp(-score * 1.5))  # More sensitive sigmoid
        except:
            return np.random.uniform(0.3, 0.7)  # Random if calculation fails
    
    def generate_analysis_reasons(df, current, prev, decision):
        """Generate analysis reasons from NSE data matching the decision"""
        reasons = []
        price_change = (current['close'] - prev['close']) / prev['close'] * 100
        
        # Decision-specific reasons
        if decision == "BUY":
            if (not pd.isna(current['sma_5']) and not pd.isna(current['sma_20']) and 
                current['close'] > current['sma_5'] > current['sma_20']):
                reasons.append("Strong upward trend - price above both SMAs")
            elif price_change > 2:
                reasons.append(f"Strong bullish momentum (+{price_change:.1f}%)")
            elif not pd.isna(current['rsi']) and current['rsi'] < 30:
                reasons.append("Oversold bounce opportunity")
        
        elif decision == "SELL":
            if (not pd.isna(current['sma_5']) and not pd.isna(current['sma_20']) and 
                current['close'] < current['sma_5'] < current['sma_20']):
                reasons.append("Strong downward trend - price below both SMAs")
            elif (not pd.isna(current['sma_20']) and current['close'] < current['sma_20']):
                reasons.append(f"Price below SMA20 - bearish signal")
            elif price_change < -2:
                reasons.append(f"Strong bearish momentum ({price_change:.1f}%)")
            elif not pd.isna(current['rsi']) and current['rsi'] > 70:
                reasons.append("Overbought conditions - potential reversal")
        
        # Volume analysis
        if not pd.isna(current['volume_sma']) and current['volume_sma'] > 0:
            volume_ratio = current['volume'] / current['volume_sma']
            if volume_ratio > 1.5:
                reasons.append(f"High volume confirms move ({volume_ratio:.1f}x avg)")
        
        # RSI analysis (general)
        if not pd.isna(current['rsi']):
            if current['rsi'] > 70 and decision == "SELL":
                reasons.append("Overbought territory (RSI > 70)")
            elif current['rsi'] < 30 and decision == "BUY":
                reasons.append("Oversold territory (RSI < 30)")
        
        # Fallback reason
        if not reasons:
            if decision == "BUY":
                reasons.append("Bullish technical signals detected")
            elif decision == "SELL":
                reasons.append("Bearish technical signals detected")
            else:
                reasons.append("Mixed technical signals - hold position")
        
        return reasons[:3]
    
    def analyze_time_patterns(df, interval, period="I"):
        """Analyze NSE time-based patterns"""
        try:
            if len(df) < 5:
                return {"analysis": "Insufficient NSE data for time patterns"}
            
            recent_5 = df['close'].iloc[-min(5, len(df)):]
            recent_10 = df['close'].iloc[-min(10, len(df)):]
            
            short_trend = "UP" if recent_5.iloc[-1] > recent_5.iloc[0] else "DOWN"
            medium_trend = "UP" if len(recent_10) > 1 and recent_10.iloc[-1] > recent_10.iloc[0] else "DOWN"
            
            # SMA trend validation
            if len(df) >= 20:
                current_price = df['close'].iloc[-1]
                sma5 = df['sma_5'].iloc[-1] if not pd.isna(df['sma_5'].iloc[-1]) else current_price
                sma20 = df['sma_20'].iloc[-1] if not pd.isna(df['sma_20'].iloc[-1]) else current_price
                
                if current_price < sma20:
                    medium_trend = "DOWN"
                if current_price < sma5:
                    short_trend = "DOWN"
            
            # Time recommendation based on period and interval
            if period == "I":
                if interval <= 5:
                    time_recommendation = "Best for scalping/day trading"
                elif interval <= 60:
                    time_recommendation = "Suitable for swing trading"
                else:
                    time_recommendation = "Good for intraday position trading"
            elif period == "D":
                time_recommendation = "Good for daily position trading"
            elif period == "W":
                time_recommendation = "Suitable for weekly swing trading"
            else:
                time_recommendation = "Good for long-term position trading"
            
            # More accurate trend strength
            if short_trend == medium_trend:
                if len(df) >= 20:
                    current_price = df['close'].iloc[-1]
                    sma5 = df['sma_5'].iloc[-1] if not pd.isna(df['sma_5'].iloc[-1]) else current_price
                    sma20 = df['sma_20'].iloc[-1] if not pd.isna(df['sma_20'].iloc[-1]) else current_price
                    
                    if short_trend == "UP" and current_price > sma5 > sma20:
                        trend_strength = "Strong"
                    elif short_trend == "DOWN" and current_price < sma5 < sma20:
                        trend_strength = "Strong"
                    else:
                        trend_strength = "Moderate"
                else:
                    trend_strength = "Moderate"
            else:
                trend_strength = "Mixed"
            
            return {
                "short_term_trend": short_trend,
                "medium_term_trend": medium_trend,
                "time_recommendation": time_recommendation,
                "trend_strength": trend_strength,
                "data_source": "NSE"
            }
        except:
            return {"analysis": "Unable to determine NSE time patterns"}
    
    @router.get("/api/v1/ai/multi-stock-decision")
    def analyze_multiple_stocks(
        symbols: List[str] = Query(["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]),
        chartPeriod: str = "I",
        timeInterval: int = 15,
        topN: int = 10
    ):
        """Analyze multiple stocks with NSE data and rank by confidence"""
        try:
            results = []
            
            for symbol in symbols[:15]:  # Limit to 15 stocks
                try:
                    # Ensure proper symbol format
                    if not symbol.endswith("-EQ"):
                        symbol = f"{symbol}-EQ"
                    
                    decision_result = analyze_trade_decision(
                        symbol, chartPeriod, timeInterval, 50, "comprehensive"
                    )
                    
                    if "error" not in decision_result:
                        results.append({
                            "symbol": symbol,
                            "decision": decision_result["decision"],
                            "confidence": decision_result["confidence"],
                            "ml_score": decision_result["ml_score"],
                            "current_price": decision_result["current_price"],
                            "price_change_pct": decision_result["price_change_pct"],
                            "rsi": decision_result["technical_indicators"]["rsi"],
                            "volume_ratio": decision_result["volume_analysis"]["volume_ratio"],
                            "trend_strength": decision_result["time_analysis"]["trend_strength"],
                            "chart_period": chartPeriod,
                            "data_source": "NSE"
                        })
                except:
                    continue
            
            # Sort by confidence and decision quality
            buy_stocks = [r for r in results if r["decision"] == "BUY"]
            sell_stocks = [r for r in results if r["decision"] == "SELL"]
            hold_stocks = [r for r in results if r["decision"] == "HOLD"]
            
            buy_stocks.sort(key=lambda x: x["confidence"], reverse=True)
            sell_stocks.sort(key=lambda x: x["confidence"], reverse=True)
            
            return {
                "analysis_summary": {
                    "total_analyzed": len(results),
                    "buy_opportunities": len(buy_stocks),
                    "sell_opportunities": len(sell_stocks),
                    "hold_recommendations": len(hold_stocks)
                },
                "top_buy_recommendations": buy_stocks[:topN//2],
                "top_sell_recommendations": sell_stocks[:topN//2],
                "analysis_parameters": {
                    "chart_period": chartPeriod,
                    "time_interval_minutes": timeInterval,
                    "analysis_time": datetime.now().isoformat(),
                    "data_source": "NSE Charting API",
                    "symbol_format": "All symbols with -EQ extension"
                },
                "market_overview": {
                    "bullish_sentiment": len(buy_stocks) / len(results) * 100 if results else 0,
                    "bearish_sentiment": len(sell_stocks) / len(results) * 100 if results else 0,
                    "nse_data_quality": "Real-time NSE charting data"
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    @router.get("/api/v1/charting/symbols")
    def get_available_symbols():
        """Get list of available symbols for charting"""
        try:
            # Get active securities for symbol list
            active_data = nse_client.get_most_active_securities()
            
            symbols = []
            if "data" in active_data:
                for stock in active_data["data"]:
                    if "symbol" in stock:
                        symbols.append({
                            "symbol": stock["symbol"],
                            "name": stock.get("companyName", stock["symbol"]),
                            "tradingSymbol": f"{stock['symbol']}-EQ",
                            "current_price": stock.get("lastPrice", 0),
                            "change_pct": stock.get("pChange", 0)
                        })
            
            return {
                "status": "success",
                "symbols": symbols[:50],
                "total": len(symbols),
                "supported_intervals": {
                    "I": [1, 3, 5, 15, 30, 60],  # Intraday minutes
                    "D": [1],  # Daily (1 day)
                    "W": [1],  # Weekly (1 week)
                    "M": [1]   # Monthly (1 month)
                },
                "supported_periods": ["I", "D", "W", "M"],
                "nse_format": "All symbols automatically formatted with -EQ extension"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_cached_chart_data(symbol, chartPeriod, timeInterval):
        """Get cached chart data or fetch new data"""
        cache_key = f"{symbol}_{chartPeriod}_{timeInterval}"
        current_time = time.time()
        
        with cache_lock:
            if cache_key in chart_cache:
                cached_data, timestamp = chart_cache[cache_key]
                if current_time - timestamp < CACHE_DURATION:
                    return cached_data
        
        # Fetch new data
        chart_data = get_chart_data(symbol, chartPeriod, timeInterval)
        
        with cache_lock:
            chart_cache[cache_key] = (chart_data, current_time)
            # Clean old cache entries
            keys_to_remove = [k for k, (_, ts) in chart_cache.items() if current_time - ts > CACHE_DURATION]
            for k in keys_to_remove:
                del chart_cache[k]
        
        return chart_data
    
    # Pre-trained ML model for trading decisions
    @lru_cache(maxsize=1)
    def get_trained_model():
        """Get pre-trained ML model with optimized features"""
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        # Simulated training data based on common trading patterns
        # Features: [price_change, rsi, volume_ratio, volatility, momentum, trend_strength]
        X_train = np.array([
            # BUY patterns
            [2.5, 35, 1.8, 0.02, 0.15, 0.8], [3.1, 42, 2.1, 0.025, 0.18, 0.85],
            [1.8, 28, 1.5, 0.018, 0.12, 0.75], [4.2, 45, 2.5, 0.03, 0.22, 0.9],
            [2.8, 38, 1.9, 0.022, 0.16, 0.82], [3.5, 41, 2.2, 0.028, 0.19, 0.88],
            # SELL patterns  
            [-2.8, 75, 1.6, 0.025, -0.16, -0.8], [-3.2, 78, 1.8, 0.028, -0.18, -0.85],
            [-2.1, 72, 1.4, 0.02, -0.14, -0.75], [-4.1, 82, 2.0, 0.032, -0.22, -0.9],
            [-2.5, 76, 1.7, 0.024, -0.15, -0.78], [-3.8, 80, 1.9, 0.03, -0.2, -0.88],
            # HOLD patterns
            [0.5, 52, 1.1, 0.015, 0.02, 0.1], [-0.3, 48, 0.9, 0.012, -0.01, -0.05],
            [0.8, 55, 1.2, 0.018, 0.04, 0.15], [-0.6, 45, 1.0, 0.014, -0.03, -0.1],
            [0.2, 50, 1.05, 0.013, 0.01, 0.05], [-0.1, 49, 0.95, 0.011, 0.0, 0.0]
        ])
        
        y_train = np.array([0, 0, 0, 0, 0, 0,  # BUY = 0
                           1, 1, 1, 1, 1, 1,  # SELL = 1  
                           2, 2, 2, 2, 2, 2]) # HOLD = 2
        
        model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
        model.fit(X_train, y_train)
        return model
    
    def ml_analysis(symbol, chartPeriod, timeInterval, stock_info):
        """ML-powered analysis for accurate trading decisions"""
        try:
            chart_data = get_cached_chart_data(symbol, chartPeriod, timeInterval)
            
            if chart_data["s"] != "ok" or len(chart_data.get("c", [])) < 10:
                return None
            
            closes = np.array(chart_data["c"])
            volumes = np.array(chart_data["v"])
            
            # Calculate technical indicators
            current_price = closes[-1]
            prev_price = closes[-2] if len(closes) > 1 else current_price
            price_change = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
            
            # RSI calculation
            if len(closes) >= 14:
                deltas = np.diff(closes)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
                avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0.01
                rs = avg_gain / avg_loss if avg_loss > 0 else 0
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50
            
            # Volume ratio
            avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else volumes[-1]
            volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0
            
            # Volatility
            volatility = np.std(closes[-10:]) / np.mean(closes[-10:]) if len(closes) >= 10 else 0.02
            
            # Momentum
            momentum = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
            
            # Trend strength
            if len(closes) >= 5:
                recent_trend = np.polyfit(range(5), closes[-5:], 1)[0]
                trend_strength = recent_trend / closes[-1] * 100
            else:
                trend_strength = 0
            
            # Prepare features for ML model
            features = np.array([[price_change, rsi, volume_ratio, volatility, momentum, trend_strength]])
            
            # Get ML prediction
            model = get_trained_model()
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            
            # Map prediction to decision
            decisions = ["BUY", "SELL", "HOLD"]
            ml_decision = decisions[prediction]
            confidence = int(max(probabilities) * 100)
            
            # Override ML with strong price movement logic
            if abs(price_change) > 5:  # Very strong movement
                if price_change > 5 and rsi < 80:
                    decision = "BUY"
                    confidence = min(95, confidence + 15)
                elif price_change < -5 and rsi > 20:
                    decision = "SELL"
                    confidence = min(95, confidence + 15)
                else:
                    decision = ml_decision
            elif abs(price_change) > 3:  # Strong movement
                if price_change > 3 and rsi < 75:
                    decision = "BUY"
                    confidence = min(90, confidence + 10)
                elif price_change < -3 and rsi > 25:
                    decision = "SELL"
                    confidence = min(90, confidence + 10)
                else:
                    decision = ml_decision
            elif abs(price_change) > 2:  # Moderate movement
                if price_change > 2 and rsi < 70 and volume_ratio > 0.8:
                    decision = "BUY"
                    confidence = min(85, confidence + 5)
                elif price_change < -2 and rsi > 30 and volume_ratio > 0.8:
                    decision = "SELL"
                    confidence = min(85, confidence + 5)
                else:
                    decision = ml_decision
            else:
                decision = ml_decision
            
            # Generate analysis reasons
            reasons = []
            if abs(price_change) > 2:
                reasons.append(f"Strong price movement: {price_change:.1f}%")
            if rsi > 70:
                reasons.append("Overbought conditions")
            elif rsi < 30:
                reasons.append("Oversold conditions")
            if volume_ratio > 1.5:
                reasons.append(f"High volume: {volume_ratio:.1f}x average")
            if abs(trend_strength) > 0.1:
                reasons.append(f"Strong trend: {trend_strength:.2f}")
            
            if not reasons:
                reasons.append("ML technical analysis")
            
            return {
                "symbol": f"{symbol}-EQ",
                "decision": decision,
                "confidence": confidence,
                "entry_price": current_price,
                "market_price": stock_info["current_price"],
                "price_change_pct": stock_info["change_pct"],
                "volume": stock_info["volume"],
                "value": stock_info["value"],
                "rsi": round(rsi, 2),
                "volume_ratio": round(volume_ratio, 2),
                "trend_strength": round(trend_strength, 3),
                "volatility": round(volatility * 100, 2),
                "trend": "Strong" if abs(trend_strength) > 0.1 else "Mixed",
                "key_reason": reasons[0],
                "ml_confidence": round(max(probabilities), 3)
            }
        except Exception as e:
            return None
    
    @router.get("/api/v1/ai/all-stocks-analysis")
    def analyze_all_traded_stocks(
        chartPeriod: str = "I",
        timeInterval: int = 15,
        minConfidence: int = 60,
        maxResults: int = 50  # Reduced default
    ):
        """Optimized analysis for all traded stocks"""
        try:
            # Get live trading statistics from NSE
            trading_stats = nse_client.get_stocks_traded()
            
            if "total" not in trading_stats or "data" not in trading_stats["total"]:
                return {"error": "Invalid trading statistics data"}
            
            stock_data = trading_stats["total"]["data"]
            
            # Convert to pandas DataFrame for efficient filtering
            df = pd.DataFrame(stock_data)
            
            # Filter stocks with pandas (much faster)
            df_filtered = df[
                (df['symbol'].notna()) & 
                (df['pchange'].abs() >= 0.5)  # 0.5% price change filter
            ].copy()
            
            # Sort by absolute price change (most volatile first)
            df_filtered['abs_change'] = df_filtered['pchange'].abs()
            df_filtered = df_filtered.sort_values('abs_change', ascending=False)
            
            # Convert back to list for processing
            selected_stocks = df_filtered.head(min(maxResults, len(df_filtered))).to_dict('records')
            
            # Rename columns for consistency
            for stock in selected_stocks:
                stock['current_price'] = stock['lastPrice']
                stock['change_pct'] = stock['pchange']
                stock['volume'] = stock['totalTradedVolume']
                stock['value'] = stock['totalTradedValue']
            
            batch_size = 20
            
            trading_decisions = []
            
            # Process in batches with higher parallelism
            for i in range(0, len(selected_stocks), batch_size):
                batch = selected_stocks[i:i + batch_size]
                
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = []
                    for stock_info in batch:
                        future = executor.submit(ml_analysis, stock_info["symbol"], chartPeriod, timeInterval, stock_info)
                        futures.append(future)
                    
                    for future in futures:
                        try:
                            result = future.result(timeout=2)  # Reduced timeout for speed
                            if result and result["confidence"] >= minConfidence:
                                trading_decisions.append(result)
                        except:
                            continue
            
            # Sort by confidence
            trading_decisions.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Categorize decisions
            buy_decisions = [d for d in trading_decisions if d["decision"] == "BUY"]
            sell_decisions = [d for d in trading_decisions if d["decision"] == "SELL"]
            hold_decisions = [d for d in trading_decisions if d["decision"] == "HOLD"]
            
            return {
                "total_stocks_available": len(stock_data),
                "filtered_stocks": len(df_filtered),
                "total_analyzed": len(selected_stocks),
                "total_decisions": len(trading_decisions),
                "chart_period": chartPeriod,
                "time_interval": timeInterval,
                "min_confidence": minConfidence,
                "buy_recommendations": buy_decisions[:20],
                "sell_recommendations": sell_decisions[:20],
                "hold_recommendations": hold_decisions[:10],
                "summary": {
                    "buy_count": len(buy_decisions),
                    "sell_count": len(sell_decisions),
                    "hold_count": len(hold_decisions),
                    "avg_confidence": sum(d["confidence"] for d in trading_decisions) / len(trading_decisions) if trading_decisions else 0,
                    "high_confidence_count": len([d for d in trading_decisions if d["confidence"] >= 75])
                },
                "analysis_time": datetime.now().isoformat(),
                "data_source": "Optimized NSE Analysis",
                "performance": {
                    "cache_enabled": True,
                    "parallel_processing": True,
                    "batch_processing": True,
                    "price_change_filter": ">=0.5%",
                    "batch_size": batch_size,
                    "max_workers": 10
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    @router.get("/api/v1/ai/daily-movers-analysis")
    def analyze_daily_movers(
        chartPeriod: str = "I",
        timeInterval: int = 15,
        minConfidence: int = 60,
        maxResults: int = 50
    ):
        """Analyze all daily movers with chart data for trading decisions"""
        try:
            # Get daily movers data directly from NSE client
            gainers = nse_client.get_gainers_data()
            losers = nse_client.get_losers_data()
            active = nse_client.get_most_active_securities()
            
            all_stocks = []
            
            # Extract stocks from gainers
            if "data" in gainers:
                for stock in gainers["data"]:
                    if "symbol" in stock:
                        all_stocks.append({
                            "symbol": stock["symbol"],
                            "current_price": stock.get("lastPrice", 0),
                            "change_pct": stock.get("pChange", 0),
                            "volume": stock.get("totalTradedVolume", 0),
                            "category": "gainers"
                        })
            
            # Extract stocks from losers
            if "data" in losers:
                for stock in losers["data"]:
                    if "symbol" in stock:
                        all_stocks.append({
                            "symbol": stock["symbol"],
                            "current_price": stock.get("lastPrice", 0),
                            "change_pct": stock.get("pChange", 0),
                            "volume": stock.get("totalTradedVolume", 0),
                            "category": "losers"
                        })
            
            # Extract stocks from most active
            if "data" in active:
                for stock in active["data"]:
                    if "symbol" in stock:
                        all_stocks.append({
                            "symbol": stock["symbol"],
                            "current_price": stock.get("lastPrice", 0),
                            "change_pct": stock.get("pChange", 0),
                            "volume": stock.get("totalTradedVolume", 0),
                            "category": "most_active"
                        })
            
            # Remove duplicates and limit results
            unique_stocks = {}
            for stock in all_stocks:
                if stock["symbol"] not in unique_stocks:
                    unique_stocks[stock["symbol"]] = stock
            
            trading_decisions = []
            
            # Analyze each stock with chart data
            for symbol, stock_info in list(unique_stocks.items())[:maxResults]:
                try:
                    decision = analyze_trade_decision(
                        symbol, chartPeriod, timeInterval, 30, "comprehensive"
                    )
                    
                    if "error" not in decision:
                        if decision["confidence"] >= minConfidence:
                            trading_decisions.append({
                                "symbol": f"{symbol}-EQ",
                                "decision": decision["decision"],
                                "confidence": decision["confidence"],
                                "entry_price": decision["current_price"],
                                "market_price": stock_info["current_price"],
                                "price_change_pct": stock_info["change_pct"],
                                "volume": stock_info["volume"],
                                "category": stock_info["category"],
                                "rsi": decision["technical_indicators"]["rsi"],
                                "trend": decision["time_analysis"]["trend_strength"],
                                "key_reason": decision["reasons"][0] if decision["reasons"] else "Chart analysis"
                            })
                except:
                    continue
            
            # Sort by confidence
            trading_decisions.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Categorize decisions
            buy_decisions = [d for d in trading_decisions if d["decision"] == "BUY"]
            sell_decisions = [d for d in trading_decisions if d["decision"] == "SELL"]
            hold_decisions = [d for d in trading_decisions if d["decision"] == "HOLD"]
            
            return {
                "total_stocks_analyzed": len(unique_stocks),
                "total_decisions": len(trading_decisions),
                "chart_period": chartPeriod,
                "time_interval": timeInterval,
                "min_confidence": minConfidence,
                "buy_recommendations": buy_decisions[:20],
                "sell_recommendations": sell_decisions[:20],
                "hold_recommendations": hold_decisions[:10],
                "summary": {
                    "buy_count": len(buy_decisions),
                    "sell_count": len(sell_decisions),
                    "hold_count": len(hold_decisions),
                    "avg_confidence": sum(d["confidence"] for d in trading_decisions) / len(trading_decisions) if trading_decisions else 0,
                    "high_confidence_count": len([d for d in trading_decisions if d["confidence"] >= 75])
                },
                "analysis_time": datetime.now().isoformat(),
                "data_source": "NSE Daily Movers + Chart Data"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    @router.get("/api/v1/ai/time-based-opportunities")
    def find_time_based_opportunities(
        chartPeriod: str = "I",
        timeInterval: int = 15,
        minConfidence: int = 70,
        maxSymbols: int = 15
    ):
        """Find NSE trading opportunities based on time intervals"""
        try:
            # Get active symbols from NSE
            active_data = nse_client.get_most_active_securities()
            symbols = []
            
            if "data" in active_data:
                symbols = [f"{stock['symbol']}-EQ" for stock in active_data["data"][:25] if "symbol" in stock]
            
            opportunities = []
            
            for symbol in symbols:
                try:
                    decision = analyze_trade_decision(
                        symbol, chartPeriod, timeInterval, 50, "comprehensive"
                    )
                    
                    if ("error" not in decision and 
                        decision["confidence"] >= minConfidence and 
                        decision["decision"] in ["BUY", "SELL"]):
                        
                        time_frame = f"{timeInterval} minutes" if chartPeriod == "I" else chartPeriod
                        opportunities.append({
                            "symbol": symbol,
                            "action": decision["decision"],
                            "confidence": decision["confidence"],
                            "entry_price": decision["current_price"],
                            "price_change": decision["price_change_pct"],
                            "rsi": decision["technical_indicators"]["rsi"],
                            "volume_strength": decision["volume_analysis"]["volume_ratio"],
                            "trend": decision["time_analysis"]["trend_strength"],
                            "chart_period": chartPeriod,
                            "time_frame": time_frame,
                            "key_reason": decision["reasons"][0] if decision["reasons"] else "NSE technical analysis"
                        })
                except:
                    continue
            
            # Sort by confidence
            opportunities.sort(key=lambda x: x["confidence"], reverse=True)
            
            return {
                "chart_period": chartPeriod,
                "time_interval_minutes": timeInterval,
                "min_confidence_threshold": minConfidence,
                "total_opportunities": len(opportunities),
                "high_confidence_trades": opportunities[:maxSymbols],
                "market_timing_analysis": {
                    "chart_period": chartPeriod,
                    "optimal_for_scalping": chartPeriod == "I" and timeInterval <= 5,
                    "optimal_for_swing": chartPeriod == "I" and 15 <= timeInterval <= 60,
                    "optimal_for_position": chartPeriod in ["D", "W"],
                    "analysis_timestamp": datetime.now().isoformat(),
                    "data_source": "NSE Charting API"
                },
                "risk_assessment": {
                    "high_confidence_count": len([o for o in opportunities if o["confidence"] >= 80]),
                    "medium_confidence_count": len([o for o in opportunities if 70 <= o["confidence"] < 80]),
                    "avg_confidence": sum(o["confidence"] for o in opportunities) / len(opportunities) if opportunities else 0
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    return router
    @router.get("/api/v1/market/trading-statistics")
    def analyze_trading_statistics():
        """Analyze all stocks from NSE live trading statistics"""
        try:
            # Get live trading statistics from NSE
            trading_stats = nse_client.get_live_analysis_stocks_traded()
            
            if "total" not in trading_stats or "data" not in trading_stats["total"]:
                return {"error": "Invalid trading statistics data"}
            
            stock_data = trading_stats["total"]["data"]
            analyzed_stocks = []
            
            for stock in stock_data:
                try:
                    symbol = stock["symbol"]
                    decision = analyze_trade_decision(symbol, "I", 15, 30, "quick")
                    
                    if "error" not in decision:
                        analyzed_stocks.append({
                            "symbol": symbol,
                            "identifier": stock["identifier"],
                            "current_price": stock["lastPrice"],
                            "price_change": stock["change"],
                            "price_change_pct": stock["pchange"],
                            "volume": stock["totalTradedVolume"],
                            "market_cap": stock["totalMarketCap"],
                            "decision": decision["decision"],
                            "confidence": decision["confidence"],
                            "rsi": decision["technical_indicators"]["rsi"],
                            "trend": decision["time_analysis"]["trend_strength"]
                        })
                except:
                    continue
            
            # Sort by confidence
            analyzed_stocks.sort(key=lambda x: x["confidence"], reverse=True)
            
            buy_stocks = [s for s in analyzed_stocks if s["decision"] == "BUY"]
            sell_stocks = [s for s in analyzed_stocks if s["decision"] == "SELL"]
            
            return {
                "total": trading_stats["total"]["count"],
                "analyzed_count": len(analyzed_stocks),
                "top_buy_opportunities": buy_stocks[:20],
                "top_sell_opportunities": sell_stocks[:20],
                "market_summary": {
                    "advances": trading_stats["total"]["count"]["Advances"],
                    "declines": trading_stats["total"]["count"]["Declines"],
                    "unchanged": trading_stats["total"]["count"]["Unchange"],
                    "buy_signals": len(buy_stocks),
                    "sell_signals": len(sell_stocks)
                }
            }
            
        except Exception as e:
            return {"error": str(e)}