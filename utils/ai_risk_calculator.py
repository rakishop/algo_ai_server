import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import os

class AIRiskCalculator:
    def __init__(self):
        self.chart_url = "https://charting.nseindia.com/Charts/ChartData/"
        self.futures_url = "https://www.nseindia.com/api/historicalOR/fo/derivatives"
        self.option_chain_url = "https://www.nseindia.com/api/option-chain-indices"
        self.stock_chart_url = "https://www.nseindia.com/api/chart-databyindex-dynamic"
        self.stock_historical_url = "https://www.nseindia.com/api/historicalOR/cm/equity"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
        os.makedirs('option_chain_data', exist_ok=True)
        
        # Add caching for performance
        self._data_cache = {}
        self._cache_timeout = 300  # 5 minutes
        
    def _get_cached_historical_data(self, cache_key, symbol, days_back):
        """Get cached historical data or fetch if not available"""
        current_time = time.time()
        
        # Check cache first
        if cache_key in self._data_cache:
            cached_data, timestamp = self._data_cache[cache_key]
            if (current_time - timestamp) < self._cache_timeout:
                return cached_data
        
        # Fetch new data with proper session handling
        hist_data = None
        try:
            if symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY']:
                hist_data = self.get_futures_historical_data(symbol, days_back)
            else:
                hist_data = self.get_stock_historical_data(symbol, days_back)
        except Exception as e:
            print(f"Historical data fetch failed for {symbol}: {e}")
            return None
        
        # Cache the result only if valid
        if hist_data and hist_data.get("close"):
            self._data_cache[cache_key] = (hist_data, current_time)
            return hist_data
        
        return None
        
    def get_futures_historical_data(self, symbol, days_back=60):
        """Get futures historical data from NSE API"""
        try:
            # Create fresh session with updated headers
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36',
                'Accept': '*/*',
                'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
                'Referer': 'https://www.nseindia.com/market-data/live-equity-market',
                'sec-ch-ua': '"Chromium";v="140", "Not=A?Brand";v="24", "Google Chrome";v="140"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-origin',
                'Cache-Control': 'no-cache'
            })
            
            # Establish session with proper page
            session.get('https://www.nseindia.com/market-data/live-equity-market', timeout=15)
            
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)
            
            # Determine instrument type
            if symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY']:
                instrument_type = 'FUTIDX'
            else:
                instrument_type = 'FUTSTK'
            
            params = {
                'from': from_date.strftime('%d-%m-%Y'),
                'to': to_date.strftime('%d-%m-%Y'),
                'instrumentType': instrument_type,
                'symbol': symbol
            }
            
            response = session.get(self.futures_url, params=params, timeout=15)
            
            if response.status_code != 200:
                print(f"NSE Futures API returned {response.status_code} for {symbol}")
                return None
            
            if not response.text.strip():
                print(f"Empty response from NSE Futures API for {symbol}")
                return None
                
            data = response.json()
            
            if 'data' in data and data['data']:
                # Convert to OHLCV format
                futures_data = data['data']
                return {
                    'open': [float(d['FH_OPENING_PRICE']) for d in futures_data],
                    'high': [float(d['FH_TRADE_HIGH_PRICE']) for d in futures_data],
                    'low': [float(d['FH_TRADE_LOW_PRICE']) for d in futures_data],
                    'close': [float(d['FH_CLOSING_PRICE']) for d in futures_data],
                    'volume': [int(d['FH_TOT_TRADED_QTY']) for d in futures_data],
                    'timestamps': [d['FH_TIMESTAMP'] for d in futures_data]
                }
            return None
            
        except Exception as e:
            print(f"Futures data error: {e}")
            return None
    
    def get_historical_data(self, symbol, period="D", interval=1, days_back=30):
        """Get historical chart data from NSE"""
        try:
            # Calculate timestamps
            to_date = int(time.time())
            from_date = to_date - (days_back * 24 * 60 * 60)
            
            # Format symbol for NSE
            if not symbol.endswith("-EQ"):
                symbol = f"{symbol}-EQ"
            
            payload = {
                "chartPeriod": period,
                "chartStart": 0,
                "exch": "N",
                "fromDate": from_date,
                "timeInterval": interval,
                "toDate": to_date,
                "tradingSymbol": symbol
            }
            
            response = requests.post(self.chart_url, data=payload, timeout=10)
            data = response.json()
            
            if data.get("s") == "Ok":
                return {
                    "timestamps": data.get("t", []),
                    "open": data.get("o", []),
                    "high": data.get("h", []),
                    "low": data.get("l", []),
                    "close": data.get("c", []),
                    "volume": data.get("v", [])
                }
            return None
            
        except Exception as e:
            print(f"Chart data error: {e}")
            return None
    
    def calculate_volatility(self, prices):
        """Calculate historical volatility"""
        if len(prices) < 2:
            return 2.0
        
        returns = np.diff(np.log(prices))
        volatility = np.std(returns) * np.sqrt(252) * 100  # Annualized
        return min(max(volatility, 1.0), 50.0)  # Cap between 1-50%
    
    def calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range"""
        if len(high) < 2 or len(low) < 2 or len(close) < 2:
            return abs(np.mean(high) - np.mean(low)) if len(high) > 0 else 0
        
        # Ensure all arrays have same length
        min_len = min(len(high), len(low), len(close))
        high = high[:min_len]
        low = low[:min_len]
        close = close[:min_len]
        
        if min_len < period + 1:
            return abs(np.mean(high) - np.mean(low))
        
        tr_list = []
        for i in range(1, min_len):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)
        
        return np.mean(tr_list[-period:]) if tr_list else 0
    
    def find_support_resistance(self, high, low, close):
        """Find key support and resistance levels"""
        if len(close) < 1:
            return 0, 0
        
        if len(close) < 10:
            current = close[-1]
            return current * 0.95, current * 1.05
        
        # Ensure arrays have same length
        min_len = min(len(high), len(low), len(close))
        high = high[:min_len]
        low = low[:min_len]
        
        # Simple pivot points
        recent_high = max(high[-20:]) if len(high) >= 20 else max(high)
        recent_low = min(low[-20:]) if len(low) >= 20 else min(low)
        
        return recent_low, recent_high
    
    def calculate_ai_levels(self, symbol, current_price, action):
        """Calculate AI-based stop loss and target using real historical data only"""
        # Use cached historical data to avoid repeated API calls
        cache_key = f"hist_data_{symbol}_60d"
        hist_data = self._get_cached_historical_data(cache_key, symbol, 60)
        
        if not hist_data or not hist_data["close"]:
            # Return null if no real data available
            return None
        
        close_prices = hist_data["close"]
        high_prices = hist_data["high"]
        low_prices = hist_data["low"]
        
        # Calculate technical indicators from real data
        volatility = self.calculate_volatility(close_prices)
        atr = self.calculate_atr(high_prices, low_prices, close_prices)
        support, resistance = self.find_support_resistance(high_prices, low_prices, close_prices)
        
        # AI-based calculation using real data
        atr_factor = atr / current_price * 100
        vol_factor = volatility / 20
        sl_pct = min(max(atr_factor * 1.5, 2.0), 8.0)
        target_pct = sl_pct * (2 + vol_factor)
        
        if action in ["BUY", "STRONG BUY"]:
            stop_loss = current_price * (1 - sl_pct/100)
            target = current_price * (1 + target_pct/100)
        else:
            stop_loss = current_price * (1 + sl_pct/100)
            target = current_price * (1 - target_pct/100)
        
        return {
            "stop_loss": round(stop_loss, 2),
            "target": round(target, 2),
            "sl_percentage": round(sl_pct, 1),
            "target_percentage": round(target_pct, 1),
            "atr": round(atr, 2),
            "volatility": round(volatility, 1),
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "risk_reward_ratio": round(target_pct/sl_pct, 2)
        }
    

    
    def get_option_chain_data(self, symbol):
        """Get current option chain data from NSE"""
        try:
            # Create fresh session with updated headers
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36',
                'Accept': '*/*',
                'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
                'Referer': 'https://www.nseindia.com/option-chain',
                'sec-ch-ua': '"Chromium";v="140", "Not=A?Brand";v="24", "Google Chrome";v="140"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-origin',
                'Cache-Control': 'no-cache'
            })
            
            # Establish session with option chain page
            session.get('https://www.nseindia.com/option-chain', timeout=15)
            
            params = {'symbol': symbol}
            response = session.get(self.option_chain_url, params=params, timeout=20)
            
            if response.status_code != 200:
                print(f"Option chain API returned {response.status_code} for {symbol}")
                return None
            
            if not response.text.strip():
                print(f"Empty response from option chain API for {symbol}")
                return None
            
            try:
                data = response.json()
            except json.JSONDecodeError:
                print(f"Invalid JSON response for {symbol}")
                return None
            
            # Save response to JSON file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'option_chain_data/{symbol}_option_chain_{timestamp}.json'
            
            # Ensure absolute path
            abs_filename = os.path.abspath(filename)
            
            with open(abs_filename, 'w') as f:
                json.dump(data, f, indent=2)
                
            filename = abs_filename
            
            print(f"Option chain data saved to: {filename}")
            return data
            
        except Exception as e:
            print(f"Option chain data error for {symbol}: {e}")
            return None
    
    def get_stock_chart_data(self, symbol):
        """Get stock chart data from NSE"""
        try:
            # Format symbol for NSE (e.g., HDFCBANK -> HDFCBANKEQN)
            if not symbol.endswith('EQN'):
                symbol = f"{symbol}EQN"
            
            params = {
                'index': symbol,
                'type': 'symbol'
            }
            
            response = requests.get(self.stock_chart_url, params=params, headers=self.headers, timeout=10)
            data = response.json()
            
            if 'grapthData' in data:
                # Convert to OHLCV format
                chart_data = data['grapthData']
                return {
                    'timestamps': [item[0] for item in chart_data],
                    'prices': [float(item[1]) for item in chart_data],
                    'close_price': data.get('closePrice', 0),
                    'symbol': data.get('name', symbol)
                }
            return None
            
        except Exception as e:
            print(f"Stock chart data error: {e}")
            return None
    
    def get_stock_historical_data(self, symbol, days_back=60):
        """Get stock historical data from NSE API"""
        try:
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36',
                'Accept': '*/*',
                'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
                'Referer': 'https://www.nseindia.com/market-data/live-equity-market',
                'Cache-Control': 'no-cache'
            })
            session.get('https://www.nseindia.com/market-data/live-equity-market', timeout=15)
            
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)
            
            params = {
                'symbol': symbol,
                'series': '["EQ"]',
                'from': from_date.strftime('%d-%m-%Y'),
                'to': to_date.strftime('%d-%m-%Y')
            }
            print(params)
            response = session.get(self.stock_historical_url, params=params, timeout=15)
            
            if response.status_code != 200:
                print(f"NSE Stock API returned {response.status_code} for {symbol}")
                return None
            
            if not response.text.strip():
                print(f"Empty response from NSE Stock API for {symbol}")
                return None
                
            data = response.json()
            
            if 'data' in data and data['data']:
                hist_data = data['data']
                return {
                    'open': [float(d['CH_OPENING_PRICE']) for d in hist_data],
                    'high': [float(d['CH_TRADE_HIGH_PRICE']) for d in hist_data],
                    'low': [float(d['CH_TRADE_LOW_PRICE']) for d in hist_data],
                    'close': [float(d['CH_CLOSING_PRICE']) for d in hist_data],
                    'volume': [int(d['CH_TOT_TRADED_QTY']) for d in hist_data],
                    'timestamps': [d['CH_TIMESTAMP'] for d in hist_data]
                }
            return None
            
        except Exception as e:
            return None
    
    def analyze_option_chain(self, option_chain, strike_price, option_type):
        """Analyze option chain data for insights"""
        if not option_chain or 'records' not in option_chain:
            return {}
        
        try:
            data = option_chain['records']['data']
            
            # Find matching strike and expiry
            target_option = None
            underlying_value = 0
            
            for strike_data in data:
                if strike_data.get('strikePrice') == strike_price:
                    option_data = strike_data.get(option_type, {})
                    if option_data:
                        target_option = option_data
                        underlying_value = option_data.get('underlyingValue', 0)
                        break
            
            if not target_option:
                # Get underlying value from any available option
                for strike_data in data:
                    for opt_type in ['CE', 'PE']:
                        opt_data = strike_data.get(opt_type, {})
                        if opt_data and opt_data.get('underlyingValue'):
                            underlying_value = opt_data['underlyingValue']
                            break
                    if underlying_value:
                        break
                
                return {'underlying_value': underlying_value}
            
            return {
                'underlying_value': underlying_value,
                'implied_volatility': target_option.get('impliedVolatility', 0),
                'open_interest': target_option.get('openInterest', 0),
                'volume': target_option.get('totalTradedVolume', 0),
                'bid_ask_spread': abs(target_option.get('askPrice', 0) - target_option.get('bidprice', 0)),
                'last_price': target_option.get('lastPrice', 0),
                'change': target_option.get('change', 0),
                'pChange': target_option.get('pChange', 0),
                'delta_approx': self._calculate_delta_approx(underlying_value, strike_price, option_type)
            }
            
        except Exception as e:
            print(f"Option chain analysis error: {e}")
            return {}
    
    def _calculate_delta_approx(self, spot, strike, option_type):
        """Approximate delta calculation"""
        if spot == 0 or strike == 0:
            return 0
        
        moneyness = spot / strike
        
        if option_type == 'CE':  # Call
            if moneyness > 1.05:
                return 0.8  # Deep ITM
            elif moneyness > 0.95:
                return 0.5  # ATM
            else:
                return 0.2  # OTM
        else:  # Put
            if moneyness < 0.95:
                return -0.8  # Deep ITM
            elif moneyness < 1.05:
                return -0.5  # ATM
            else:
                return -0.2  # OTM
    
    def calculate_options_levels(self, symbol, current_price, action, option_type, strike_price, expiry_days):
        """Calculate levels for options with time decay consideration"""
        # Get current option chain data
        option_chain = self.get_option_chain_data(symbol)
        chain_analysis = self.analyze_option_chain(option_chain, strike_price, option_type)
        
        # Try futures data first for underlying, fallback to chart data
        hist_data = self.get_futures_historical_data(symbol, days_back=30)
        if not hist_data:
            hist_data = self.get_historical_data(symbol, period="I", interval=15, days_back=5)
        
        if not hist_data or not hist_data["close"]:
            return {"error": "No historical data available for options analysis"}
        
        close_prices = hist_data["close"]
        volatility = self.calculate_volatility(close_prices)
        
        # Use implied volatility if available and valid
        iv = chain_analysis.get('implied_volatility', 0)
        if iv > 0 and iv < 100:  # Valid IV range
            volatility = iv
        elif iv == 0:
            # Try to get IV from nearby strikes
            volatility = self._get_nearby_iv(option_chain, strike_price, option_type) or volatility
        
        # Time decay factor
        time_factor = max(0.3, expiry_days / 30)
        
        # Delta-adjusted calculation
        delta = abs(chain_analysis.get('delta_approx', 0.5))
        delta_factor = max(delta, 0.3)  # Minimum 0.3
        
        # Options-specific calculation with option chain insights
        base_sl = min(max(volatility * 1.5 * delta_factor, 15), 70)
        base_target = base_sl * (1.8 + time_factor)
        
        # Adjust for liquidity (bid-ask spread)
        spread = chain_analysis.get('bid_ask_spread', 0)
        if spread > current_price * 0.05:  # Wide spread
            base_sl *= 1.2
            base_target *= 1.1
        
        if "BUY" in action:
            stop_loss = current_price * (1 - base_sl/100)
            target = current_price * (1 + base_target/100)
        else:
            stop_loss = current_price * (1 + base_sl/100)
            target = current_price * (1 - base_target/100)
        
        return {
            "stop_loss": round(max(stop_loss, 0.05), 2),
            "target": round(target, 2),
            "sl_percentage": round(base_sl, 1),
            "target_percentage": round(base_target, 1),
            "volatility": round(volatility, 1),
            "implied_volatility": iv,
            "time_factor": round(time_factor, 2),
            "delta_approx": chain_analysis.get('delta_approx', 0),
            "open_interest": chain_analysis.get('open_interest', 0),
            "underlying_value": chain_analysis.get('underlying_value', 0),
            "risk_reward_ratio": round(base_target/base_sl, 2),
            "option_price": chain_analysis.get('last_price', current_price),
            "price_change": chain_analysis.get('change', 0),
            "price_change_percent": chain_analysis.get('pChange', 0)
        }
    
    def _get_nearby_iv(self, option_chain, target_strike, option_type):
        """Get IV from nearby strikes if target strike has no IV"""
        if not option_chain or 'records' not in option_chain:
            return None
        
        try:
            data = option_chain['records']['data']
            ivs = []
            
            for strike_data in data:
                strike = strike_data.get('strikePrice', 0)
                if abs(strike - target_strike) <= 500:  # Within 500 points
                    option_data = strike_data.get(option_type, {})
                    iv = option_data.get('impliedVolatility', 0)
                    if iv > 0 and iv < 100:
                        ivs.append(iv)
            
            return sum(ivs) / len(ivs) if ivs else None
            
        except:
            return None
    

    
    def calculate_futures_levels(self, symbol, current_price, action):
        """Calculate levels for futures using historical data and AI models"""
        # Use cached historical data for AI-based calculation
        cache_key = f"futures_hist_{symbol}_90d"
        hist_data = self._get_cached_historical_data(cache_key, symbol, 90)
        
        if hist_data and hist_data.get("close"):
            # AI-based calculation using real historical data
            close_prices = hist_data["close"]
            high_prices = hist_data["high"]
            low_prices = hist_data["low"]
            
            # Calculate advanced technical indicators
            volatility = self.calculate_volatility(close_prices)
            atr = self.calculate_atr(high_prices, low_prices, close_prices)
            support, resistance = self.find_support_resistance(high_prices, low_prices, close_prices)
            
            # Calculate price momentum and trend strength
            if len(close_prices) >= 20:
                recent_prices = close_prices[-20:]
                recent_returns = np.diff(recent_prices) / recent_prices[:-1] * 100
                momentum = np.mean(recent_returns)
                trend_strength = abs(momentum)
            else:
                momentum = 0
                trend_strength = 1
            
            # Dynamic risk calculation based on historical patterns
            atr_pct = (atr / current_price) * 100
            vol_factor = volatility / 20  # Normalize volatility
            
            # AI-enhanced stop loss calculation
            base_sl = max(atr_pct * 1.2, 1.5)  # Minimum 1.5%
            
            # Adjust for symbol characteristics
            if symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY']:
                base_sl = min(base_sl, 3.0)  # Cap index futures SL
            else:
                base_sl = min(base_sl, 6.0)  # Cap stock futures SL
            
            # Dynamic target calculation based on trend and volatility
            if trend_strength > 2:  # Strong trend
                target_multiplier = 2.5 + (vol_factor * 0.5)
            elif trend_strength > 1:  # Moderate trend
                target_multiplier = 2.0 + (vol_factor * 0.3)
            else:  # Weak trend
                target_multiplier = 1.8 + (vol_factor * 0.2)
            
            target_pct = base_sl * target_multiplier
            
            # Calculate support/resistance based levels
            if action in ["BUY", "STRONG BUY"]:
                stop_loss = current_price * (1 - base_sl/100)
                target = current_price * (1 + target_pct/100)
            else:  # SELL positions
                stop_loss = current_price * (1 + base_sl/100)
                target = current_price * (1 - target_pct/100)
            
            # Recalculate percentages based on adjusted levels
            actual_sl_pct = abs((stop_loss - current_price) / current_price * 100)
            actual_target_pct = abs((target - current_price) / current_price * 100)
            risk_reward = actual_target_pct / actual_sl_pct if actual_sl_pct > 0 else 2.0
            
            return {
                "stop_loss": round(stop_loss, 2),
                "target": round(target, 2),
                "sl_percentage": round(actual_sl_pct, 1),
                "target_percentage": round(actual_target_pct, 1),
                "atr": round(atr, 2),
                "volatility": round(volatility, 1),
                "support": round(support, 2),
                "resistance": round(resistance, 2),
                "risk_reward_ratio": round(risk_reward, 2),
                "momentum": round(momentum, 2),
                "trend_strength": round(trend_strength, 2),
                "data_source": "historical_ai_analysis"
            }
        
        # Fallback to enhanced static calculation if no historical data
        symbol_volatility = {
            'NIFTY': 12, 'BANKNIFTY': 15, 'FINNIFTY': 18,
            'SBIN': 20, 'ICICIBANK': 18, 'HDFCBANK': 16,
            'RELIANCE': 14, 'TCS': 12, 'INFY': 16
        }
        
        vol = symbol_volatility.get(symbol, 20)
        sl_pct = min(max(vol * 0.15, 2.0), 5.0)
        
        # Dynamic target based on volatility
        if vol > 25:
            target_multiplier = 3.0
        elif vol > 15:
            target_multiplier = 2.5
        else:
            target_multiplier = 2.0
            
        target_pct = sl_pct * target_multiplier
        
        if action in ["BUY", "STRONG BUY"]:
            stop_loss = current_price * (1 - sl_pct/100)
            target = current_price * (1 + target_pct/100)
        else:
            stop_loss = current_price * (1 + sl_pct/100)
            target = current_price * (1 - target_pct/100)
        
        return {
            "stop_loss": round(stop_loss, 2),
            "target": round(target, 2),
            "sl_percentage": round(sl_pct, 1),
            "target_percentage": round(target_pct, 1),
            "atr": round(current_price * (vol/100) * 0.02, 2),
            "volatility": vol,
            "support": round(current_price * 0.97, 2),
            "resistance": round(current_price * 1.03, 2),
            "risk_reward_ratio": round(target_multiplier, 2),
            "data_source": "enhanced_static_calculation"
        }
        
