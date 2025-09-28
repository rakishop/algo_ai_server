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
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
        os.makedirs('option_chain_data', exist_ok=True)
        
    def get_futures_historical_data(self, symbol, days_back=60):
        """Get futures historical data from NSE API"""
        try:
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
            
            response = requests.get(self.futures_url, params=params, headers=self.headers, timeout=10)
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
        if len(high) < period + 1:
            return np.mean(high) - np.mean(low)
        
        tr_list = []
        for i in range(1, len(high)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)
        
        return np.mean(tr_list[-period:])
    
    def find_support_resistance(self, high, low, close):
        """Find key support and resistance levels"""
        if len(close) < 10:
            current = close[-1]
            return current * 0.95, current * 1.05
        
        # Simple pivot points
        recent_high = max(high[-20:]) if len(high) >= 20 else max(high)
        recent_low = min(low[-20:]) if len(low) >= 20 else min(low)
        
        return recent_low, recent_high
    
    def calculate_ai_levels(self, symbol, current_price, action):
        """Calculate AI-based stop loss and target using historical data"""
        # Try futures data first, fallback to chart data
        hist_data = self.get_futures_historical_data(symbol, days_back=60)
        if not hist_data:
            hist_data = self.get_historical_data(symbol, period="D", interval=1, days_back=60)
        
        if not hist_data or not hist_data["close"]:
            # Fallback to basic calculation
            return self._fallback_calculation(current_price, action)
        
        close_prices = hist_data["close"]
        high_prices = hist_data["high"]
        low_prices = hist_data["low"]
        
        # Calculate technical indicators
        volatility = self.calculate_volatility(close_prices)
        atr = self.calculate_atr(high_prices, low_prices, close_prices)
        support, resistance = self.find_support_resistance(high_prices, low_prices, close_prices)
        
        # AI-based calculation
        atr_factor = atr / current_price * 100  # ATR as percentage
        vol_factor = volatility / 20  # Normalize volatility
        
        # Dynamic stop loss based on ATR and volatility
        if action in ["BUY", "STRONG BUY"]:
            # For buy positions
            sl_pct = min(max(atr_factor * 1.5, 2.0), 8.0)  # 2-8% range
            target_pct = sl_pct * (2 + vol_factor)  # Dynamic R:R based on volatility
            
            # Use support/resistance levels
            technical_sl = max(support, current_price * (1 - sl_pct/100))
            technical_target = min(resistance, current_price * (1 + target_pct/100))
            
        else:  # SELL positions
            sl_pct = min(max(atr_factor * 1.5, 2.0), 8.0)
            target_pct = sl_pct * (2 + vol_factor)
            
            technical_sl = min(resistance, current_price * (1 + sl_pct/100))
            technical_target = max(support, current_price * (1 - target_pct/100))
        
        return {
            "stop_loss": round(technical_sl, 2),
            "target": round(technical_target, 2),
            "sl_percentage": round(abs(technical_sl - current_price) / current_price * 100, 1),
            "target_percentage": round(abs(technical_target - current_price) / current_price * 100, 1),
            "atr": round(atr, 2),
            "volatility": round(volatility, 1),
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "risk_reward_ratio": round(abs(technical_target - current_price) / abs(technical_sl - current_price), 2)
        }
    
    def _fallback_calculation(self, current_price, action):
        """Fallback calculation when historical data unavailable"""
        base_sl = 3.0
        base_target = 6.0
        
        if action in ["BUY", "STRONG BUY"]:
            stop_loss = current_price * (1 - base_sl/100)
            target = current_price * (1 + base_target/100)
        else:
            stop_loss = current_price * (1 + base_sl/100)
            target = current_price * (1 - base_target/100)
        
        return {
            "stop_loss": round(stop_loss, 2),
            "target": round(target, 2),
            "sl_percentage": base_sl,
            "target_percentage": base_target,
            "atr": 0,
            "volatility": 0,
            "support": 0,
            "resistance": 0,
            "risk_reward_ratio": 2.0
        }
    
    def get_option_chain_data(self, symbol):
        """Get current option chain data from NSE"""
        try:
            # Create session for NSE API
            session = requests.Session()
            session.headers.update(self.headers)
            
            # First get NSE homepage to establish session
            session.get('https://www.nseindia.com', timeout=10)
            
            params = {'symbol': symbol}
            response = session.get(self.option_chain_url, params=params, timeout=15)
            
            print(f"Response status: {response.status_code}")
            print(f"Response text preview: {response.text[:200]}")
            
            if response.status_code == 200 and response.text.strip():
                data = response.json()
            else:
                print(f"Empty or invalid response for {symbol}")
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
    
    def analyze_option_chain(self, option_chain, strike_price, option_type):
        """Analyze option chain data for insights"""
        if not option_chain or 'records' not in option_chain:
            return {}
        
        try:
            data = option_chain['records']['data']
            underlying_value = option_chain['records'].get('underlyingValue', 0)
            
            # Find matching strike
            target_strike = None
            for strike_data in data:
                if strike_data.get('strikePrice') == strike_price:
                    target_strike = strike_data
                    break
            
            if not target_strike:
                return {'underlying_value': underlying_value}
            
            option_data = target_strike.get(option_type, {})
            
            return {
                'underlying_value': underlying_value,
                'implied_volatility': option_data.get('impliedVolatility', 0),
                'open_interest': option_data.get('openInterest', 0),
                'volume': option_data.get('totalTradedVolume', 0),
                'bid_ask_spread': abs(option_data.get('askPrice', 0) - option_data.get('bidprice', 0)),
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
            return self._fallback_options_calculation(current_price, action, expiry_days)
        
        close_prices = hist_data["close"]
        volatility = self.calculate_volatility(close_prices)
        
        # Use implied volatility if available
        iv = chain_analysis.get('implied_volatility', 0)
        if iv > 0:
            volatility = iv
        
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
            "risk_reward_ratio": round(base_target/base_sl, 2)
        }
    
    def _fallback_options_calculation(self, current_price, action, expiry_days):
        """Fallback for options when no historical data"""
        time_factor = max(0.5, expiry_days / 30)
        base_sl = 25 * time_factor
        base_target = base_sl * 2
        
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
            "volatility": 0,
            "time_factor": round(time_factor, 2),
            "risk_reward_ratio": 2.0
        }
    
    def calculate_futures_levels(self, symbol, current_price, action):
        """Calculate levels specifically for futures contracts"""
        # Get current option chain data for additional insights
        option_chain = self.get_option_chain_data(symbol)
        
        # Get futures historical data
        hist_data = self.get_futures_historical_data(symbol, days_back=90)
        
        if not hist_data or not hist_data["close"]:
            return self._fallback_calculation(current_price, action)
        
        close_prices = hist_data["close"]
        high_prices = hist_data["high"]
        low_prices = hist_data["low"]
        
        # Calculate technical indicators
        volatility = self.calculate_volatility(close_prices)
        atr = self.calculate_atr(high_prices, low_prices, close_prices)
        support, resistance = self.find_support_resistance(high_prices, low_prices, close_prices)
        
        # Futures-specific calculation (tighter stops due to leverage)
        atr_factor = atr / current_price * 100
        vol_factor = volatility / 25
        
        if action in ["BUY", "STRONG BUY"]:
            sl_pct = min(max(atr_factor * 1.2, 1.5), 4.0)
            target_pct = sl_pct * (2.5 + vol_factor)
            
            technical_sl = max(support, current_price * (1 - sl_pct/100))
            technical_target = min(resistance, current_price * (1 + target_pct/100))
        else:
            sl_pct = min(max(atr_factor * 1.2, 1.5), 4.0)
            target_pct = sl_pct * (2.5 + vol_factor)
            
            technical_sl = min(resistance, current_price * (1 + sl_pct/100))
            technical_target = max(support, current_price * (1 - target_pct/100))
        
        return {
            "stop_loss": round(technical_sl, 2),
            "target": round(technical_target, 2),
            "sl_percentage": round(abs(technical_sl - current_price) / current_price * 100, 1),
            "target_percentage": round(abs(technical_target - current_price) / current_price * 100, 1),
            "atr": round(atr, 2),
            "volatility": round(volatility, 1),
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "risk_reward_ratio": round(abs(technical_target - current_price) / abs(technical_sl - current_price), 2)
        }