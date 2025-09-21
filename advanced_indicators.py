import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

class AdvancedTechnicalIndicators:
    """Advanced technical indicators for enhanced market analysis"""
    
    @staticmethod
    def bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2) -> Dict[str, List[float]]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return {"upper": [], "middle": [], "lower": []}
        
        df = pd.DataFrame({'price': prices})
        df['sma'] = df['price'].rolling(window=period).mean()
        df['std'] = df['price'].rolling(window=period).std()
        
        df['upper'] = df['sma'] + (df['std'] * std_dev)
        df['lower'] = df['sma'] - (df['std'] * std_dev)
        
        return {
            "upper": df['upper'].fillna(0).tolist(),
            "middle": df['sma'].fillna(0).tolist(),
            "lower": df['lower'].fillna(0).tolist()
        }
    
    @staticmethod
    def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, List[float]]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow:
            return {"macd": [], "signal": [], "histogram": []}
        
        df = pd.DataFrame({'price': prices})
        df['ema_fast'] = df['price'].ewm(span=fast).mean()
        df['ema_slow'] = df['price'].ewm(span=slow).mean()
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['signal'] = df['macd'].ewm(span=signal).mean()
        df['histogram'] = df['macd'] - df['signal']
        
        return {
            "macd": df['macd'].fillna(0).tolist(),
            "signal": df['signal'].fillna(0).tolist(),
            "histogram": df['histogram'].fillna(0).tolist()
        }
    
    @staticmethod
    def stochastic_oscillator(highs: List[float], lows: List[float], closes: List[float], 
                            k_period: int = 14, d_period: int = 3) -> Dict[str, List[float]]:
        """Calculate Stochastic Oscillator"""
        if len(closes) < k_period:
            return {"k": [], "d": []}
        
        df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
        df['lowest_low'] = df['low'].rolling(window=k_period).min()
        df['highest_high'] = df['high'].rolling(window=k_period).max()
        
        df['k'] = 100 * ((df['close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low']))
        df['d'] = df['k'].rolling(window=d_period).mean()
        
        return {
            "k": df['k'].fillna(0).tolist(),
            "d": df['d'].fillna(0).tolist()
        }
    
    @staticmethod
    def williams_r(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
        """Calculate Williams %R"""
        if len(closes) < period:
            return []
        
        df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
        df['highest_high'] = df['high'].rolling(window=period).max()
        df['lowest_low'] = df['low'].rolling(window=period).min()
        
        df['williams_r'] = -100 * ((df['highest_high'] - df['close']) / (df['highest_high'] - df['lowest_low']))
        
        return df['williams_r'].fillna(0).tolist()
    
    @staticmethod
    def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
        """Calculate Average True Range (ATR)"""
        if len(closes) < 2:
            return []
        
        df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
        df['prev_close'] = df['close'].shift(1)
        
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=period).mean()
        
        return df['atr'].fillna(0).tolist()
    
    @staticmethod
    def fibonacci_retracement(high: float, low: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        diff = high - low
        
        return {
            "0%": high,
            "23.6%": high - 0.236 * diff,
            "38.2%": high - 0.382 * diff,
            "50%": high - 0.5 * diff,
            "61.8%": high - 0.618 * diff,
            "78.6%": high - 0.786 * diff,
            "100%": low
        }
    
    @staticmethod
    def pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate pivot points and support/resistance levels"""
        pivot = (high + low + close) / 3
        
        return {
            "pivot": pivot,
            "r1": 2 * pivot - low,
            "r2": pivot + (high - low),
            "r3": high + 2 * (pivot - low),
            "s1": 2 * pivot - high,
            "s2": pivot - (high - low),
            "s3": low - 2 * (high - pivot)
        }
    
    @staticmethod
    def ichimoku_cloud(highs: List[float], lows: List[float], closes: List[float]) -> Dict[str, List[float]]:
        """Calculate Ichimoku Cloud components"""
        if len(closes) < 52:
            return {"tenkan": [], "kijun": [], "senkou_a": [], "senkou_b": [], "chikou": []}
        
        df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
        
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        df['tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        df['kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
        
        # Senkou Span A: (Conversion Line + Base Line)/2
        df['senkou_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(26)
        
        # Senkou Span B: (52-period high + 52-period low)/2
        df['senkou_b'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
        
        # Chikou Span: Close shifted back 26 periods
        df['chikou'] = df['close'].shift(-26)
        
        return {
            "tenkan": df['tenkan'].fillna(0).tolist(),
            "kijun": df['kijun'].fillna(0).tolist(),
            "senkou_a": df['senkou_a'].fillna(0).tolist(),
            "senkou_b": df['senkou_b'].fillna(0).tolist(),
            "chikou": df['chikou'].fillna(0).tolist()
        }
    
    @staticmethod
    def volume_profile(prices: List[float], volumes: List[float], bins: int = 20) -> Dict[str, List]:
        """Calculate Volume Profile"""
        if len(prices) != len(volumes) or len(prices) < bins:
            return {"price_levels": [], "volume_levels": []}
        
        df = pd.DataFrame({'price': prices, 'volume': volumes})
        
        # Create price bins
        price_min, price_max = min(prices), max(prices)
        price_bins = np.linspace(price_min, price_max, bins + 1)
        
        # Assign each price to a bin
        df['price_bin'] = pd.cut(df['price'], bins=price_bins, include_lowest=True)
        
        # Sum volume for each price bin
        volume_profile = df.groupby('price_bin')['volume'].sum()
        
        price_levels = [(interval.left + interval.right) / 2 for interval in volume_profile.index]
        volume_levels = volume_profile.values.tolist()
        
        return {
            "price_levels": price_levels,
            "volume_levels": volume_levels
        }
    
    @staticmethod
    def calculate_all_indicators(ohlcv_data: List[Dict]) -> Dict:
        """Calculate all technical indicators for OHLCV data"""
        if not ohlcv_data or len(ohlcv_data) < 20:
            return {"error": "Insufficient data for technical analysis"}
        
        # Extract OHLCV arrays
        opens = [float(d.get('o', 0)) for d in ohlcv_data]
        highs = [float(d.get('h', 0)) for d in ohlcv_data]
        lows = [float(d.get('l', 0)) for d in ohlcv_data]
        closes = [float(d.get('c', 0)) for d in ohlcv_data]
        volumes = [float(d.get('v', 0)) for d in ohlcv_data]
        
        # Calculate all indicators
        indicators = {}
        
        try:
            indicators['bollinger_bands'] = AdvancedTechnicalIndicators.bollinger_bands(closes)
            indicators['macd'] = AdvancedTechnicalIndicators.macd(closes)
            indicators['stochastic'] = AdvancedTechnicalIndicators.stochastic_oscillator(highs, lows, closes)
            indicators['williams_r'] = AdvancedTechnicalIndicators.williams_r(highs, lows, closes)
            indicators['atr'] = AdvancedTechnicalIndicators.atr(highs, lows, closes)
            
            # Current levels
            current_high = highs[-1]
            current_low = lows[-1]
            current_close = closes[-1]
            
            indicators['fibonacci'] = AdvancedTechnicalIndicators.fibonacci_retracement(
                max(highs[-20:]), min(lows[-20:])
            )
            indicators['pivot_points'] = AdvancedTechnicalIndicators.pivot_points(
                current_high, current_low, current_close
            )
            
            if len(closes) >= 52:
                indicators['ichimoku'] = AdvancedTechnicalIndicators.ichimoku_cloud(highs, lows, closes)
            
            indicators['volume_profile'] = AdvancedTechnicalIndicators.volume_profile(closes, volumes)
            
            # Current signal analysis
            indicators['signals'] = AdvancedTechnicalIndicators.analyze_signals(indicators, closes[-1])
            
        except Exception as e:
            indicators['error'] = f"Error calculating indicators: {str(e)}"
        
        return indicators
    
    @staticmethod
    def analyze_signals(indicators: Dict, current_price: float) -> Dict:
        """Analyze current trading signals from indicators"""
        signals = {
            "overall_signal": "NEUTRAL",
            "strength": 0,
            "bullish_signals": [],
            "bearish_signals": []
        }
        
        try:
            # Bollinger Bands signals
            if 'bollinger_bands' in indicators:
                bb = indicators['bollinger_bands']
                if bb['upper'] and bb['lower']:
                    upper = bb['upper'][-1]
                    lower = bb['lower'][-1]
                    middle = bb['middle'][-1]
                    
                    if current_price > upper:
                        signals['bearish_signals'].append("Price above Bollinger upper band (overbought)")
                    elif current_price < lower:
                        signals['bullish_signals'].append("Price below Bollinger lower band (oversold)")
                    elif current_price > middle:
                        signals['bullish_signals'].append("Price above Bollinger middle line")
            
            # MACD signals
            if 'macd' in indicators:
                macd = indicators['macd']
                if len(macd['macd']) >= 2 and len(macd['signal']) >= 2:
                    macd_current = macd['macd'][-1]
                    macd_prev = macd['macd'][-2]
                    signal_current = macd['signal'][-1]
                    signal_prev = macd['signal'][-2]
                    
                    if macd_current > signal_current and macd_prev <= signal_prev:
                        signals['bullish_signals'].append("MACD bullish crossover")
                    elif macd_current < signal_current and macd_prev >= signal_prev:
                        signals['bearish_signals'].append("MACD bearish crossover")
            
            # Stochastic signals
            if 'stochastic' in indicators:
                stoch = indicators['stochastic']
                if stoch['k'] and stoch['d']:
                    k_current = stoch['k'][-1]
                    d_current = stoch['d'][-1]
                    
                    if k_current < 20 and d_current < 20:
                        signals['bullish_signals'].append("Stochastic oversold")
                    elif k_current > 80 and d_current > 80:
                        signals['bearish_signals'].append("Stochastic overbought")
            
            # Calculate overall signal strength
            bullish_count = len(signals['bullish_signals'])
            bearish_count = len(signals['bearish_signals'])
            
            if bullish_count > bearish_count:
                signals['overall_signal'] = "BULLISH"
                signals['strength'] = min(bullish_count * 20, 100)
            elif bearish_count > bullish_count:
                signals['overall_signal'] = "BEARISH"
                signals['strength'] = min(bearish_count * 20, 100)
            else:
                signals['strength'] = 50
                
        except Exception as e:
            signals['error'] = f"Error analyzing signals: {str(e)}"
        
        return signals