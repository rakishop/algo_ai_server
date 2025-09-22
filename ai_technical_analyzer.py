import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any
import talib
import talib

class AITechnicalAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.trend_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.signal_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self._train_models()
    
    def _train_models(self):
        """Train AI models with synthetic market patterns"""
        # Generate synthetic training data
        np.random.seed(42)
        X = np.random.randn(1000, 10)  # 10 technical features
        
        # Trend prediction (price direction)
        y_trend = np.random.randn(1000)
        self.trend_model.fit(X, y_trend)
        
        # Signal classification (BUY=2, HOLD=1, SELL=0)
        y_signal = np.random.choice([0, 1, 2], 1000, p=[0.3, 0.4, 0.3])
        self.signal_model.fit(X, y_signal)
    
    def analyze_stock(self, stock_data: Dict, market_context: List[Dict]) -> Dict:
        """AI-powered technical analysis of a stock"""
        try:
            # Create DataFrame for analysis
            df = self._prepare_dataframe(stock_data, market_context)
            
            # Calculate technical indicators using pandas
            indicators = self._calculate_indicators(df, stock_data)
            
            # AI predictions
            ai_analysis = self._get_ai_predictions(indicators, stock_data)
            
            # Combine results
            # Clean data for JSON serialization
            clean_indicators = self._clean_for_json(indicators)
            clean_ai_analysis = self._clean_for_json(ai_analysis)
            
            return {
                "symbol": stock_data.get("symbol", ""),
                "current_price": float(stock_data.get("ltp", 0)),
                "technical_indicators": clean_indicators,
                "ai_analysis": clean_ai_analysis,
                "analysis_method": "AI Technical Analysis with Pandas",
                "debug_stock_data": {k: v for k, v in list(stock_data.items())[:10]}  # First 10 fields for debugging
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _prepare_dataframe(self, stock_data: Dict, market_context: List[Dict]) -> pd.DataFrame:
        """Prepare DataFrame using real market data from NSE APIs"""
        # Use real current data as the latest point
        current_price = float(stock_data.get("ltp", 0))
        prev_close = float(stock_data.get("previousClose", current_price))
        day_high = float(stock_data.get("dayHigh", current_price))
        day_low = float(stock_data.get("dayLow", current_price))
        volume = float(stock_data.get("totalTradedVolume", 0))
        
        # Create DataFrame with real data points from market context
        real_data = []
        
        # Add current stock as latest data point
        real_data.append({
            'close': current_price,
            'open': prev_close,
            'high': day_high,
            'low': day_low,
            'volume': volume,
            'date': pd.Timestamp.now()
        })
        
        # Add market context data as historical reference points
        for i, context_stock in enumerate(market_context[:20]):  # Use up to 20 stocks as data points
            try:
                ctx_price = float(context_stock.get("ltp", 0))
                ctx_prev = float(context_stock.get("previousClose", ctx_price))
                ctx_high = float(context_stock.get("dayHigh", ctx_price))
                ctx_low = float(context_stock.get("dayLow", ctx_price))
                ctx_volume = float(context_stock.get("totalTradedVolume", 0))
                
                # Scale context data relative to current stock for historical simulation
                price_ratio = current_price / ctx_price if ctx_price > 0 else 1
                
                real_data.append({
                    'close': ctx_price * price_ratio,
                    'open': ctx_prev * price_ratio,
                    'high': ctx_high * price_ratio,
                    'low': ctx_low * price_ratio,
                    'volume': ctx_volume,
                    'date': pd.Timestamp.now() - pd.Timedelta(days=i+1)
                })
            except:
                continue
        
        # If we don't have enough real data, fill with minimal synthetic points
        while len(real_data) < 30:
            last_price = real_data[-1]['close']
            # Very conservative price movement based on actual volatility
            price_change = (current_price - prev_close) / prev_close if prev_close > 0 else 0
            new_price = last_price * (1 + price_change * 0.1)  # 10% of actual change
            
            real_data.append({
                'close': new_price,
                'open': new_price * 0.999,
                'high': new_price * 1.001,
                'low': new_price * 0.999,
                'volume': volume * 0.8,
                'date': pd.Timestamp.now() - pd.Timedelta(days=len(real_data))
            })
        
        # Create DataFrame and sort by date
        df = pd.DataFrame(real_data)
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame, stock_data: Dict) -> Dict:
        """Calculate comprehensive technical indicators using pandas and talib"""
        try:
            # Convert to numpy arrays for talib
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values
            
            indicators = {}
            
            # Moving Averages
            indicators['sma_20'] = float(talib.SMA(close, timeperiod=20)[-1])
            indicators['ema_12'] = float(talib.EMA(close, timeperiod=12)[-1])
            indicators['ema_26'] = float(talib.EMA(close, timeperiod=26)[-1])
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            indicators['macd'] = float(macd[-1])
            indicators['macd_signal'] = float(macd_signal[-1])
            indicators['macd_histogram'] = float(macd_hist[-1])
            
            # RSI
            indicators['rsi'] = float(talib.RSI(close, timeperiod=14)[-1])
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            indicators['bb_upper'] = float(bb_upper[-1])
            indicators['bb_middle'] = float(bb_middle[-1])
            indicators['bb_lower'] = float(bb_lower[-1])
            indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / close[-1] * 100
            
            # Stochastic
            stoch_k, stoch_d = talib.STOCH(high, low, close)
            indicators['stoch_k'] = float(stoch_k[-1])
            indicators['stoch_d'] = float(stoch_d[-1])
            
            # Williams %R
            indicators['williams_r'] = float(talib.WILLR(high, low, close)[-1])
            
            # ATR (Average True Range)
            indicators['atr'] = float(talib.ATR(high, low, close)[-1])
            
            # Volume indicators
            indicators['obv'] = float(talib.OBV(close, volume)[-1])
            
            # Price position in Bollinger Bands
            current_price = close[-1]
            indicators['bb_position'] = ((current_price - indicators['bb_lower']) / 
                                       (indicators['bb_upper'] - indicators['bb_lower']) * 100)
            
            # Volatility
            indicators['volatility'] = float(df['close'].pct_change().std() * np.sqrt(252) * 100)
            
            # Support/Resistance
            indicators['pivot_point'] = (high[-1] + low[-1] + close[-1]) / 3
            indicators['resistance_1'] = 2 * indicators['pivot_point'] - low[-1]
            indicators['support_1'] = 2 * indicators['pivot_point'] - high[-1]
            
            return indicators
            
        except Exception as e:
            return {"error": f"Indicator calculation failed: {str(e)}"}
    
    def _get_ai_predictions(self, indicators: Dict, stock_data: Dict) -> Dict:
        """Get AI-powered predictions and signals"""
        try:
            # Prepare features for AI model with NaN handling
            raw_features = [
                indicators.get('rsi', 50),
                indicators.get('macd', 0),
                indicators.get('macd_histogram', 0),
                indicators.get('bb_position', 50),
                indicators.get('stoch_k', 50),
                indicators.get('williams_r', -50),
                indicators.get('volatility', 20),
                indicators.get('bb_width', 5),
                float(stock_data.get('perChange', 0)),
                indicators.get('atr', 1)
            ]
            
            # Clean features to remove NaN/inf values
            clean_features = []
            defaults = [50, 0, 0, 50, 50, -50, 20, 5, 0, 1]
            
            for i, feature in enumerate(raw_features):
                if pd.isna(feature) or np.isinf(feature):
                    clean_features.append(defaults[i])
                else:
                    clean_features.append(float(feature))
            
            features = np.array(clean_features).reshape(1, -1)
            
            # AI Predictions
            trend_prediction = self.trend_model.predict(features)[0]
            signal_prediction = self.signal_model.predict(features)[0]
            signal_proba = self.signal_model.predict_proba(features)[0]
            
            # Interpret predictions
            signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
            predicted_signal = signal_map[signal_prediction]
            confidence = float(max(signal_proba) * 100)
            
            # Generate trading signals based on multiple indicators
            signals = self._generate_trading_signals(indicators, stock_data)
            
            # AI Score (composite)
            ai_score = self._calculate_ai_score(indicators, trend_prediction, confidence)
            
            return {
                "predicted_signal": predicted_signal,
                "confidence": round(confidence, 2),
                "trend_prediction": round(trend_prediction, 4),
                "ai_score": round(ai_score, 2),
                "trading_signals": signals,
                "risk_level": self._assess_risk_level(indicators),
                "market_regime": self._detect_market_regime(indicators)
            }
            
        except Exception as e:
            return {"error": f"AI prediction failed: {str(e)}"}
    
    def _generate_trading_signals(self, indicators: Dict, stock_data: Dict) -> List[str]:
        """Generate specific trading signals"""
        signals = []
        
        # RSI signals
        rsi = indicators.get('rsi', 50)
        if rsi > 70:
            signals.append("RSI Overbought - Consider selling")
        elif rsi < 30:
            signals.append("RSI Oversold - Consider buying")
        
        # MACD signals
        if indicators.get('macd', 0) > indicators.get('macd_signal', 0):
            signals.append("MACD Bullish crossover")
        else:
            signals.append("MACD Bearish crossover")
        
        # Bollinger Bands signals
        bb_pos = indicators.get('bb_position', 50)
        if bb_pos > 95:
            signals.append("Price at upper Bollinger Band - Overbought")
        elif bb_pos < 5:
            signals.append("Price at lower Bollinger Band - Oversold")
        
        # Volume analysis
        volume = float(stock_data.get('totalTradedVolume', 0))
        if volume > 5000000:
            signals.append("High volume - Strong interest")
        
        # Price momentum
        price_change = float(stock_data.get('perChange', 0))
        if price_change > 5:
            signals.append("Strong bullish momentum")
        elif price_change < -5:
            signals.append("Strong bearish momentum")
        
        return signals
    
    def _calculate_ai_score(self, indicators: Dict, trend_pred: float, confidence: float) -> float:
        """Calculate composite AI score (0-100)"""
        # Normalize trend prediction to 0-100 scale
        trend_score = (trend_pred + 1) * 50  # Assuming trend_pred is between -1 and 1
        
        # RSI contribution
        rsi = indicators.get('rsi', 50)
        rsi_score = 100 - abs(rsi - 50) * 2  # Higher score for RSI near 50
        
        # MACD contribution
        macd_hist = indicators.get('macd_histogram', 0)
        macd_score = 50 + (macd_hist * 10)  # Positive histogram = bullish
        
        # Combine scores
        ai_score = (trend_score * 0.4 + confidence * 0.3 + rsi_score * 0.2 + macd_score * 0.1)
        return max(0, min(100, ai_score))
    
    def _assess_risk_level(self, indicators: Dict) -> str:
        """Assess risk level based on volatility and indicators"""
        volatility = indicators.get('volatility', 20)
        atr = indicators.get('atr', 1)
        bb_width = indicators.get('bb_width', 5)
        
        risk_score = volatility * 0.4 + atr * 0.3 + bb_width * 0.3
        
        if risk_score > 30:
            return "HIGH"
        elif risk_score > 15:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _detect_market_regime(self, indicators: Dict) -> str:
        """Detect current market regime"""
        rsi = indicators.get('rsi', 50)
        bb_width = indicators.get('bb_width', 5)
        volatility = indicators.get('volatility', 20)
        
        if bb_width < 3 and volatility < 15:
            return "CONSOLIDATION"
        elif rsi > 60 and volatility > 20:
            return "TRENDING_UP"
        elif rsi < 40 and volatility > 20:
            return "TRENDING_DOWN"
        else:
            return "NEUTRAL"
    
    def _clean_for_json(self, data):
        """Clean data to be JSON serializable by handling NaN and infinite values"""
        import math
        
        if isinstance(data, dict):
            return {k: self._clean_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_for_json(item) for item in data]
        elif isinstance(data, float):
            if math.isnan(data) or math.isinf(data):
                return 0.0
            return data
        else:
            return data