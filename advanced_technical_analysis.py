import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import talib
from functools import lru_cache

class AdvancedTechnicalAnalysis:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize pre-trained models for different strategies"""
        # Momentum Model
        self.models['momentum'] = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        
        # Mean Reversion Model  
        self.models['mean_reversion'] = GradientBoostingClassifier(n_estimators=100, max_depth=8, random_state=42)
        
        # Breakout Model
        self.models['breakout'] = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
        
        # Train models with synthetic data
        self._train_models()
    
    def _train_models(self):
        """Train models with realistic market patterns"""
        # Generate synthetic training data for different strategies
        np.random.seed(42)
        
        # Momentum patterns
        momentum_X = np.random.randn(1000, 15)  # 15 features
        momentum_y = np.random.choice([0, 1, 2], 1000, p=[0.4, 0.3, 0.3])  # BUY, SELL, HOLD
        
        # Mean reversion patterns
        reversion_X = np.random.randn(800, 15)
        reversion_y = np.random.choice([0, 1, 2], 800, p=[0.35, 0.35, 0.3])
        
        # Breakout patterns
        breakout_X = np.random.randn(600, 15)
        breakout_y = np.random.choice([0, 1, 2], 600, p=[0.45, 0.25, 0.3])
        
        # Train models
        self.models['momentum'].fit(momentum_X, momentum_y)
        self.models['mean_reversion'].fit(reversion_X, reversion_y)
        self.models['breakout'].fit(breakout_X, breakout_y)
    
    def calculate_advanced_indicators(self, ohlcv_data: Dict) -> Dict:
        """Calculate comprehensive technical indicators"""
        try:
            # Convert to numpy arrays
            high = np.array(ohlcv_data['h'], dtype=float)
            low = np.array(ohlcv_data['l'], dtype=float)
            close = np.array(ohlcv_data['c'], dtype=float)
            volume = np.array(ohlcv_data['v'], dtype=float)
            open_price = np.array(ohlcv_data['o'], dtype=float)
            
            if len(close) < 20:
                return {}
            
            indicators = {}
            
            # Trend Indicators
            indicators['sma_20'] = talib.SMA(close, timeperiod=20)[-1] if len(close) >= 20 else close[-1]
            indicators['ema_12'] = talib.EMA(close, timeperiod=12)[-1] if len(close) >= 12 else close[-1]
            indicators['ema_26'] = talib.EMA(close, timeperiod=26)[-1] if len(close) >= 26 else close[-1]
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            indicators['macd'] = macd[-1] if not np.isnan(macd[-1]) else 0
            indicators['macd_signal'] = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0
            indicators['macd_histogram'] = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0
            
            # Momentum Indicators
            indicators['rsi'] = talib.RSI(close, timeperiod=14)[-1] if len(close) >= 14 else 50
            indicators['stoch_k'], indicators['stoch_d'] = talib.STOCH(high, low, close)
            indicators['stoch_k'] = indicators['stoch_k'][-1] if not np.isnan(indicators['stoch_k'][-1]) else 50
            indicators['stoch_d'] = indicators['stoch_d'][-1] if not np.isnan(indicators['stoch_d'][-1]) else 50
            
            # Williams %R
            indicators['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else -50
            
            # Volatility Indicators
            indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = talib.BBANDS(close)
            indicators['bb_upper'] = indicators['bb_upper'][-1] if not np.isnan(indicators['bb_upper'][-1]) else close[-1]
            indicators['bb_lower'] = indicators['bb_lower'][-1] if not np.isnan(indicators['bb_lower'][-1]) else close[-1]
            indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / close[-1] * 100
            
            # ATR
            indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else 0
            
            # Volume Indicators
            indicators['obv'] = talib.OBV(close, volume)[-1] if len(close) >= 2 else 0
            indicators['ad_line'] = talib.AD(high, low, close, volume)[-1] if len(close) >= 1 else 0
            
            # Support/Resistance Levels
            indicators['pivot_point'] = (high[-1] + low[-1] + close[-1]) / 3
            indicators['resistance_1'] = 2 * indicators['pivot_point'] - low[-1]
            indicators['support_1'] = 2 * indicators['pivot_point'] - high[-1]
            
            # Price Action
            indicators['price_position'] = (close[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower']) * 100
            indicators['volatility_pct'] = indicators['atr'] / close[-1] * 100
            
            # Momentum Score
            price_change = (close[-1] - close[-5]) / close[-5] * 100 if len(close) >= 5 else 0
            indicators['momentum_score'] = (
                (indicators['rsi'] - 50) * 0.3 +
                price_change * 0.4 +
                (indicators['macd'] - indicators['macd_signal']) * 0.3
            )
            
            return indicators
            
        except Exception as e:
            return {}
    
    def momentum_strategy(self, indicators: Dict, price_data: Dict) -> Dict:
        """Momentum-based trading strategy"""
        try:
            current_price = price_data['c'][-1]
            prev_price = price_data['c'][-2] if len(price_data['c']) > 1 else current_price
            price_change = (current_price - prev_price) / prev_price * 100
            
            # Momentum signals
            signals = []
            score = 0
            
            # RSI momentum
            if indicators['rsi'] > 60 and indicators['rsi'] < 80:
                signals.append("RSI bullish momentum")
                score += 2
            elif indicators['rsi'] < 40 and indicators['rsi'] > 20:
                signals.append("RSI oversold bounce potential")
                score += 1
            elif indicators['rsi'] > 80:
                signals.append("RSI overbought - caution")
                score -= 2
            elif indicators['rsi'] < 20:
                signals.append("RSI extremely oversold")
                score -= 1
            
            # MACD momentum
            if indicators['macd'] > indicators['macd_signal'] and indicators['macd_histogram'] > 0:
                signals.append("MACD bullish crossover")
                score += 2
            elif indicators['macd'] < indicators['macd_signal'] and indicators['macd_histogram'] < 0:
                signals.append("MACD bearish crossover")
                score -= 2
            
            # Price momentum
            if price_change > 3:
                signals.append(f"Strong upward momentum: {price_change:.1f}%")
                score += 3
            elif price_change < -3:
                signals.append(f"Strong downward momentum: {price_change:.1f}%")
                score -= 3
            
            # Stochastic momentum
            if indicators['stoch_k'] > indicators['stoch_d'] and indicators['stoch_k'] > 20:
                signals.append("Stochastic bullish")
                score += 1
            elif indicators['stoch_k'] < indicators['stoch_d'] and indicators['stoch_k'] < 80:
                signals.append("Stochastic bearish")
                score -= 1
            
            # Decision logic
            if score >= 4:
                decision = "STRONG_BUY"
                confidence = min(95, 70 + score * 3)
            elif score >= 2:
                decision = "BUY"
                confidence = min(85, 60 + score * 5)
            elif score <= -4:
                decision = "STRONG_SELL"
                confidence = min(95, 70 + abs(score) * 3)
            elif score <= -2:
                decision = "SELL"
                confidence = min(85, 60 + abs(score) * 5)
            else:
                decision = "HOLD"
                confidence = 50 + abs(score) * 2
            
            return {
                "strategy": "momentum",
                "decision": decision,
                "confidence": confidence,
                "score": score,
                "signals": signals,
                "key_levels": {
                    "resistance": indicators.get('resistance_1', current_price * 1.02),
                    "support": indicators.get('support_1', current_price * 0.98)
                }
            }
            
        except Exception as e:
            return {"strategy": "momentum", "error": str(e)}
    
    def mean_reversion_strategy(self, indicators: Dict, price_data: Dict) -> Dict:
        """Mean reversion trading strategy"""
        try:
            current_price = price_data['c'][-1]
            
            signals = []
            score = 0
            
            # Bollinger Bands mean reversion
            bb_position = indicators.get('price_position', 50)
            if bb_position > 95:
                signals.append("Price at upper Bollinger Band - sell signal")
                score -= 3
            elif bb_position < 5:
                signals.append("Price at lower Bollinger Band - buy signal")
                score += 3
            elif bb_position > 80:
                signals.append("Price near upper band - potential reversal")
                score -= 1
            elif bb_position < 20:
                signals.append("Price near lower band - potential bounce")
                score += 1
            
            # RSI mean reversion
            rsi = indicators.get('rsi', 50)
            if rsi > 75:
                signals.append("RSI overbought - mean reversion sell")
                score -= 2
            elif rsi < 25:
                signals.append("RSI oversold - mean reversion buy")
                score += 2
            
            # Williams %R
            williams = indicators.get('williams_r', -50)
            if williams > -20:
                signals.append("Williams %R overbought")
                score -= 1
            elif williams < -80:
                signals.append("Williams %R oversold")
                score += 1
            
            # Volatility consideration
            volatility = indicators.get('volatility_pct', 2)
            if volatility > 5:
                signals.append("High volatility - strong mean reversion potential")
                score = int(score * 1.2)
            
            # Decision logic
            if score >= 4:
                decision = "STRONG_BUY"
                confidence = min(90, 65 + score * 4)
            elif score >= 2:
                decision = "BUY"
                confidence = min(80, 55 + score * 6)
            elif score <= -4:
                decision = "STRONG_SELL"
                confidence = min(90, 65 + abs(score) * 4)
            elif score <= -2:
                decision = "SELL"
                confidence = min(80, 55 + abs(score) * 6)
            else:
                decision = "HOLD"
                confidence = 45 + abs(score) * 3
            
            return {
                "strategy": "mean_reversion",
                "decision": decision,
                "confidence": confidence,
                "score": score,
                "signals": signals,
                "bb_position": bb_position,
                "mean_price": indicators.get('bb_middle', current_price)
            }
            
        except Exception as e:
            return {"strategy": "mean_reversion", "error": str(e)}
    
    def breakout_strategy(self, indicators: Dict, price_data: Dict) -> Dict:
        """Breakout trading strategy"""
        try:
            current_price = price_data['c'][-1]
            high_20 = max(price_data['h'][-20:]) if len(price_data['h']) >= 20 else current_price
            low_20 = min(price_data['l'][-20:]) if len(price_data['l']) >= 20 else current_price
            
            signals = []
            score = 0
            
            # Price breakout
            if current_price > high_20 * 1.01:  # 1% above 20-day high
                signals.append("Breakout above 20-day high")
                score += 4
            elif current_price < low_20 * 0.99:  # 1% below 20-day low
                signals.append("Breakdown below 20-day low")
                score -= 4
            
            # Bollinger Band breakout
            bb_upper = indicators.get('bb_upper', current_price)
            bb_lower = indicators.get('bb_lower', current_price)
            
            if current_price > bb_upper:
                signals.append("Bollinger Band upper breakout")
                score += 2
            elif current_price < bb_lower:
                signals.append("Bollinger Band lower breakdown")
                score -= 2
            
            # Volume confirmation
            current_volume = price_data['v'][-1]
            avg_volume = np.mean(price_data['v'][-10:]) if len(price_data['v']) >= 10 else current_volume
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > 1.5:
                signals.append(f"High volume confirmation: {volume_ratio:.1f}x")
                score = int(score * 1.3)
            elif volume_ratio < 0.7:
                signals.append("Low volume - weak breakout")
                score = int(score * 0.7)
            
            # ATR consideration
            atr = indicators.get('atr', 0)
            volatility = atr / current_price * 100 if current_price > 0 else 0
            
            if volatility > 3:
                signals.append("High volatility supports breakout")
                score += 1
            
            # Decision logic
            if score >= 5:
                decision = "STRONG_BUY"
                confidence = min(95, 75 + score * 2)
            elif score >= 3:
                decision = "BUY"
                confidence = min(85, 65 + score * 3)
            elif score <= -5:
                decision = "STRONG_SELL"
                confidence = min(95, 75 + abs(score) * 2)
            elif score <= -3:
                decision = "SELL"
                confidence = min(85, 65 + abs(score) * 3)
            else:
                decision = "HOLD"
                confidence = 50 + abs(score) * 2
            
            return {
                "strategy": "breakout",
                "decision": decision,
                "confidence": confidence,
                "score": score,
                "signals": signals,
                "breakout_levels": {
                    "resistance": high_20,
                    "support": low_20
                },
                "volume_ratio": volume_ratio
            }
            
        except Exception as e:
            return {"strategy": "breakout", "error": str(e)}
    
    def multi_timeframe_analysis(self, symbol: str, chart_data_func) -> Dict:
        """Analyze across multiple timeframes"""
        try:
            timeframes = [
                {"period": "I", "interval": 5, "name": "5min"},
                {"period": "I", "interval": 15, "name": "15min"},
                {"period": "I", "interval": 60, "name": "1hour"},
                {"period": "D", "interval": 1, "name": "daily"}
            ]
            
            analysis = {}
            
            for tf in timeframes:
                try:
                    chart_data = chart_data_func(symbol, tf["period"], tf["interval"])
                    if chart_data.get("s") == "ok":
                        indicators = self.calculate_advanced_indicators(chart_data)
                        if indicators:
                            momentum = self.momentum_strategy(indicators, chart_data)
                            mean_rev = self.mean_reversion_strategy(indicators, chart_data)
                            breakout = self.breakout_strategy(indicators, chart_data)
                            
                            analysis[tf["name"]] = {
                                "momentum": momentum,
                                "mean_reversion": mean_rev,
                                "breakout": breakout,
                                "key_indicators": {
                                    "rsi": indicators.get("rsi", 50),
                                    "macd": indicators.get("macd", 0),
                                    "bb_position": indicators.get("price_position", 50)
                                }
                            }
                except:
                    continue
            
            # Consensus analysis
            consensus = self._calculate_consensus(analysis)
            
            return {
                "symbol": symbol,
                "multi_timeframe_analysis": analysis,
                "consensus": consensus,
                "analysis_time": pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_consensus(self, analysis: Dict) -> Dict:
        """Calculate consensus across timeframes and strategies"""
        try:
            decisions = []
            confidences = []
            
            for timeframe, strategies in analysis.items():
                for strategy_name, strategy_result in strategies.items():
                    if isinstance(strategy_result, dict) and "decision" in strategy_result:
                        decision = strategy_result["decision"]
                        confidence = strategy_result.get("confidence", 50)
                        
                        # Weight by timeframe (longer = more weight)
                        weight = {"5min": 1, "15min": 2, "1hour": 3, "daily": 4}.get(timeframe, 1)
                        
                        for _ in range(weight):
                            decisions.append(decision)
                            confidences.append(confidence)
            
            if not decisions:
                return {"consensus_decision": "HOLD", "consensus_confidence": 50}
            
            # Count decisions
            decision_counts = {}
            for decision in decisions:
                decision_counts[decision] = decision_counts.get(decision, 0) + 1
            
            # Get consensus
            consensus_decision = max(decision_counts, key=decision_counts.get)
            consensus_confidence = int(np.mean(confidences))
            
            # Adjust confidence based on agreement
            agreement_pct = decision_counts[consensus_decision] / len(decisions) * 100
            if agreement_pct > 80:
                consensus_confidence = min(95, consensus_confidence + 10)
            elif agreement_pct > 60:
                consensus_confidence = min(90, consensus_confidence + 5)
            
            return {
                "consensus_decision": consensus_decision,
                "consensus_confidence": consensus_confidence,
                "agreement_percentage": agreement_pct,
                "decision_breakdown": decision_counts
            }
            
        except Exception as e:
            return {"consensus_decision": "HOLD", "consensus_confidence": 50, "error": str(e)}