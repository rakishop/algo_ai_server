import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class OptionsAIAnalyzer:
    def __init__(self):
        self.sentiment_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.volatility_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self._train_models()
    
    def _train_models(self):
        """Train AI models with synthetic market data patterns"""
        # Generate synthetic training data based on market patterns
        np.random.seed(42)
        n_samples = 1000
        
        # Features: PCR, IV, Volume Ratio, Price Change, OI Change
        X = np.random.rand(n_samples, 8)
        X[:, 0] = np.random.normal(1.0, 0.3, n_samples)  # PCR
        X[:, 1] = np.random.normal(20, 10, n_samples)    # IV
        X[:, 2] = np.random.exponential(2, n_samples)    # Volume ratio
        X[:, 3] = np.random.normal(0, 15, n_samples)     # Price change
        X[:, 4] = np.random.normal(0, 20, n_samples)     # OI change
        X[:, 5] = np.random.normal(0.5, 0.2, n_samples) # Moneyness
        X[:, 6] = np.random.normal(30, 15, n_samples)    # Days to expiry
        X[:, 7] = np.random.normal(0, 5, n_samples)      # Skew
        
        # Labels for sentiment (0=Bearish, 1=Neutral, 2=Bullish)
        y_sentiment = []
        y_volatility = []
        
        for i in range(n_samples):
            pcr, iv, vol_ratio, price_chg, oi_chg = X[i, :5]
            
            # Sentiment logic
            if pcr > 1.2 and price_chg < -5:
                sentiment = 0  # Bearish
            elif pcr < 0.8 and price_chg > 5:
                sentiment = 2  # Bullish
            else:
                sentiment = 1  # Neutral
            
            # Volatility prediction
            volatility = abs(price_chg) + iv/10 + vol_ratio*2
            
            y_sentiment.append(sentiment)
            y_volatility.append(volatility)
        
        X_scaled = self.scaler.fit_transform(X)
        self.sentiment_model.fit(X_scaled, y_sentiment)
        self.volatility_model.fit(X_scaled, y_volatility)
        self.is_trained = True
    
    def extract_advanced_features(self, options_data: List[Dict]) -> np.ndarray:
        """Extract advanced features for AI analysis"""
        calls = [opt for opt in options_data if opt.get('optionType') == 'Call']
        puts = [opt for opt in options_data if opt.get('optionType') == 'Put']
        
        if not calls or not puts:
            return np.zeros(8)
        
        # Calculate advanced metrics
        call_volume = sum(opt.get('numberOfContractsTraded', 0) for opt in calls)
        put_volume = sum(opt.get('numberOfContractsTraded', 0) for opt in puts)
        pcr = put_volume / call_volume if call_volume > 0 else 1
        
        # Implied volatility estimation (using price changes as proxy)
        call_changes = [opt.get('pChange', 0) for opt in calls]
        put_changes = [opt.get('pChange', 0) for opt in puts]
        iv_estimate = np.std(call_changes + put_changes) if call_changes + put_changes else 20
        
        # Volume ratio
        total_volume = call_volume + put_volume
        volume_ratio = total_volume / 1000000 if total_volume > 0 else 0
        
        # Average price changes
        avg_price_change = np.mean(call_changes + put_changes) if call_changes + put_changes else 0
        
        # OI change estimation
        total_oi = sum(opt.get('openInterest', 0) for opt in options_data)
        oi_change = np.random.normal(0, 10)  # Simulated as real OI change not available
        
        # Moneyness (ATM ratio)
        strikes = [opt.get('strikePrice', 0) for opt in options_data]
        if strikes:
            spot_estimate = np.median(strikes)  # Rough spot estimation
            atm_strikes = [s for s in strikes if abs(s - spot_estimate) < 100]
            moneyness = len(atm_strikes) / len(strikes)
        else:
            moneyness = 0.5
        
        # Days to expiry (estimated)
        days_to_expiry = 30  # Default assumption
        
        # Volatility skew
        if len(call_changes) > 1 and len(put_changes) > 1:
            skew = np.mean(put_changes) - np.mean(call_changes)
        else:
            skew = 0
        
        return np.array([pcr, iv_estimate, volume_ratio, avg_price_change, 
                        oi_change, moneyness, days_to_expiry, skew])
    
    def analyze_market_regime(self, features: np.ndarray) -> Dict[str, Any]:
        """Analyze current market regime using clustering"""
        # Define market regimes based on feature patterns
        if features[0] > 1.3:  # High PCR
            regime = "High Fear"
        elif features[0] < 0.7:  # Low PCR
            regime = "Extreme Greed"
        elif abs(features[3]) > 20:  # High price change
            regime = "High Volatility"
        elif features[1] > 30:  # High IV
            regime = "Uncertainty"
        else:
            regime = "Normal"
        
        return {
            "regime": regime,
            "fear_greed_index": min(max((1 - features[0]) * 100, 0), 100),
            "volatility_percentile": min(features[1] * 2, 100),
            "liquidity_score": min(features[2] * 20, 100)
        }
    
    def find_optimal_strikes(self, options_data: List[Dict], strategy_type: str) -> Dict[str, Any]:
        """Find optimal strikes for given strategy"""
        strikes = sorted(set(opt.get('strikePrice', 0) for opt in options_data))
        if not strikes:
            return {}
        
        # Estimate current spot price
        spot_estimate = np.median(strikes)
        
        if strategy_type == "straddle":
            # Find ATM strike with highest volume
            atm_options = [opt for opt in options_data if abs(opt.get('strikePrice', 0) - spot_estimate) < 100]
            if atm_options:
                best_strike = max(atm_options, key=lambda x: x.get('numberOfContractsTraded', 0)).get('strikePrice')
                return {
                    "recommended_strike": best_strike,
                    "spot_estimate": spot_estimate,
                    "strike_selection_reason": f"ATM strike {best_strike} selected for maximum gamma exposure"
                }
        
        elif strategy_type == "bear_spread":
            # ITM put + OTM put
            itm_strike = next((s for s in reversed(strikes) if s > spot_estimate), strikes[-1])
            otm_strike = next((s for s in strikes if s < spot_estimate - 100), strikes[0])
            return {
                "buy_strike": itm_strike,
                "sell_strike": otm_strike,
                "spot_estimate": spot_estimate,
                "strike_selection_reason": f"Buy {itm_strike} put, sell {otm_strike} put"
            }
        
        elif strategy_type == "bull_spread":
            # ITM call + OTM call
            itm_strike = next((s for s in strikes if s < spot_estimate), strikes[0])
            otm_strike = next((s for s in strikes if s > spot_estimate + 100), strikes[-1])
            return {
                "buy_strike": itm_strike,
                "sell_strike": otm_strike,
                "spot_estimate": spot_estimate,
                "strike_selection_reason": f"Buy {itm_strike} call, sell {otm_strike} call"
            }
        
        return {"recommended_strike": spot_estimate}
    
    def predict_optimal_strategies(self, options_data: List[Dict]) -> List[Dict[str, Any]]:
        """Use AI models to predict optimal options strategies"""
        if not self.is_trained:
            return []
        
        features = self.extract_advanced_features(options_data)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # AI predictions
        sentiment_pred = self.sentiment_model.predict(features_scaled)[0]
        sentiment_proba = self.sentiment_model.predict_proba(features_scaled)[0]
        volatility_pred = self.volatility_model.predict(features_scaled)[0]
        
        # Market regime analysis
        regime_analysis = self.analyze_market_regime(features)
        
        strategies = []
        
        # Strategy selection based on AI predictions
        if sentiment_pred == 0:  # Bearish
            confidence = sentiment_proba[0] * 100
            strike_info = self.find_optimal_strikes(options_data, "bear_spread")
            strategies.append({
                "name": "AI Bear Put Spread",
                "type": "Bearish",
                "confidence_score": confidence,
                "risk_reward": 1.6,
                "description": f"AI model predicts bearish sentiment with {confidence:.1f}% confidence",
                "strike_recommendation": strike_info,
                "ai_insights": {
                    "predicted_volatility": volatility_pred,
                    "market_regime": regime_analysis["regime"],
                    "fear_greed_index": regime_analysis["fear_greed_index"]
                }
            })
        elif sentiment_pred == 2:  # Bullish
            confidence = sentiment_proba[2] * 100
            strike_info = self.find_optimal_strikes(options_data, "bull_spread")
            strategies.append({
                "name": "AI Bull Call Spread",
                "type": "Bullish",
                "confidence_score": confidence,
                "risk_reward": 1.8,
                "description": f"AI model predicts bullish sentiment with {confidence:.1f}% confidence",
                "strike_recommendation": strike_info,
                "ai_insights": {
                    "predicted_volatility": volatility_pred,
                    "market_regime": regime_analysis["regime"],
                    "fear_greed_index": regime_analysis["fear_greed_index"]
                }
            })
        
        # Volatility-based strategy
        if volatility_pred > 15:
            reasoning = []
            if abs(features[3]) > 15: reasoning.append(f"High price uncertainty ({features[3]:.1f}%)")
            if features[7] > 50: reasoning.append(f"Extreme volatility skew ({features[7]:.1f})")
            if features[0] > 1.1: reasoning.append(f"Elevated PCR ({features[0]:.2f}) shows fear")
            
            strike_info = self.find_optimal_strikes(options_data, "straddle")
            strategies.append({
                "name": "AI Long Straddle",
                "type": "Volatility",
                "confidence_score": min(volatility_pred * 3, 95),
                "risk_reward": 2.2,
                "description": f"AI predicts high volatility ({volatility_pred:.1f}). Long straddle profits from big moves in either direction.",
                "reasoning": reasoning,
                "strategy_explanation": "Buy Call + Put at same strike. Profits if price moves significantly up OR down. Ideal for volatile but directionless markets.",
                "strike_recommendation": strike_info,
                "ai_insights": {
                    "predicted_volatility": volatility_pred,
                    "market_regime": regime_analysis["regime"],
                    "volatility_percentile": regime_analysis["volatility_percentile"]
                }
            })
        elif volatility_pred < 8:
            strategies.append({
                "name": "AI Iron Condor",
                "type": "Low Volatility",
                "confidence_score": min((15 - volatility_pred) * 6, 85),
                "risk_reward": 1.9,
                "description": f"AI predicts low volatility ({volatility_pred:.1f}). Iron condor recommended.",
                "ai_insights": {
                    "predicted_volatility": volatility_pred,
                    "market_regime": regime_analysis["regime"],
                    "volatility_percentile": regime_analysis["volatility_percentile"]
                }
            })
        
        return strategies[:2]  # Return top 2 strategies