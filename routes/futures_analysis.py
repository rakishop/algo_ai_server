import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any
from utils.ai_risk_calculator import AIRiskCalculator
import warnings
warnings.filterwarnings('ignore')

class FuturesAnalyzer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.risk_calculator = AIRiskCalculator()
    
    def extract_futures_data(self, oi_spurts_data: Dict) -> pd.DataFrame:
        """Extract only futures contracts from OI spurts data"""
        futures_data = []
        
        if not oi_spurts_data or "data" not in oi_spurts_data:
            return pd.DataFrame()
        
        for category in oi_spurts_data["data"]:
            for category_name, contracts in category.items():
                for contract in contracts:
                    if contract.get("instrumentType") in ["FUTSTK", "FUTIDX"]:
                        futures_data.append({
                            "symbol": contract.get("symbol"),
                            "instrument": contract.get("instrument"),
                            "instrumentType": contract.get("instrumentType"),
                            "expiryDate": contract.get("expiryDate"),
                            "ltp": contract.get("ltp", 0),
                            "prevClose": contract.get("prevClose", 0),
                            "pChange": contract.get("pChange", 0),
                            "latestOI": contract.get("latestOI", 0),
                            "prevOI": contract.get("prevOI", 0),
                            "changeInOI": contract.get("changeInOI", 0),
                            "pChangeInOI": contract.get("pChangeInOI", 0),
                            "volume": contract.get("volume", 0),
                            "turnover": contract.get("turnover", 0),
                            "underlyingValue": contract.get("underlyingValue", 0),
                            "category": category_name,
                            "type": contract.get("type", "")
                        })
        
        return pd.DataFrame(futures_data)
    
    def analyze_futures(self, df: pd.DataFrame) -> Dict:
        """Analyze futures data with pandas"""
        if df.empty:
            return {"error": "No futures data available"}
        
        # Basic statistics
        analysis = {
            "total_futures": len(df),
            "stock_futures": len(df[df["instrumentType"] == "FUTSTK"]),
            "index_futures": len(df[df["instrumentType"] == "FUTIDX"]),
            "categories": df["category"].value_counts().to_dict()
        }
        
        # Price movement analysis
        df["abs_pChange"] = df["pChange"].abs()
        analysis["price_stats"] = {
            "avg_price_change": df["pChange"].mean(),
            "max_gainer": df.loc[df["pChange"].idxmax()]["symbol"] if not df.empty else None,
            "max_loser": df.loc[df["pChange"].idxmin()]["symbol"] if not df.empty else None,
            "high_volatility": df[df["abs_pChange"] > 2]["symbol"].tolist()
        }
        
        # OI analysis
        analysis["oi_stats"] = {
            "avg_oi_change": df["pChangeInOI"].mean(),
            "oi_gainers": df[df["pChangeInOI"] > 20]["symbol"].tolist(),
            "oi_losers": df[df["pChangeInOI"] < -20]["symbol"].tolist()
        }
        
        # Volume analysis
        df["volume_score"] = pd.qcut(df["volume"], q=4, labels=["Low", "Medium", "High", "Very High"])
        analysis["volume_distribution"] = df["volume_score"].value_counts().to_dict()
        
        return analysis
    
    def calculate_stop_loss_target(self, symbol: str, current_price: float, action: str) -> Dict:
        """Calculate AI-based stop loss and target using historical data"""
        
        # Use AI risk calculator for futures with specific futures method
        ai_levels = self.risk_calculator.calculate_futures_levels(symbol, current_price, action)
        
        return ai_levels
    
    def generate_recommendations(self, df: pd.DataFrame) -> List[Dict]:
        """Generate buy/sell recommendations with stop loss and targets"""
        if df.empty:
            return []
        
        recommendations = []
        
        for _, row in df.iterrows():
            score = 0
            signals = []
            
            # Price momentum (30%)
            if row["pChange"] > 2:
                score += 30
                signals.append("Strong Bullish")
            elif row["pChange"] > 0.5:
                score += 15
                signals.append("Bullish")
            elif row["pChange"] < -2:
                score += 25  # Short opportunity
                signals.append("Strong Bearish")
            elif row["pChange"] < -0.5:
                score += 10
                signals.append("Bearish")
            
            # OI analysis (25%)
            if row["pChangeInOI"] > 30:
                score += 25
                signals.append("High OI Build-up")
            elif row["pChangeInOI"] > 10:
                score += 15
                signals.append("OI Build-up")
            elif row["pChangeInOI"] < -30:
                score += 20
                signals.append("OI Unwinding")
            
            # Volume analysis (25%)
            if row["volume"] > 100000:
                score += 25
                signals.append("High Volume")
            elif row["volume"] > 50000:
                score += 15
                signals.append("Good Volume")
            
            # Category analysis (20%)
            if row["category"] == "Rise-in-OI-Rise":
                score += 20
                signals.append("Bullish Setup")
                action = "BUY"
            elif row["category"] == "Slide-in-OI-Slide":
                score += 20
                signals.append("Bearish Setup")
                action = "SELL"
            elif row["category"] == "Rise-in-OI-Slide":
                score += 15
                signals.append("Short Build-up")
                action = "SELL"
            else:
                score += 10
                action = "BUY" if row["pChange"] > 0 else "SELL"
            
            # Risk assessment
            risk = "LOW" if score >= 70 else "MEDIUM" if score >= 50 else "HIGH"
            
            # Calculate AI-based stop loss and target
            sl_target = self.calculate_stop_loss_target(
                row["symbol"], row["ltp"], action
            )
            
            recommendations.append({
                "symbol": row["symbol"],
                "action": action,
                "current_price": row["ltp"],
                "stop_loss": sl_target["stop_loss"],
                "target": sl_target["target"],
                "sl_percentage": sl_target["sl_percentage"],
                "target_percentage": sl_target["target_percentage"],
                "risk_reward_ratio": sl_target["risk_reward_ratio"],
                "price_change_pct": row["pChange"],
                "oi_change_pct": row["pChangeInOI"],
                "volume": row["volume"],
                "ai_score": score,
                "risk_level": risk,
                "signals": signals,
                "expiry": row["expiryDate"],
                "category": row["category"],
                "trading_plan": {
                    "entry": f"{action} at ₹{row['ltp']}",
                    "stop_loss": f"₹{sl_target['stop_loss']} ({sl_target['sl_percentage']}%)",
                    "target": f"₹{sl_target['target']} ({sl_target['target_percentage']}%)",
                    "risk_reward": f"1:{sl_target['risk_reward_ratio']}"
                }
            })
        
        return sorted(recommendations, key=lambda x: x["ai_score"], reverse=True)
    
    def train_ml_model(self, df: pd.DataFrame):
        """Train ML model for future predictions"""
        if df.empty or len(df) < 10:
            return {"error": "Insufficient data for training"}
        
        # Prepare features
        features = ["pChange", "pChangeInOI", "volume", "latestOI"]
        X = df[features].fillna(0)
        
        # Create target (1 for buy, 0 for sell)
        y = ((df["pChange"] > 0) & (df["pChangeInOI"] > 0)).astype(int)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        return {
            "status": "Model trained successfully",
            "samples": len(df),
            "features": features,
            "accuracy": self.model.score(X_scaled, y)
        }
    
    def predict_recommendations(self, df: pd.DataFrame) -> List[Dict]:
        """Get ML-based predictions with stop loss and targets"""
        if not self.is_trained or df.empty:
            return []
        
        features = ["pChange", "pChangeInOI", "volume", "latestOI"]
        X = df[features].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        ml_recommendations = []
        for i, (_, row) in enumerate(df.iterrows()):
            confidence = max(probabilities[i]) * 100
            action = "BUY" if predictions[i] == 1 else "SELL"
            
            # Calculate AI-based stop loss and target for ML predictions
            sl_target = self.calculate_stop_loss_target(
                row["symbol"], row["ltp"], action
            )
            
            # Determine risk based on confidence
            risk = "LOW" if confidence > 75 else "MEDIUM" if confidence > 60 else "HIGH"
            
            ml_recommendations.append({
                "symbol": row["symbol"],
                "ml_action": action,
                "confidence": round(confidence, 2),
                "current_price": row["ltp"],
                "stop_loss": sl_target["stop_loss"],
                "target": sl_target["target"],
                "risk_reward_ratio": sl_target["risk_reward_ratio"],
                "predicted_direction": "UP" if predictions[i] == 1 else "DOWN",
                "risk_level": risk,
                "ml_trading_plan": {
                    "entry": f"{action} at ₹{row['ltp']}",
                    "stop_loss": f"₹{sl_target['stop_loss']} ({sl_target['sl_percentage']}%)",
                    "target": f"₹{sl_target['target']} ({sl_target['target_percentage']}%)",
                    "confidence": f"{confidence:.1f}%"
                }
            })
        
        return sorted(ml_recommendations, key=lambda x: x["confidence"], reverse=True)

def create_futures_analysis_routes(nse_client):
    from fastapi import APIRouter, Query
    from typing import Optional
    
    router = APIRouter(prefix="/api/v1/futures", tags=["futures"])
    analyzer = FuturesAnalyzer()
    
    @router.get("/analysis")
    def get_futures_analysis():
        """Analyze futures contracts from OI spurts data"""
        try:
            oi_data = nse_client.get_oi_spurts_contracts()
            df = analyzer.extract_futures_data(oi_data)
            
            if df.empty:
                return {"error": "No futures data found"}
            
            analysis = analyzer.analyze_futures(df)
            recommendations = analyzer.generate_recommendations(df)
            
            # Train ML model
            training_result = analyzer.train_ml_model(df)
            ml_predictions = analyzer.predict_recommendations(df)
            
            return {
                "futures_analysis": analysis,
                "top_recommendations": recommendations[:10],
                "ml_predictions": ml_predictions[:10],
                "training_info": training_result,
                "total_contracts": len(df)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    @router.get("/stock-futures")
    def get_stock_futures_only():
        """Get analysis for stock futures only"""
        try:
            oi_data = nse_client.get_oi_spurts_contracts()
            df = analyzer.extract_futures_data(oi_data)
            
            # Filter only stock futures
            stock_futures = df[df["instrumentType"] == "FUTSTK"]
            
            if stock_futures.empty:
                return {"error": "No stock futures found"}
            
            analysis = analyzer.analyze_futures(stock_futures)
            recommendations = analyzer.generate_recommendations(stock_futures)
            
            return {
                "stock_futures_analysis": analysis,
                "recommendations": recommendations,
                "count": len(stock_futures)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    @router.get("/index-futures")
    def get_index_futures_only():
        """Get analysis for index futures only"""
        try:
            oi_data = nse_client.get_oi_spurts_contracts()
            df = analyzer.extract_futures_data(oi_data)
            
            # Filter only index futures
            index_futures = df[df["instrumentType"] == "FUTIDX"]
            
            if index_futures.empty:
                return {"error": "No index futures found"}
            
            analysis = analyzer.analyze_futures(index_futures)
            recommendations = analyzer.generate_recommendations(index_futures)
            
            return {
                "index_futures_analysis": analysis,
                "recommendations": recommendations,
                "count": len(index_futures)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    @router.get("/position-sizing")
    def calculate_position_sizing(
        symbol: str,
        account_size: float = 100000,
        risk_per_trade: float = 2.0
    ):
        """Calculate position size based on account size and risk"""
        try:
            oi_data = nse_client.get_oi_spurts_contracts()
            df = analyzer.extract_futures_data(oi_data)
            
            # Find the symbol
            symbol_data = df[df["symbol"] == symbol.upper()]
            if symbol_data.empty:
                return {"error": f"Symbol {symbol} not found"}
            
            row = symbol_data.iloc[0]
            recommendations = analyzer.generate_recommendations(symbol_data)
            
            if not recommendations:
                return {"error": "No recommendations available"}
            
            rec = recommendations[0]
            
            # Calculate position size
            risk_amount = account_size * (risk_per_trade / 100)
            price_diff = abs(rec["current_price"] - rec["stop_loss"])
            
            if price_diff == 0:
                return {"error": "Invalid stop loss calculation"}
            
            # For futures, lot size matters (assuming 1 lot = 1 unit for simplicity)
            position_size = int(risk_amount / price_diff)
            max_loss = position_size * price_diff
            potential_profit = position_size * abs(rec["target"] - rec["current_price"])
            
            return {
                "symbol": symbol.upper(),
                "account_size": account_size,
                "risk_per_trade_pct": risk_per_trade,
                "risk_amount": risk_amount,
                "position_sizing": {
                    "recommended_quantity": position_size,
                    "entry_price": rec["current_price"],
                    "stop_loss": rec["stop_loss"],
                    "target": rec["target"],
                    "max_loss": round(max_loss, 2),
                    "potential_profit": round(potential_profit, 2),
                    "risk_reward_ratio": rec["risk_reward_ratio"]
                },
                "trading_plan": rec["trading_plan"],
                "action": rec["action"],
                "ai_score": rec["ai_score"],
                "risk_level": rec["risk_level"]
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    return router