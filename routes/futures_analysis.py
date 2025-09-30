import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any
from utils.ai_risk_calculator import AIRiskCalculator
import warnings
import os
import json
from datetime import datetime
warnings.filterwarnings('ignore')

class FuturesAnalyzer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.risk_calculator = AIRiskCalculator()
        self.training_data_dir = "futures_training_data"
        os.makedirs(self.training_data_dir, exist_ok=True)
        # Remove nse_client dependency - use AI risk calculator directly
    
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
            
            # Pattern analysis enhancement
            pattern_analysis = self.analyze_futures_pattern(row["symbol"], row.to_dict())
            if "pattern_analysis" in pattern_analysis:
                pattern_score = pattern_analysis["pattern_analysis"]["pattern_score"]
                if pattern_score > 50:
                    score += 20
                    signals.extend(pattern_analysis["signals"])
                elif pattern_score > 30:
                    score += 10
            
            # Risk assessment
            risk = "LOW" if score >= 70 else "MEDIUM" if score >= 50 else "HIGH"
            
            # Calculate AI-based stop loss and target
            sl_target = self.calculate_stop_loss_target(
                row["symbol"], row["ltp"], action
            )
            
            # Handle error response from AI calculator
            if "error" in sl_target:
                sl_target = {
                    "stop_loss": row["ltp"] * 0.97,  # 3% default
                    "target": row["ltp"] * 1.06,     # 6% default
                    "sl_percentage": 3.0,
                    "target_percentage": 6.0,
                    "risk_reward_ratio": 2.0,
                    "atr": 0,
                    "volatility": 0,
                    "support": 0,
                    "resistance": 0
                }
            
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
                "pattern_analysis": pattern_analysis,
                "trading_plan": {
                    "entry": f"{action} at ₹{row['ltp']}",
                    "stop_loss": f"₹{sl_target['stop_loss']} ({sl_target['sl_percentage']}%)",
                    "target": f"₹{sl_target['target']} ({sl_target['target_percentage']}%)",
                    "risk_reward": f"1:{sl_target['risk_reward_ratio']}"
                }
            })
        
        return sorted(recommendations, key=lambda x: x["ai_score"], reverse=True)
    
    def analyze_futures_pattern(self, symbol: str, current_data: Dict) -> Dict:
        """Analyze futures pattern using current OI spurts data"""
        try:
            # Extract key metrics from current futures data
            current_price = current_data.get("ltp", 0)
            price_change = current_data.get("pChange", 0)
            oi_change = current_data.get("pChangeInOI", 0)
            volume = current_data.get("volume", 0)
            category = current_data.get("category", "")
            prev_close = current_data.get("prevClose", current_price)
            
            # Calculate pattern strength
            pattern_score = 0
            signals = []
            
            # OI and Price Pattern Analysis
            if category == "Rise-in-OI-Rise":
                pattern_score += 25
                signals.append("Bullish: Fresh Long Build-up")
                recommendation = "STRONG BUY"
            elif category == "Slide-in-OI-Slide":
                pattern_score += 25
                signals.append("Bearish: Fresh Short Build-up")
                recommendation = "STRONG SELL"
            elif category == "Rise-in-OI-Slide":
                pattern_score += 20
                signals.append("Bearish: Short Build-up")
                recommendation = "SELL"
            elif category == "Slide-in-OI-Rise":
                pattern_score += 15
                signals.append("Bullish: Short Covering")
                recommendation = "BUY"
            else:
                pattern_score += 5
                recommendation = "HOLD"
            
            # Volume Analysis
            if volume > 100000:
                pattern_score += 20
                signals.append("High Volume Confirmation")
            elif volume > 50000:
                pattern_score += 10
                signals.append("Good Volume")
            
            # Price Momentum
            if abs(price_change) > 3:
                pattern_score += 15
                signals.append(f"Strong Momentum: {price_change:.1f}%")
            elif abs(price_change) > 1:
                pattern_score += 8
                signals.append(f"Good Momentum: {price_change:.1f}%")
            
            # OI Change Analysis
            if abs(oi_change) > 30:
                pattern_score += 15
                signals.append(f"High OI Activity: {oi_change:.1f}%")
            elif abs(oi_change) > 15:
                pattern_score += 8
                signals.append(f"Moderate OI Activity: {oi_change:.1f}%")
            
            # Risk Assessment
            risk_level = "LOW" if pattern_score >= 60 else "MEDIUM" if pattern_score >= 40 else "HIGH"
            confidence = min(95, pattern_score + 20)
            
            return {
                "pattern_analysis": {
                    "category": category,
                    "pattern_score": pattern_score,
                    "confidence": confidence,
                    "recommendation": recommendation,
                    "risk_level": risk_level
                },
                "signals": signals,
                "metrics": {
                    "price_change": price_change,
                    "oi_change": oi_change,
                    "volume": volume,
                    "current_price": current_price
                }
            }
            
        except Exception as e:
            return {"error": f"Pattern analysis error: {str(e)}"}
    
    def get_best_futures_bets(self, df: pd.DataFrame) -> List[Dict]:
        """Get best futures bets based on OI spurts analysis - one per symbol"""
        if df.empty:
            return []
        
        # Group by symbol and get best contract per symbol
        symbol_best = {}
        
        for _, row in df.iterrows():
            pattern_analysis = self.analyze_futures_pattern(row["symbol"], row.to_dict())
            
            if "error" in pattern_analysis:
                continue
            
            pattern = pattern_analysis["pattern_analysis"]
            
            # Only include high confidence trades
            if pattern["confidence"] >= 60:
                symbol = row["symbol"]
                
                # Calculate comprehensive score for this contract
                volume_score = min(row["volume"] / 10000, 10)  # Volume factor (max 10)
                oi_activity_score = min(abs(row["pChangeInOI"]) / 10, 5)  # OI activity factor (max 5)
                momentum_score = min(abs(row["pChange"]) * 2, 10)  # Price momentum factor (max 10)
                
                # Days to expiry score (closer expiry gets higher score for short-term trades)
                expiry_date = row["expiryDate"]
                if "Sep" in expiry_date:
                    expiry_score = 3  # Near expiry - higher urgency
                else:
                    expiry_score = 2  # Far expiry - more time
                
                # Comprehensive score combining all factors
                total_score = (
                    pattern["confidence"] * 0.4 +  # 40% confidence
                    volume_score * 0.2 +           # 20% volume
                    oi_activity_score * 0.2 +      # 20% OI activity
                    momentum_score * 0.1 +         # 10% momentum
                    expiry_score * 0.1              # 10% expiry timing
                )
                
                # If symbol not seen or this contract has higher total score
                if symbol not in symbol_best or total_score > symbol_best[symbol].get("total_score", 0):
                    # Calculate AI-based levels
                    sl_target = self.calculate_stop_loss_target(
                        row["symbol"], row["ltp"], pattern["recommendation"]
                    )
                    
                    if "error" in sl_target:
                        sl_target = {
                            "stop_loss": row["ltp"] * 0.97,
                            "target": row["ltp"] * 1.06,
                            "sl_percentage": 3.0,
                            "target_percentage": 6.0,
                            "risk_reward_ratio": 2.0
                        }
                    
                    symbol_best[symbol] = {
                        "symbol": row["symbol"],
                        "recommendation": pattern["recommendation"],
                        "confidence": pattern["confidence"],
                        "pattern_score": pattern["pattern_score"],
                        "risk_level": pattern["risk_level"],
                        "current_price": row["ltp"],
                        "price_change": row["pChange"],
                        "oi_change": row["pChangeInOI"],
                        "volume": row["volume"],
                        "category": row["category"],
                        "expiry": row["expiryDate"],
                        "stop_loss": sl_target["stop_loss"],
                        "target": sl_target["target"],
                        "sl_percentage": sl_target["sl_percentage"],
                        "target_percentage": sl_target["target_percentage"],
                        "risk_reward_ratio": sl_target["risk_reward_ratio"],
                        "signals": pattern_analysis["signals"],
                        "total_score": total_score,
                        "selection_reason": f"Best among expiries (Score: {total_score:.1f})",
                        "trading_plan": {
                            "action": pattern["recommendation"],
                            "entry": f"₹{row['ltp']}",
                            "stop_loss": f"₹{sl_target['stop_loss']} ({sl_target['sl_percentage']}%)",
                            "target": f"₹{sl_target['target']} ({sl_target['target_percentage']}%)",
                            "risk_reward": f"1:{sl_target['risk_reward_ratio']}",
                            "rationale": f"{row['category']} with {pattern['confidence']}% confidence - {row['expiryDate']} expiry"
                        }
                    }
        
        # Convert to list and sort by total score
        best_bets = list(symbol_best.values())
        return sorted(best_bets, key=lambda x: x["total_score"], reverse=True)
    
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
            
            # Handle error response from AI calculator
            if "error" in sl_target:
                sl_target = {
                    "stop_loss": row["ltp"] * 0.97,  # 3% default
                    "target": row["ltp"] * 1.06,     # 6% default
                    "sl_percentage": 3.0,
                    "target_percentage": 6.0,
                    "risk_reward_ratio": 2.0,
                    "atr": 0,
                    "volatility": 0,
                    "support": 0,
                    "resistance": 0
                }
            
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
    # No need to set nse_client since we use AI risk calculator for historical data
    
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
            best_bets = analyzer.get_best_futures_bets(df)
            
            # Save futures data for training
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"futures_data_{timestamp}.json"
            filepath = os.path.join(analyzer.training_data_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(df.to_dict('records'), f)
            
            # Train ML model
            training_result = analyzer.train_ml_model(df)
            ml_predictions = analyzer.predict_recommendations(df)
            
            # Count total contracts in raw data
            total_raw_contracts = 0
            futures_count = 0
            options_count = 0
            
            for category in oi_data.get("data", []):
                for category_name, contracts in category.items():
                    for contract in contracts:
                        total_raw_contracts += 1
                        if contract.get("instrumentType") in ["FUTSTK", "FUTIDX"]:
                            futures_count += 1
                        else:
                            options_count += 1
            
            return {
                "futures_analysis": analysis,
                "top_recommendations": recommendations[:10],
                "best_bets": best_bets[:10],
                "ml_predictions": ml_predictions[:10],
                "training_info": training_result,
                "total_contracts": len(df),
                "data_breakdown": {
                    "total_raw_contracts": total_raw_contracts,
                    "futures_contracts": futures_count,
                    "options_contracts": options_count,
                    "futures_analyzed": len(df)
                },
                "best_bets_summary": {
                    "strong_buy": len([bet for bet in best_bets if bet["recommendation"] == "STRONG BUY"]),
                    "strong_sell": len([bet for bet in best_bets if bet["recommendation"] == "STRONG SELL"]),
                    "high_confidence_count": len([bet for bet in best_bets if bet["confidence"] >= 80])
                }
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
            best_bets = analyzer.get_best_futures_bets(stock_futures)
            
            # Train ML model on stock futures
            training_result = analyzer.train_ml_model(stock_futures)
            ml_predictions = analyzer.predict_recommendations(stock_futures)
            
            return {
                "stock_futures_analysis": analysis,
                "recommendations": recommendations[:10],
                "best_bets": best_bets[:10],
                "ml_predictions": ml_predictions[:10],
                "training_info": training_result,
                "count": len(stock_futures),
                "best_bets_summary": {
                    "strong_buy": len([bet for bet in best_bets if bet["recommendation"] == "STRONG BUY"]),
                    "strong_sell": len([bet for bet in best_bets if bet["recommendation"] == "STRONG SELL"]),
                    "high_confidence_count": len([bet for bet in best_bets if bet["confidence"] >= 80])
                }
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
            best_bets = analyzer.get_best_futures_bets(index_futures)
            
            # Train ML model on index futures
            training_result = analyzer.train_ml_model(index_futures)
            ml_predictions = analyzer.predict_recommendations(index_futures)
            
            return {
                "index_futures_analysis": analysis,
                "recommendations": recommendations,
                "best_bets": best_bets,
                "ml_predictions": ml_predictions,
                "training_info": training_result,
                "count": len(index_futures),
                "best_bets_summary": {
                    "strong_buy": len([bet for bet in best_bets if bet["recommendation"] == "STRONG BUY"]),
                    "strong_sell": len([bet for bet in best_bets if bet["recommendation"] == "STRONG SELL"]),
                    "high_confidence_count": len([bet for bet in best_bets if bet["confidence"] >= 80])
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    

    
    @router.get("/position-sizing")
    def calculate_position_sizing(
        symbol: str,
        account_size: float = 100000,
        risk_per_trade: float = 2.0
    ):
        """Calculate position size using comprehensive NSE symbol list"""
        try:
            # Get comprehensive underlying information from NSE
            underlying_info = nse_client.get_underlying_information()
            
            if "error" in underlying_info:
                return {"error": "Failed to fetch underlying information"}
            
            # Extract all available symbols
            all_symbols = []
            if "data" in underlying_info:
                if "IndexList" in underlying_info["data"]:
                    for index in underlying_info["data"]["IndexList"]:
                        all_symbols.append(index["symbol"])
                if "UnderlyingList" in underlying_info["data"]:
                    for underlying in underlying_info["data"]["UnderlyingList"]:
                        all_symbols.append(underlying["symbol"])
            
            symbol_upper = symbol.upper()
            if symbol_upper not in all_symbols:
                return {"error": f"Symbol {symbol} not available for futures trading"}
            
            # Try OI spurts data first
            oi_data = nse_client.get_oi_spurts_contracts()
            df = analyzer.extract_futures_data(oi_data)
            symbol_data = df[df["symbol"] == symbol_upper]
            
            if symbol_data.empty:
                # Use AI calculator for symbols not in current OI data
                current_price = 100
                sl_target = analyzer.calculate_stop_loss_target(symbol_upper, current_price, "BUY")
                
                if "error" in sl_target:
                    sl_target = {
                        "stop_loss": current_price * 0.97,
                        "target": current_price * 1.06,
                        "sl_percentage": 3.0,
                        "target_percentage": 6.0,
                        "risk_reward_ratio": 2.0
                    }
                
                risk_amount = account_size * (risk_per_trade / 100)
                price_diff = abs(current_price - sl_target["stop_loss"])
                position_size = int(risk_amount / price_diff) if price_diff > 0 else 0
                
                return {
                    "symbol": symbol_upper,
                    "status": "Symbol found in NSE list but not in current active futures",
                    "account_size": account_size,
                    "risk_per_trade_pct": risk_per_trade,
                    "position_sizing": {
                        "recommended_quantity": position_size,
                        "entry_price": current_price,
                        "stop_loss": sl_target["stop_loss"],
                        "target": sl_target["target"],
                        "risk_reward_ratio": sl_target["risk_reward_ratio"]
                    },
                    "total_available_symbols": len(all_symbols)
                }
            
            # Process active futures data
            recommendations = analyzer.generate_recommendations(symbol_data)
            if not recommendations:
                return {"error": "No recommendations available"}
            
            rec = recommendations[0]
            risk_amount = account_size * (risk_per_trade / 100)
            price_diff = abs(rec["current_price"] - rec["stop_loss"])
            
            if price_diff == 0:
                return {"error": "Invalid stop loss calculation"}
            
            position_size = int(risk_amount / price_diff)
            
            return {
                "symbol": symbol_upper,
                "status": "Active in current futures market",
                "account_size": account_size,
                "risk_per_trade_pct": risk_per_trade,
                "position_sizing": {
                    "recommended_quantity": position_size,
                    "entry_price": rec["current_price"],
                    "stop_loss": rec["stop_loss"],
                    "target": rec["target"],
                    "risk_reward_ratio": rec["risk_reward_ratio"]
                },
                "trading_plan": rec["trading_plan"],
                "action": rec["action"],
                "ai_score": rec["ai_score"],
                "total_available_symbols": len(all_symbols)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    @router.get("/available-symbols")
    def get_available_futures_symbols():
        """Get all available futures and options symbols from NSE"""
        try:
            underlying_info = nse_client.get_underlying_information()
            
            if "error" in underlying_info:
                return underlying_info
            
            result = {
                "indices": [],
                "stocks": [],
                "total_count": 0
            }
            
            if "data" in underlying_info:
                if "IndexList" in underlying_info["data"]:
                    for index in underlying_info["data"]["IndexList"]:
                        result["indices"].append({
                            "symbol": index["symbol"],
                            "name": index["underlying"],
                            "serial_number": index["serialNumber"]
                        })
                
                if "UnderlyingList" in underlying_info["data"]:
                    for stock in underlying_info["data"]["UnderlyingList"]:
                        result["stocks"].append({
                            "symbol": stock["symbol"],
                            "name": stock["underlying"],
                            "serial_number": stock["serialNumber"]
                        })
            
            result["total_count"] = len(result["indices"]) + len(result["stocks"])
            result["indices_count"] = len(result["indices"])
            result["stocks_count"] = len(result["stocks"])
            
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    @router.get("/trend-analysis")
    def get_trend_analysis():
        """Analyze trends across all futures with historical data"""
        try:
            oi_data = nse_client.get_oi_spurts_contracts()
            df = analyzer.extract_futures_data(oi_data)
            
            if df.empty:
                return {"error": "No futures data found"}
            
            trend_analysis = {
                "bullish_trends": [],
                "bearish_trends": [],
                "high_confidence": [],
                "summary": {
                    "total_analyzed": 0,
                    "bullish_count": 0,
                    "bearish_count": 0,
                    "high_confidence_count": 0
                }
            }
            
            # Use existing recommendations for trend analysis
            recommendations = analyzer.generate_recommendations(df)
            
            for rec in recommendations[:15]:  # Top 15 for trend analysis
                analysis_data = {
                    "symbol": rec["symbol"],
                    "current_price": rec["current_price"],
                    "price_change": rec["price_change_pct"],
                    "volume": rec["volume"],
                    "oi_change": rec["oi_change_pct"],
                    "ai_score": rec["ai_score"],
                    "recommendation": rec["action"],
                    "signals": rec["signals"]
                }
                
                trend_analysis["summary"]["total_analyzed"] += 1
                
                if rec["ai_score"] > 70:
                    trend_analysis["high_confidence"].append(analysis_data)
                    trend_analysis["summary"]["high_confidence_count"] += 1
                
                if rec["action"] == "BUY":
                    trend_analysis["bullish_trends"].append(analysis_data)
                    trend_analysis["summary"]["bullish_count"] += 1
                elif rec["action"] == "SELL":
                    trend_analysis["bearish_trends"].append(analysis_data)
                    trend_analysis["summary"]["bearish_count"] += 1
            
            return trend_analysis
            
        except Exception as e:
            return {"error": str(e)}
    
    return router