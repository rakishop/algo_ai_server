import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any
from utils.ai_risk_calculator import AIRiskCalculator
import warnings
import os
import json
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

class FuturesAnalyzer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.risk_calculator = AIRiskCalculator()
        self.training_data_dir = "futures_training_data"
        os.makedirs(self.training_data_dir, exist_ok=True)
    
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
                score += 25
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
            
            # Skip if AI calculator fails - only use AI-based calculations
            if "error" in sl_target:
                continue
            
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
            current_price = current_data.get("ltp", 0)
            price_change = current_data.get("pChange", 0)
            oi_change = current_data.get("pChangeInOI", 0)
            volume = current_data.get("volume", 0)
            category = current_data.get("category", "")
            
            pattern_score = 0
            signals = []
            
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
            
            if volume > 100000:
                pattern_score += 20
                signals.append("High Volume Confirmation")
            elif volume > 50000:
                pattern_score += 10
                signals.append("Good Volume")
            
            if abs(price_change) > 3:
                pattern_score += 15
                signals.append(f"Strong Momentum: {price_change:.1f}%")
            elif abs(price_change) > 1:
                pattern_score += 8
                signals.append(f"Good Momentum: {price_change:.1f}%")
            
            if abs(oi_change) > 30:
                pattern_score += 15
                signals.append(f"High OI Activity: {oi_change:.1f}%")
            elif abs(oi_change) > 15:
                pattern_score += 8
                signals.append(f"Moderate OI Activity: {oi_change:.1f}%")
            
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
        
        symbol_best = {}
        
        for _, row in df.iterrows():
            pattern_analysis = self.analyze_futures_pattern(row["symbol"], row.to_dict())
            
            if "error" in pattern_analysis:
                continue
            
            pattern = pattern_analysis["pattern_analysis"]
            
            if pattern["confidence"] >= 60:
                symbol = row["symbol"]
                
                volume_score = min(row["volume"] / 10000, 10)
                oi_activity_score = min(abs(row["pChangeInOI"]) / 10, 5)
                momentum_score = min(abs(row["pChange"]) * 2, 10)
                
                expiry_date = row["expiryDate"]
                if "Sep" in expiry_date:
                    expiry_score = 3
                else:
                    expiry_score = 2
                
                total_score = (
                    pattern["confidence"] * 0.4 +
                    volume_score * 0.2 +
                    oi_activity_score * 0.2 +
                    momentum_score * 0.1 +
                    expiry_score * 0.1
                )
                
                if symbol not in symbol_best or total_score > symbol_best[symbol].get("total_score", 0):
                    sl_target = self.calculate_stop_loss_target(
                        row["symbol"], row["ltp"], pattern["recommendation"]
                    )
                    
                    if "error" in sl_target:
                        continue
                    
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
        
        best_bets = list(symbol_best.values())
        return sorted(best_bets, key=lambda x: x["total_score"], reverse=True)
    
    def train_ml_model(self, df: pd.DataFrame):
        """Train ML model for future predictions"""
        if df.empty or len(df) < 10:
            return {"error": "Insufficient data for training"}
        
        features = ["pChange", "pChangeInOI", "volume", "latestOI"]
        X = df[features].fillna(0)
        y = ((df["pChange"] > 0) & (df["pChangeInOI"] > 0)).astype(int)
        
        X_scaled = self.scaler.fit_transform(X)
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
            
            sl_target = self.calculate_stop_loss_target(
                row["symbol"], row["ltp"], action
            )
            
            # Skip if AI calculator fails - only use AI-based calculations
            if "error" in sl_target:
                continue
            
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
            best_bets = analyzer.get_best_futures_bets(df)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"futures_data_{timestamp}.json"
            filepath = os.path.join(analyzer.training_data_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(df.to_dict('records'), f)
            
            training_result = analyzer.train_ml_model(df)
            ml_predictions = analyzer.predict_recommendations(df)
            
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
            
            stock_futures = df[df["instrumentType"] == "FUTSTK"]
            
            if stock_futures.empty:
                return {"error": "No stock futures found"}
            
            analysis = analyzer.analyze_futures(stock_futures)
            recommendations = analyzer.generate_recommendations(stock_futures)
            best_bets = analyzer.get_best_futures_bets(stock_futures)
            
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
    
    @router.get("/position-sizing")
    def calculate_position_sizing(
        symbol: str = Query(..., description="Symbol like NIFTY, SBIN, ICICI")
    ):
        """Calculate position sizing for futures trading"""
        try:
            # Get futures data for the symbol
            oi_data = nse_client.get_oi_spurts_contracts()
            df = analyzer.extract_futures_data(oi_data)
            
            symbol = symbol.upper()
            
            # Check if symbol exists in master-quote first (skip for indices)
            if symbol not in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]:
                master_quote = nse_client.get_futures_master_quote()
                if "error" not in master_quote and symbol not in master_quote:
                    return {
                        "error": f"Symbol {symbol} not available for futures trading",
                        "available_symbols": master_quote[:20] if isinstance(master_quote, list) else []
                    }
            
            # Find the symbol in futures data
            symbol_data = df[df["symbol"] == symbol]
            
            if symbol_data.empty:
                # Get historical data for analysis
                from datetime import datetime, timedelta
                to_date = datetime.now().strftime("%d-%m-%Y")
                from_date = (datetime.now() - timedelta(days=180)).strftime("%d-%m-%Y")
                
                # For indices, use quote-derivative API instead of historical
                if symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]:
                    quote_data = nse_client.get_quote_derivative(symbol)
                    if "error" in quote_data:
                        return {"error": f"Could not fetch data for {symbol}"}
                    
                    stock_data = quote_data["stocks"][0]
                    metadata = stock_data["metadata"]
                    current_price = metadata["lastPrice"]
                    price_change = ((current_price - metadata["prevClose"]) / metadata["prevClose"]) * 100
                    
                    volume = 0
                    try:
                        volume = int(stock_data["marketDeptOrderBook"]["tradeInfo"].get("totalTradedVolume", 0))
                    except:
                        volume = 0
                    
                    best_contract = {
                        "symbol": symbol,
                        "ltp": float(current_price),
                        "pChange": float(price_change),
                        "pChangeInOI": 0,
                        "volume": volume,
                        "category": "Index Futures"
                    }
                else:
                    hist_data = nse_client.get_historical_data(symbol, from_date, to_date)
                    if "error" in hist_data or not hist_data.get("data"):
                        return {"error": f"Could not fetch historical data for {symbol}"}
                
                    # Analyze historical data with pandas
                    import pandas as pd
                    df = pd.DataFrame(hist_data["data"])
                    latest = df.iloc[0]
                    
                    # Calculate technical indicators
                    df['close'] = pd.to_numeric(df['CH_CLOSING_PRICE'], errors='coerce')
                    df['volume'] = pd.to_numeric(df['CH_TOT_TRADED_QTY'], errors='coerce')
                    df['VWAP'] = pd.to_numeric(df['VWAP'], errors='coerce')
                    
                    # Drop NaN values
                    df = df.dropna(subset=['close', 'volume', 'VWAP'])
                    
                    sma_5 = df['close'].head(5).mean()
                    sma_20 = df['close'].head(20).mean()
                    avg_volume = df['volume'].head(20).mean()
                    
                    current_price = latest["CH_LAST_TRADED_PRICE"]
                    price_change = ((current_price - latest["CH_PREVIOUS_CLS_PRICE"]) / latest["CH_PREVIOUS_CLS_PRICE"]) * 100
                    
                    # Generate signals
                    signals = []
                    if current_price > sma_5 > sma_20:
                        signals.append("Bullish Trend")
                    elif current_price < sma_5 < sma_20:
                        signals.append("Bearish Trend")
                    
                    if latest["CH_TOT_TRADED_QTY"] > avg_volume * 1.5:
                        signals.append("High Volume")
                    
                    best_contract = {
                        "symbol": symbol,
                        "ltp": float(current_price),
                        "pChange": float(price_change),
                        "pChangeInOI": 0,
                        "volume": int(latest["CH_TOT_TRADED_QTY"]),
                        "category": "Historical Analysis",
                        "high": float(latest["CH_TRADE_HIGH_PRICE"]),
                        "low": float(latest["CH_TRADE_LOW_PRICE"]),
                        "sma_5": float(sma_5),
                        "sma_20": float(sma_20),
                        "signals": signals,
                        "week_52_high": float(latest["CH_52WEEK_HIGH_PRICE"]),
                        "week_52_low": float(latest["CH_52WEEK_LOW_PRICE"])
                    }
            else:
                best_contract = symbol_data.iloc[0]
                current_price = best_contract["ltp"]
            
            # Calculate AI-based stop loss and target
            action = "BUY"  # Default action
            sl_target = analyzer.calculate_stop_loss_target(symbol.upper(), current_price, action)
            
            if "error" in sl_target:
                return {"error": f"Could not calculate risk levels for {symbol.upper()}"}
            
            # Try to get lot size from NSE quote-derivative API
            lot_size = 1  # Default
            try:
                quote_data = nse_client.get_quote_derivative(symbol.upper())
                if "error" not in quote_data and "stocks" in quote_data:
                    for stock in quote_data["stocks"]:
                        if "marketDeptOrderBook" in stock and "tradeInfo" in stock["marketDeptOrderBook"]:
                            lot_size = stock["marketDeptOrderBook"]["tradeInfo"]["marketLot"]
                            break
            except:
                pass
            
            # Fallback lot size mapping if API fails
            if lot_size <= 1:
                lot_sizes = {
                    "NIFTY": 75, "BANKNIFTY": 15, "FINNIFTY": 40, "MIDCPNIFTY": 75,
                    "SBIN": 3000, "ICICIBANK": 1375, "HDFCBANK": 550, "AXISBANK": 1200,
                    "KOTAKBANK": 400, "INDUSINDBK": 900, "BAJFINANCE": 125, "BAJAJFINSV": 125,
                    "RELIANCE": 250, "TCS": 150, "INFY": 300, "WIPRO": 3000, "HCLTECH": 500,
                    "TECHM": 400, "LT": 150, "MARUTI": 50, "TATAMOTORS": 1500, "M&M": 75,
                    "BHARTIARTL": 275, "ADANIENT": 250, "ADANIPORTS": 400, "ADANIGREEN": 250,
                    "ITC": 3200, "HINDUNILVR": 300, "ASIANPAINT": 150, "NESTLEIND": 50,
                    "TITAN": 300, "ULTRACEMCO": 150, "JSWSTEEL": 800, "TATASTEEL": 1500,
                    "HINDALCO": 1000, "VEDL": 4800, "COALINDIA": 4000, "NTPC": 2000,
                    "POWERGRID": 2700, "ONGC": 2750, "IOC": 1000, "BPCL": 500, "GAIL": 1250,
                    "SAIL": 10000, "NMDC": 8400, "TRENT": 125, "DMART": 100, "GODREJCP": 1000,
                    "BRITANNIA": 200, "DABUR": 1800, "MARICO": 3000, "COLPAL": 400,
                    "PIDILITIND": 300, "BERGEPAINT": 1200, "AKZOINDIA": 500, "KANSAINER": 400,
                    "DRREDDY": 125, "SUNPHARMA": 400, "CIPLA": 1000, "LUPIN": 500,
                    "AUROPHARMA": 1000, "DIVISLAB": 50, "BIOCON": 2500, "TORNTPHARM": 125,
                    "APOLLOHOSP": 125, "FORTIS": 2000, "MAXHEALTH": 500, "BEL": 4000,
                    "HAL": 500, "BHEL": 8000, "LTIM": 600, "PERSISTENT": 125, "COFORGE": 400,
                    "MPHASIS": 200, "OFSS": 100, "MINDTREE": 400, "LTTS": 700
                }
                lot_size = lot_sizes.get(symbol.upper(), 1)
            
            stop_loss_distance = abs(current_price - sl_target["stop_loss"])
            total_investment = current_price * lot_size
            max_loss = stop_loss_distance * lot_size
            potential_profit = abs(sl_target["target"] - current_price) * lot_size
            
            # Analyze futures pattern for trade recommendation
            pattern_analysis = analyzer.analyze_futures_pattern(symbol.upper(), best_contract)
            
            # Determine if trade should be taken
            should_trade = False
            trade_action = "HOLD"
            confidence = 0
            reasons = []
            
            if "pattern_analysis" in pattern_analysis:
                pattern = pattern_analysis["pattern_analysis"]
                confidence = pattern["confidence"]
                
                # AI model prediction using historical data
                if 'df' in locals() and len(df) > 0 and 'close' in df.columns:
                    # Train AI model with historical data
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.preprocessing import StandardScaler
                    
                    # Advanced technical features
                    df['returns'] = df['close'].pct_change()
                    df['volatility'] = df['returns'].rolling(5).std()
                    df['rsi'] = 100 - (100 / (1 + (df['close'].diff().where(df['close'].diff() > 0, 0).rolling(14).mean() / 
                                                   df['close'].diff().where(df['close'].diff() < 0, 0).abs().rolling(14).mean())))
                    df['vwap_signal'] = (df['close'] > df['VWAP']).astype(int)
                    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
                    df['bb_upper'] = df['close'].rolling(20).mean() + (df['close'].rolling(20).std() * 2)
                    df['bb_lower'] = df['close'].rolling(20).mean() - (df['close'].rolling(20).std() * 2)
                    df['bb_signal'] = ((df['close'] < df['bb_lower']) | (df['close'] > df['bb_upper'])).astype(int)
                    df['volume_sma'] = df['volume'].rolling(20).mean()
                    df['volume_signal'] = (df['volume'] > df['volume_sma'] * 1.5).astype(int)
                    
                    # Create target (next day up/down)
                    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
                    
                    # Features for ML
                    features = ['returns', 'volatility', 'rsi', 'vwap_signal', 'macd', 'bb_signal', 'volume_signal']
                    
                    # Align X and y by dropping NaN from both
                    df_clean = df[features + ['target']].dropna()
                    X = df_clean[features]
                    y = df_clean['target']
                    
                    if len(X) > 10 and len(y) > 10 and len(X) == len(y):
                        # Train model
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X[:-1])  # Exclude last row
                        y_train = y[:-1]
                        
                        from sklearn.ensemble import GradientBoostingClassifier
                        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
                        model.fit(X_scaled, y_train)
                        
                        # Predict for current data
                        current_features = X.iloc[-1:]
                        current_scaled = scaler.transform(current_features)
                        prediction = model.predict(current_scaled)[0]
                        ai_confidence = model.predict_proba(current_scaled)[0].max() * 100
                        
                        # AI decision
                        if ai_confidence > 70:
                            should_trade = True
                            trade_action = "BUY" if prediction == 1 else "SELL"
                            confidence = ai_confidence
                            reasons.append(f"AI Model: {ai_confidence:.1f}% confidence {trade_action}")
                        elif ai_confidence > 60:
                            should_trade = True
                            trade_action = "BUY" if prediction == 1 else "SELL"
                            confidence = ai_confidence
                            reasons.append(f"AI Model: {ai_confidence:.1f}% confidence {trade_action}")
                        else:
                            confidence = ai_confidence
                            reasons.append(f"AI Model: Low confidence ({ai_confidence:.1f}%) - avoid trade")
                    else:
                        reasons.append("Insufficient historical data for AI model")
                else:
                    # Fallback to pattern analysis
                    if confidence >= 60:
                        should_trade = True
                        trade_action = "BUY" if pattern["recommendation"] in ["STRONG BUY", "BUY"] else "SELL"
                        reasons.append(f"Pattern analysis: {confidence}% confidence {trade_action}")
                    else:
                        reasons.append(f"Pattern analysis: Low confidence ({confidence}%) - avoid trade")
                
                # Add pattern reasons
                category = best_contract["category"]
                if category == "Rise-in-OI-Rise":
                    reasons.append("Fresh long build-up")
                elif category == "Slide-in-OI-Slide":
                    reasons.append("Fresh short build-up")
                
                if best_contract["volume"] > 100000:
                    reasons.append("High volume")
                if abs(best_contract["pChange"]) > 2:
                    reasons.append(f"Strong momentum: {best_contract['pChange']:.1f}%")
            
            # Recalculate for trade action
            if should_trade:
                sl_target = analyzer.calculate_stop_loss_target(symbol.upper(), current_price, trade_action)
                if "error" not in sl_target:
                    stop_loss_distance = abs(current_price - sl_target["stop_loss"])
                    total_investment = current_price * lot_size
                    max_loss = stop_loss_distance * lot_size
                    potential_profit = abs(sl_target["target"] - current_price) * lot_size
            
            return {
                "symbol": symbol.upper(),
                "trade_recommendation": {
                    "should_trade": should_trade,
                    "action": trade_action,
                    "confidence": confidence,
                    "reasons": reasons
                },
                "entry_exit_plan": {
                    "entry": current_price,
                    "stop_loss": sl_target["stop_loss"] if should_trade else None,
                    "target": sl_target["target"] if should_trade else None,
                    "risk_reward": sl_target["risk_reward_ratio"] if should_trade else None
                } if should_trade else {"message": "No trade recommended"},
                "position_sizing": {
                    "lots": 1 if should_trade else 0,
                    "lot_size": int(lot_size),
                    "quantity": int(lot_size) if should_trade else 0,
                    "investment": float(total_investment) if should_trade else 0,
                    "max_loss": float(max_loss) if should_trade else 0,
                    "potential_profit": float(potential_profit) if should_trade else 0
                },
                "market_data": {
                    "price_change": float(best_contract["pChange"]),
                    "oi_change": float(best_contract["pChangeInOI"]),
                    "volume": int(best_contract["volume"]),
                    "category": str(best_contract["category"])
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    return router