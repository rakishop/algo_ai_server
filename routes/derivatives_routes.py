from fastapi import APIRouter, Query, UploadFile, File
from typing import Optional
from .ml_derivatives_model import DerivativesMLModel
import json

def create_derivatives_routes(nse_client):
    router = APIRouter(prefix="/api/v1/derivatives", tags=["derivatives"])
    ml_model = DerivativesMLModel()
    
    @router.get("/market-snapshot")
    def get_derivatives_snapshot():
        return nse_client.get_derivatives_snapshot()

    @router.get("/active-underlyings")
    def get_most_active_underlying():
        return nse_client.get_most_active_underlying()

    @router.get("/open-interest-spurts")
    def get_oi_spurts():
        underlyings = nse_client.get_oi_spurts_underlyings()
        contracts = nse_client.get_oi_spurts_contracts()
        return {
            "underlyings": underlyings,
            "contracts": contracts
        }
    
    @router.get("/equity-snapshot")
    def get_derivatives_equity_snapshot(limit: Optional[int] = Query(20, description="Number of contracts to fetch")):
        """Get derivatives equity snapshot with volume data"""
        return nse_client.get_derivatives_equity_snapshot(limit)
    
    @router.get("/ai-trading-calls")
    def get_ai_derivatives_trading_calls(limit: Optional[int] = Query(20, description="Number of contracts to analyze")):
        """Get AI-powered derivatives trading recommendations"""
        try:
            # Get derivatives data
            data = nse_client.get_derivatives_equity_snapshot(limit)
            
            if "error" in data:
                return data
            
            if "volume" not in data or "data" not in data["volume"]:
                return {"error": "No volume data available"}
            
            contracts = data["volume"]["data"]
            
            # AI Analysis for best trading calls
            trading_calls = []
            high_volume_calls = []
            momentum_calls = []
            
            for contract in contracts:
                # Calculate key metrics
                volume = contract.get("numberOfContractsTraded", 0)
                price_change = contract.get("pChange", 0)
                oi = contract.get("openInterest", 0)
                last_price = contract.get("lastPrice", 0)
                strike = contract.get("strikePrice", 0)
                underlying_value = contract.get("underlyingValue", 0)
                option_type = contract.get("optionType", "")
                
                # AI Scoring Algorithm
                ai_score = 0
                signals = []
                
                # Volume Analysis (40% weight)
                if volume > 5000000:
                    ai_score += 40
                    signals.append("Exceptional Volume")
                elif volume > 2000000:
                    ai_score += 30
                    signals.append("High Volume")
                elif volume > 1000000:
                    ai_score += 20
                    signals.append("Good Volume")
                elif volume > 500000:
                    ai_score += 10
                    signals.append("Moderate Volume")
                
                # Price Movement Analysis (30% weight)
                if abs(price_change) > 200:
                    ai_score += 30
                    signals.append("Exceptional Movement")
                elif abs(price_change) > 100:
                    ai_score += 25
                    signals.append("Strong Movement")
                elif abs(price_change) > 50:
                    ai_score += 20
                    signals.append("Good Movement")
                elif abs(price_change) > 20:
                    ai_score += 10
                    signals.append("Moderate Movement")
                
                # Open Interest Analysis (20% weight)
                if oi > 100000:
                    ai_score += 20
                    signals.append("Very High OI")
                elif oi > 50000:
                    ai_score += 15
                    signals.append("High OI")
                elif oi > 20000:
                    ai_score += 10
                    signals.append("Good OI")
                elif oi > 10000:
                    ai_score += 5
                    signals.append("Moderate OI")
                
                # Moneyness Analysis (10% weight)
                if underlying_value and strike:
                    moneyness = abs(underlying_value - strike) / underlying_value * 100
                    if moneyness < 1:  # Very near ATM
                        ai_score += 10
                        signals.append("ATM")
                    elif moneyness < 3:  # Near ATM
                        ai_score += 7
                        signals.append("Near ATM")
                    elif moneyness < 5:  # Close to money
                        ai_score += 4
                        signals.append("Close to Money")
                    elif moneyness < 10:
                        ai_score += 2
                        signals.append("Reasonable Strike")
                
                # Trading Recommendation Logic
                if option_type == "Call":
                    if price_change > 0:
                        recommendation = "BUY CALL"
                        trend = "BULLISH"
                    else:
                        recommendation = "SELL CALL"
                        trend = "BEARISH"
                else:  # Put option
                    if price_change > 0:
                        recommendation = "BUY PUT"
                        trend = "BEARISH"  # Put gaining = underlying falling
                    else:
                        recommendation = "SELL PUT"
                        trend = "BULLISH"  # Put losing = underlying rising
                
                # Risk Level
                if ai_score >= 70:
                    risk_level = "LOW"
                elif ai_score >= 50:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "HIGH"
                
                call_data = {
                    "identifier": contract.get("identifier"),
                    "underlying": contract.get("underlying"),
                    "option_type": option_type,
                    "strike_price": strike,
                    "expiry_date": contract.get("expiryDate"),
                    "last_price": last_price,
                    "price_change_percent": price_change,
                    "volume": volume,
                    "open_interest": oi,
                    "underlying_value": underlying_value,
                    "ai_score": round(ai_score, 2),
                    "recommendation": recommendation,
                    "trend": trend,
                    "risk_level": risk_level,
                    "signals": signals,
                    "premium_turnover": contract.get("premiumTurnover", 0)
                }
                
                trading_calls.append(call_data)
                
                # Categorize calls
                if volume > 2000000:
                    high_volume_calls.append(call_data)
                
                if abs(price_change) > 30:
                    momentum_calls.append(call_data)
            
            # Sort by AI score
            trading_calls.sort(key=lambda x: x["ai_score"], reverse=True)
            high_volume_calls.sort(key=lambda x: x["volume"], reverse=True)
            momentum_calls.sort(key=lambda x: abs(x["price_change_percent"]), reverse=True)
            
            # Market Analysis
            total_calls = len([c for c in trading_calls if c["option_type"] == "Call"])
            total_puts = len([c for c in trading_calls if c["option_type"] == "Put"])
            bullish_calls = len([c for c in trading_calls if c["trend"] == "BULLISH"])
            bearish_calls = len([c for c in trading_calls if c["trend"] == "BEARISH"])
            
            market_sentiment = "BULLISH" if bullish_calls > bearish_calls else "BEARISH" if bearish_calls > bullish_calls else "NEUTRAL"
            
            return {
                "analysis_timestamp": data.get("timestamp", ""),
                "total_contracts_analyzed": len(trading_calls),
                "market_sentiment": market_sentiment,
                "market_analysis": {
                    "total_calls": total_calls,
                    "total_puts": total_puts,
                    "bullish_signals": bullish_calls,
                    "bearish_signals": bearish_calls,
                    "call_put_ratio": round(total_calls / total_puts if total_puts > 0 else 0, 2)
                },
                "top_ai_recommendations": trading_calls[:10],
                "high_volume_opportunities": high_volume_calls[:5],
                "momentum_plays": momentum_calls[:5],
                "trading_strategy": {
                    "primary_focus": "High volume contracts with strong price movement",
                    "risk_management": "Monitor open interest and underlying price movement",
                    "market_outlook": f"Current sentiment is {market_sentiment.lower()}"
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    @router.post("/ml-train")
    async def train_ml_model(file: UploadFile = File(...)):
        """Train ML model with historical derivatives data"""
        try:
            content = await file.read()
            historical_data = json.loads(content)
            result = ml_model.train(historical_data)
            return result
        except Exception as e:
            return {"error": str(e)}
    
    @router.get("/ml-predictions")
    def get_ml_predictions(limit: Optional[int] = Query(20)):
        """Get ML-powered derivatives predictions"""
        try:
            current_data = nse_client.get_derivatives_equity_snapshot(limit)
            if "error" in current_data:
                return current_data
            
            predictions = ml_model.predict_recommendations(current_data)
            return predictions
        except Exception as e:
            return {"error": str(e)}
    
    @router.get("/auto-ml-recommendations")
    def get_auto_ml_recommendations(limit: Optional[int] = Query(20)):
        """Auto-train ML model with current data and get recommendations"""
        try:
            # Get current data
            current_data = nse_client.get_derivatives_equity_snapshot(limit)
            if "error" in current_data:
                return current_data
            
            # Auto-train model with all historical data + current data
            training_result = ml_model.train_with_historical_data(current_data)
            
            # Get ML predictions
            predictions = ml_model.predict_recommendations(current_data)
            
            # Combine training info with predictions
            predictions["training_info"] = training_result
            predictions["auto_trained"] = True
            
            return predictions
            
        except Exception as e:
            return {"error": str(e)}
    
    @router.get("/pandas-analysis")
    def get_pandas_analysis(limit: Optional[int] = Query(20)):
        """Get pandas-based data analysis of derivatives"""
        try:
            current_data = nse_client.get_derivatives_equity_snapshot(limit)
            if "error" in current_data:
                return current_data
            
            analysis = ml_model.analyze_data_with_pandas(current_data)
            return analysis
        except Exception as e:
            return {"error": str(e)}
    
    return router