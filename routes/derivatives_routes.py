from fastapi import APIRouter, Query, UploadFile, File
from typing import Optional
from .ml_derivatives_model import DerivativesMLModel
from utils.ai_risk_calculator import AIRiskCalculator
import json
from datetime import datetime

def create_derivatives_routes(nse_client):
    router = APIRouter(prefix="/api/v1/derivatives", tags=["derivatives"])
    ml_model = DerivativesMLModel()
    risk_calculator = AIRiskCalculator()
    
    @router.get("/market-snapshot")
    def get_derivatives_snapshot():
        return nse_client.get_derivatives_snapshot()

    @router.get("/active-underlyings")
    def get_most_active_underlying():
        return nse_client.get_most_active_underlying()

    @router.get("/oi-spurts-underlyings")
    def get_oi_spurts_underlyings():
        return nse_client.get_oi_spurts_underlyings()
    
    @router.get("/oi-spurts-contracts")
    def get_oi_spurts_contracts():
        return nse_client.get_oi_spurts_contracts()
    
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
        """Get derivatives equity snapshot data"""
        return nse_client.get_derivatives_equity_snapshot(limit)
    
    @router.get("/enhanced-snapshot")
    def get_enhanced_derivatives_snapshot(limit: Optional[int] = Query(20)):
        """Get enhanced derivatives data from multiple sources"""
        try:
            snapshot_data = nse_client.get_derivatives_equity_snapshot(limit)
            top20_data = nse_client.get_top20_derivatives_contracts()
            underlying_data = nse_client.get_most_active_underlying()
            oi_spurts_underlyings = nse_client.get_oi_spurts_underlyings()
            oi_spurts_contracts = nse_client.get_oi_spurts_contracts()
            
            return {
                "snapshot_contracts": snapshot_data.get("volume", {}).get("data", []),
                "top20_contracts": top20_data.get("data", []),
                "underlying_analysis": underlying_data.get("data", []),
                "oi_spurts_underlyings": oi_spurts_underlyings.get("data", []),
                "oi_spurts_contracts": oi_spurts_contracts.get("data", []),
                "data_sources": {
                    "snapshot_count": len(snapshot_data.get("volume", {}).get("data", [])),
                    "top20_count": len(top20_data.get("data", [])),
                    "underlying_count": len(underlying_data.get("data", [])),
                    "oi_spurts_underlyings_count": len(oi_spurts_underlyings.get("data", [])),
                    "oi_spurts_contracts_count": len(oi_spurts_contracts.get("data", []))
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    @router.get("/ai-trading-calls")
    def get_ai_derivatives_trading_calls(limit: Optional[int] = Query(20, description="Number of contracts to analyze")):
        """Get AI-powered derivatives trading recommendations with OI spurts analysis"""
        try:
            # Get derivatives data
            data = nse_client.get_derivatives_equity_snapshot(limit)
            oi_spurts = nse_client.get_oi_spurts_contracts()
            
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
                
                # AI-based risk calculation for options
                underlying_symbol = contract.get("underlying", "NIFTY")
                expiry_date = contract.get("expiryDate", "")
                
                # Calculate days to expiry
                try:
                    expiry_dt = datetime.strptime(expiry_date, "%d-%b-%Y")
                    days_to_expiry = (expiry_dt - datetime.now()).days
                except:
                    days_to_expiry = 7  # Default 1 week
                
                # Get AI-calculated levels for options
                ai_levels = risk_calculator.calculate_options_levels(
                    underlying_symbol, last_price, recommendation, 
                    option_type, strike, max(days_to_expiry, 1)
                )
                
                stop_loss = ai_levels['stop_loss']
                target = ai_levels['target']
                base_sl = ai_levels['sl_percentage']
                base_target = ai_levels['target_percentage']
                
                call_data = {
                    "identifier": contract.get("identifier"),
                    "underlying": contract.get("underlying"),
                    "option_type": option_type,
                    "strike_price": strike,
                    "expiry_date": contract.get("expiryDate"),
                    "last_price": last_price,
                    "stop_loss": stop_loss,
                    "target": target,
                    "sl_percentage": base_sl,
                    "target_percentage": base_target,
                    "risk_reward_ratio": ai_levels['risk_reward_ratio'],
                    "volatility": ai_levels.get('volatility', 0),
                    "implied_volatility": ai_levels.get('implied_volatility', 0),
                    "time_factor": ai_levels.get('time_factor', 1),
                    "days_to_expiry": days_to_expiry,
                    "delta_approx": ai_levels.get('delta_approx', 0),
                    "open_interest": ai_levels.get('open_interest', 0),
                    "underlying_value": ai_levels.get('underlying_value', 0),
                    "price_change_percent": price_change,
                    "volume": volume,
                    "open_interest": oi,
                    "underlying_value": underlying_value,
                    "ai_score": round(ai_score, 2),
                    "recommendation": recommendation,
                    "trend": trend,
                    "risk_level": risk_level,
                    "signals": signals,
                    "premium_turnover": contract.get("premiumTurnover", 0),
                    "trading_plan": {
                        "entry": f"{recommendation} at ₹{last_price}",
                        "stop_loss": f"₹{stop_loss} ({base_sl}%)",
                        "target": f"₹{target} ({base_target}%)",
                        "risk_reward": f"1:{ai_levels['risk_reward_ratio']}",
                        "time_analysis": f"Expiry: {days_to_expiry} days, Time Factor: {ai_levels.get('time_factor', 1)}",
                        "option_greeks": f"Delta: {ai_levels.get('delta_approx', 0)}, IV: {ai_levels.get('implied_volatility', 0)}%",
                        "market_data": f"OI: {ai_levels.get('open_interest', 0)}, Underlying: {ai_levels.get('underlying_value', 0)}"
                    }
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
        """Auto-train ML model with enhanced dataset and get recommendations"""
        try:
            # Get enhanced data from multiple sources
            enhanced_data = get_enhanced_derivatives_snapshot(limit)
            if "error" in enhanced_data:
                return enhanced_data
            
            # Use snapshot contracts for ML training and predictions
            training_data = {"volume": {"data": enhanced_data["snapshot_contracts"]}}
            
            # Auto-train model
            training_result = ml_model.train_with_historical_data(training_data)
            
            # Get ML predictions
            predictions = ml_model.predict_recommendations(training_data)
            
            # Add AI risk calculations to recommendations
            if "top_3_recommendations" in predictions:
                for rec in predictions["top_3_recommendations"]:
                    try:
                        # Calculate AI levels for each recommendation
                        underlying_symbol = rec.get("underlying", "NIFTY")
                        current_price = rec.get("last_price", 0)
                        action = rec.get("recommendation", "BUY")
                        option_type = rec.get("option_type", "Call")
                        strike_price = rec.get("strike_price", 0)
                        
                        # Calculate days to expiry
                        expiry_date = rec.get("expiry_date", "")
                        try:
                            expiry_dt = datetime.strptime(expiry_date, "%d-%b-%Y")
                            days_to_expiry = (expiry_dt - datetime.now()).days
                        except:
                            days_to_expiry = 7
                        
                        # Get AI-calculated levels
                        ai_levels = risk_calculator.calculate_options_levels(
                            underlying_symbol, current_price, action, 
                            option_type, strike_price, max(days_to_expiry, 1)
                        )
                        
                        # Add to recommendation
                        rec.update({
                            "stop_loss": ai_levels['stop_loss'],
                            "target": ai_levels['target'],
                            "sl_percentage": ai_levels['sl_percentage'],
                            "target_percentage": ai_levels['target_percentage'],
                            "risk_reward_ratio": ai_levels['risk_reward_ratio'],
                            "days_to_expiry": days_to_expiry,
                            "trading_plan": {
                                "entry": f"{action} at ₹{current_price}",
                                "stop_loss": f"₹{ai_levels['stop_loss']} ({ai_levels['sl_percentage']}%)",
                                "target": f"₹{ai_levels['target']} ({ai_levels['target_percentage']}%)",
                                "risk_reward": f"1:{ai_levels['risk_reward_ratio']}",
                                "expiry": f"{days_to_expiry} days remaining"
                            }
                        })
                    except Exception as e:
                        print(f"Error calculating levels for {rec.get('identifier', 'unknown')}: {e}")
            
            # Add to primary recommendation as well
            if "best_trades" in predictions and "primary_recommendation" in predictions["best_trades"]:
                primary = predictions["best_trades"]["primary_recommendation"]
                try:
                    underlying_symbol = primary.get("underlying", "NIFTY")
                    current_price = primary.get("last_price", 0)
                    action = primary.get("recommendation", "BUY")
                    option_type = primary.get("option_type", "Call")
                    strike_price = primary.get("strike_price", 0)
                    
                    # Calculate days to expiry
                    expiry_date = primary.get("expiry_date", "")
                    try:
                        expiry_dt = datetime.strptime(expiry_date, "%d-%b-%Y")
                        days_to_expiry = (expiry_dt - datetime.now()).days
                    except:
                        days_to_expiry = 7
                    
                    # Get AI-calculated levels
                    ai_levels = risk_calculator.calculate_options_levels(
                        underlying_symbol, current_price, action, 
                        option_type, strike_price, max(days_to_expiry, 1)
                    )
                    
                    # Add to primary recommendation
                    primary.update({
                        "stop_loss": ai_levels['stop_loss'],
                        "target": ai_levels['target'],
                        "sl_percentage": ai_levels['sl_percentage'],
                        "target_percentage": ai_levels['target_percentage'],
                        "risk_reward_ratio": ai_levels['risk_reward_ratio'],
                        "days_to_expiry": days_to_expiry,
                        "trading_plan": {
                            "entry": f"{action} at ₹{current_price}",
                            "stop_loss": f"₹{ai_levels['stop_loss']} ({ai_levels['sl_percentage']}%)",
                            "target": f"₹{ai_levels['target']} ({ai_levels['target_percentage']}%)",
                            "risk_reward": f"1:{ai_levels['risk_reward_ratio']}",
                            "expiry": f"{days_to_expiry} days remaining"
                        }
                    })
                except Exception as e:
                    print(f"Error calculating levels for primary recommendation: {e}")
            
            # Add enhanced info
            predictions["training_info"] = training_result
            predictions["auto_trained"] = True
            predictions["data_sources"] = enhanced_data["data_sources"]
            
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