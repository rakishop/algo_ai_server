from fastapi import APIRouter, Query, UploadFile, File
from typing import Optional
from .ml_derivatives_model import DerivativesMLModel
from utils.ai_risk_calculator import AIRiskCalculator
import json
from datetime import datetime
import time

def create_derivatives_routes(nse_client):
    router = APIRouter(prefix="/api/v1/derivatives", tags=["derivatives"])
    ml_model = DerivativesMLModel()
    risk_calculator = AIRiskCalculator()
    
    # Simple cache to avoid repeated API calls (5 minute cache)
    _cache = {}
    _cache_timeout = 300  # 5 minutes
    
    def get_cached_data(key, fetch_func):
        current_time = time.time()
        if key in _cache and (current_time - _cache[key]['timestamp']) < _cache_timeout:
            return _cache[key]['data']
        
        data = fetch_func()
        _cache[key] = {'data': data, 'timestamp': current_time}
        return data
    
    def calculate_consistent_ai_score(volume, price_change, current_price):
        """Single consistent AI scoring function used across all endpoints"""
        volume_score = min((volume / 1000) * 10, 40)
        price_score = min(abs(price_change) * 1.5, 30)
        liquidity_score = min((current_price * volume) / 100000, 20)
        base_score = 10
        
        ai_score = volume_score + price_score + liquidity_score + base_score
        return max(30, min(ai_score, 95))
    
    def _cleanup_old_files():
        """Keep 100 recent files with 1 hour difference for derivatives, equity, futures, and option chain data"""
        import os
        import glob
        from datetime import datetime
        
        max_files = 100
        hour_gap = 3600  # 1 hour in seconds
        
        def filter_files_by_hour_gap(files):
            if len(files) <= max_files:
                return files
            
            files.sort(key=os.path.getmtime, reverse=True)
            filtered = [files[0]]  # Keep most recent
            
            for file_path in files[1:]:
                if len(filtered) >= max_files:
                    break
                last_time = os.path.getmtime(filtered[-1])
                current_time = os.path.getmtime(file_path)
                if last_time - current_time >= hour_gap:
                    filtered.append(file_path)
            
            return filtered
        
        # Clean all data directories
        directories = [
            "training_data/derivatives_data_*.json",
            "training_data/equity_data_*.json",
            "training_data/futures_data_*.json",
            "option_chain_data/*_option_chain_*.json",
            "equity_training_data/equity_data_*.json",
            "futures_training_data/futures_data_*.json"
        ]
        
        for pattern in directories:
            files = glob.glob(pattern)
            if files:
                keep_files = filter_files_by_hour_gap(files)
                for file_path in files:
                    if file_path not in keep_files:
                        try:
                            os.remove(file_path)
                        except:
                            pass
    
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
        return get_cached_data(f"equity_snapshot_{limit}", lambda: nse_client.get_derivatives_equity_snapshot(limit))
    
    @router.get("/enhanced-snapshot")
    def get_enhanced_derivatives_snapshot(limit: Optional[int] = Query(20)):
        """Get enhanced derivatives data from multiple sources"""
        try:
            # Use cached data to avoid repeated API calls
            snapshot_data = get_cached_data(f"equity_snapshot_{limit}", lambda: nse_client.get_derivatives_equity_snapshot(limit))
            if "error" in snapshot_data:
                return snapshot_data
            
            # Only fetch additional data if specifically needed
            oi_spurts_contracts = get_cached_data("oi_spurts_contracts", lambda: nse_client.get_oi_spurts_contracts())
            
            return {
                "snapshot_contracts": snapshot_data.get("volume", {}).get("data", []),
                "oi_spurts_contracts": oi_spurts_contracts.get("data", []),
                "data_sources": {
                    "snapshot_count": len(snapshot_data.get("volume", {}).get("data", [])),
                    "oi_spurts_contracts_count": len(oi_spurts_contracts.get("data", []))
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    @router.get("/ai-trading-calls")
    def get_ai_derivatives_trading_calls(limit: Optional[int] = Query(20, description="Number of contracts to analyze")):
        """Get AI-powered derivatives trading recommendations with OI spurts analysis"""
        try:
            # Use cached data to avoid repeated API calls
            data = get_cached_data(f"equity_snapshot_{limit}", lambda: nse_client.get_derivatives_equity_snapshot(limit))
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
                
                # Use consistent AI scoring function
                ai_score = calculate_consistent_ai_score(volume, price_change, last_price)
                
                # Generate signals based on the scoring factors
                signals = []
                if volume > 2000000:
                    signals.append("High Volume")
                elif volume > 1000000:
                    signals.append("Good Volume")
                elif volume > 500000:
                    signals.append("Moderate Volume")
                
                if abs(price_change) > 50:
                    signals.append("Strong Movement")
                elif abs(price_change) > 20:
                    signals.append("Good Movement")
                elif abs(price_change) > 10:
                    signals.append("Moderate Movement")
                
                if oi > 50000:
                    signals.append("High OI")
                elif oi > 20000:
                    signals.append("Good OI")
                elif oi > 10000:
                    signals.append("Moderate OI")
                
                if underlying_value and strike:
                    moneyness = abs(underlying_value - strike) / underlying_value * 100
                    if moneyness < 1:
                        signals.append("ATM")
                    elif moneyness < 3:
                        signals.append("Near ATM")
                    elif moneyness < 5:
                        signals.append("Close to Money")
                
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
    
    @router.get("/option-chain/{symbol}")
    def get_option_chain_with_contract_info(symbol: str, expiry: Optional[str] = Query(None)):
        """Get option chain using two-step process: first get contract info, then fetch option chain"""
        try:
            # Step 1: Get contract info to fetch available expiry dates and strike prices
            contract_info = get_cached_data(f"contract_info_{symbol}", lambda: nse_client.get_option_chain_info(symbol))
            
            if "error" in contract_info:
                return contract_info
            
            # Extract available expiry dates from contract info
            available_expiries = []
            if "records" in contract_info and "expiryDates" in contract_info["records"]:
                available_expiries = contract_info["records"]["expiryDates"]
            
            # Use provided expiry or default to nearest expiry
            selected_expiry = expiry if expiry and expiry in available_expiries else (available_expiries[0] if available_expiries else None)
            
            if not selected_expiry:
                return {"error": "No expiry dates available for this symbol"}
            
            # Step 2: Fetch actual option chain data with the selected expiry
            option_chain_data = get_cached_data(f"option_chain_{symbol}_{selected_expiry}", lambda: nse_client.get_option_chain(symbol, selected_expiry))
            
            if "error" in option_chain_data:
                return option_chain_data
            
            # Enhance response with contract info metadata
            return {
                "symbol": symbol,
                "selected_expiry": selected_expiry,
                "available_expiries": available_expiries,
                "contract_info": contract_info,
                "option_chain": option_chain_data,
                "metadata": {
                    "total_expiries": len(available_expiries),
                    "data_source": "NSE via two-step process",
                    "timestamp": option_chain_data.get("timestamp", "")
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    @router.get("/contract-info/{symbol}")
    def get_contract_info(symbol: str):
        """Get contract info including available expiry dates and strike prices"""
        try:
            contract_info = get_cached_data(f"contract_info_{symbol}", lambda: nse_client.get_option_chain_info(symbol))
            
            if "error" in contract_info:
                return contract_info
            
            # Extract key information
            expiry_dates = []
            strike_prices = []
            underlying_value = 0
            
            if "records" in contract_info:
                records = contract_info["records"]
                expiry_dates = records.get("expiryDates", [])
                strike_prices = records.get("strikePrices", [])
                underlying_value = records.get("underlyingValue", 0)
            
            return {
                "symbol": symbol,
                "underlying_value": underlying_value,
                "expiry_dates": expiry_dates,
                "strike_prices": strike_prices,
                "total_expiries": len(expiry_dates),
                "total_strikes": len(strike_prices),
                "atm_strike": min(strike_prices, key=lambda x: abs(x - underlying_value)) if strike_prices and underlying_value else None,
                "contract_info": contract_info
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    @router.get("/multi-option-chains")
    def get_multiple_option_chains(symbols: str = Query("NIFTY,BANKNIFTY", description="Comma-separated symbols")):
        """Get option chains for multiple symbols using two-step process"""
        try:
            symbol_list = [s.strip().upper() for s in symbols.split(",")]
            results = {}
            
            for symbol in symbol_list:
                try:
                    # Step 1: Get contract info
                    contract_info = get_cached_data(f"contract_info_{symbol}", lambda s=symbol: nse_client.get_option_chain_info(s))
                    
                    if "error" in contract_info:
                        results[symbol] = {"error": contract_info["error"]}
                        continue
                    
                    # Extract available expiry dates
                    available_expiries = []
                    underlying_value = 0
                    strike_prices = []
                    
                    if "records" in contract_info:
                        records = contract_info["records"]
                        available_expiries = records.get("expiryDates", [])
                        underlying_value = records.get("underlyingValue", 0)
                        strike_prices = records.get("strikePrices", [])
                    
                    if not available_expiries:
                        results[symbol] = {"error": "No expiry dates available"}
                        continue
                    
                    # Use nearest expiry
                    nearest_expiry = available_expiries[0]
                    
                    # Step 2: Get option chain with specific expiry
                    option_chain_data = get_cached_data(f"option_chain_{symbol}_{nearest_expiry}", lambda s=symbol, e=nearest_expiry: nse_client.get_option_chain(s, e))
                    
                    if "error" in option_chain_data:
                        results[symbol] = {"error": option_chain_data["error"]}
                        continue
                    
                    # Find ATM strike
                    atm_strike = min(strike_prices, key=lambda x: abs(x - underlying_value)) if strike_prices and underlying_value else None
                    
                    # Extract relevant option data (ATM and near-ATM)
                    relevant_options = []
                    if "records" in option_chain_data and "data" in option_chain_data["records"]:
                        for strike_data in option_chain_data["records"]["data"]:
                            strike_price = strike_data.get("strikePrice", 0)
                            if underlying_value and strike_price:
                                # Only include strikes within ±3% of underlying price
                                price_diff_percent = abs(strike_price - underlying_value) / underlying_value * 100
                                if price_diff_percent <= 3:
                                    relevant_options.append(strike_data)
                    
                    results[symbol] = {
                        "symbol": symbol,
                        "underlying_value": underlying_value,
                        "selected_expiry": nearest_expiry,
                        "available_expiries": available_expiries,
                        "total_expiries": len(available_expiries),
                        "atm_strike": atm_strike,
                        "relevant_options": relevant_options[:10],  # Top 10 relevant options
                        "metadata": {
                            "total_strikes_available": len(strike_prices),
                            "relevant_strikes_count": len(relevant_options),
                            "data_source": "Two-step process"
                        }
                    }
                    
                except Exception as e:
                    results[symbol] = {"error": str(e)}
            
            return {
                "symbols_requested": symbol_list,
                "results": results,
                "summary": {
                    "total_symbols": len(symbol_list),
                    "successful": len([r for r in results.values() if "error" not in r]),
                    "failed": len([r for r in results.values() if "error" in r])
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    @router.get("/auto-ml-recommendations")
    def get_auto_ml_recommendations(limit: Optional[int] = Query(20)):
        """Auto-train ML model with enhanced dataset and get recommendations for all major indices"""
        try:
            # Define major indices to analyze
            major_indices = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYIT"]
            all_contracts = []
            
            # Fetch option chain data for each major index using two-step process
            for index in major_indices:
                try:
                    # Step 1: Get contract info
                    contract_info = get_cached_data(f"contract_info_{index}", lambda idx=index: nse_client.get_option_chain_info(idx))
                    
                    if "error" in contract_info or "records" not in contract_info:
                        continue
                    
                    # Get nearest expiry
                    expiry_dates = contract_info["records"].get("expiryDates", [])
                    if not expiry_dates:
                        continue
                    
                    nearest_expiry = expiry_dates[0]
                    
                    # Step 2: Get option chain with specific expiry
                    option_chain_data = get_cached_data(f"option_chain_{index}_{nearest_expiry}", lambda idx=index, exp=nearest_expiry: nse_client.get_option_chain(idx, exp))
                    
                    if "error" not in option_chain_data and "records" in option_chain_data:
                        # Extract contracts from option chain data
                        records = option_chain_data["records"]
                        if "data" in records:
                            # Get current underlying price to find ATM strikes
                            underlying_price = option_chain_data.get("records", {}).get("underlyingValue", 0)
                            
                            # Filter strikes to get ATM and near-ATM options only
                            relevant_strikes = []
                            for strike_data in records["data"]:
                                strike_price = strike_data.get("strikePrice", 0)
                                if underlying_price and strike_price:
                                    # Only include strikes within ±5% of underlying price
                                    price_diff_percent = abs(strike_price - underlying_price) / underlying_price * 100
                                    if price_diff_percent <= 5:  # Within 5% of ATM
                                        relevant_strikes.append(strike_data)
                            
                            # Sort by proximity to ATM and take top 6
                            relevant_strikes.sort(key=lambda x: abs(x.get("strikePrice", 0) - underlying_price))
                            
                            for strike_data in relevant_strikes[:6]:  # Top 6 ATM/near-ATM strikes
                                # Process Call options
                                if "CE" in strike_data:
                                    ce_data = strike_data["CE"]
                                    contract = {
                                        "identifier": f"OPTIDX{index}{ce_data.get('expiryDate', '')}{ce_data.get('optionType', 'CE')}{ce_data.get('strikePrice', 0)}.00",
                                        "underlying": index,
                                        "optionType": "Call",
                                        "strikePrice": ce_data.get("strikePrice", 0),
                                        "expiryDate": ce_data.get("expiryDate", ""),
                                        "lastPrice": ce_data.get("lastPrice", 0),
                                        "change": ce_data.get("change", 0),
                                        "pChange": ce_data.get("pChange", 0),
                                        "volume": ce_data.get("totalTradedVolume", 0),
                                        "numberOfContractsTraded": ce_data.get("totalTradedVolume", 0),
                                        "openInterest": ce_data.get("openInterest", 0),
                                        "underlyingValue": underlying_price,
                                        "premiumTurnover": ce_data.get("totalTradedVolume", 0) * ce_data.get("lastPrice", 0)
                                    }
                                    all_contracts.append(contract)
                                
                                # Process Put options
                                if "PE" in strike_data:
                                    pe_data = strike_data["PE"]
                                    contract = {
                                        "identifier": f"OPTIDX{index}{pe_data.get('expiryDate', '')}{pe_data.get('optionType', 'PE')}{pe_data.get('strikePrice', 0)}.00",
                                        "underlying": index,
                                        "optionType": "Put",
                                        "strikePrice": pe_data.get("strikePrice", 0),
                                        "expiryDate": pe_data.get("expiryDate", ""),
                                        "lastPrice": pe_data.get("lastPrice", 0),
                                        "change": pe_data.get("change", 0),
                                        "pChange": pe_data.get("pChange", 0),
                                        "volume": pe_data.get("totalTradedVolume", 0),
                                        "numberOfContractsTraded": pe_data.get("totalTradedVolume", 0),
                                        "openInterest": pe_data.get("openInterest", 0),
                                        "underlyingValue": underlying_price,
                                        "premiumTurnover": pe_data.get("totalTradedVolume", 0) * pe_data.get("lastPrice", 0)
                                    }
                                    all_contracts.append(contract)
                except Exception as e:
                    print(f"Error fetching option chain for {index}: {e}")
                    continue
            
            # Fallback to equity snapshot if no option chain data
            if not all_contracts:
                snapshot_data = get_cached_data(f"equity_snapshot_{limit}", lambda: nse_client.get_derivatives_equity_snapshot(limit))
                if "error" in snapshot_data:
                    return snapshot_data
                all_contracts = snapshot_data.get("volume", {}).get("data", [])
            
            if not all_contracts:
                return {"error": "No contracts data available"}
            
            # Ensure balanced representation from each index (at least 3 per index)
            contracts_by_index = {}
            for contract in all_contracts:
                underlying = contract.get("underlying", "UNKNOWN")
                if underlying not in contracts_by_index:
                    contracts_by_index[underlying] = []
                contracts_by_index[underlying].append(contract)
            
            # Sort contracts within each index by volume
            for index in contracts_by_index:
                contracts_by_index[index].sort(key=lambda x: x.get("numberOfContractsTraded", 0), reverse=True)
            
            # Take at least 3 from each index, then fill remaining with top overall
            contracts = []
            min_per_index = 3
            
            # First, take minimum from each index
            for index, index_contracts in contracts_by_index.items():
                contracts.extend(index_contracts[:min_per_index])
            
            # If we still have room, add more from top overall
            if len(contracts) < limit:
                remaining_slots = limit - len(contracts)
                all_remaining = []
                for index, index_contracts in contracts_by_index.items():
                    all_remaining.extend(index_contracts[min_per_index:])
                all_remaining.sort(key=lambda x: x.get("numberOfContractsTraded", 0), reverse=True)
                contracts.extend(all_remaining[:remaining_slots])
            
            # Limit to requested size
            contracts = contracts[:limit]
            
            # Smart historical data fetching - only for top 3 contracts with caching
            enhanced_contracts = []
            unique_symbols = set()
            
            # Get historical data for top 3 unique symbols only
            for contract in contracts[:10]:  # Check top 10 to find 3 unique symbols
                symbol = contract.get("underlying")
                if symbol and symbol not in unique_symbols and len(unique_symbols) < 3:
                    unique_symbols.add(symbol)
                    
                    try:
                        option_type = "CE" if contract.get("optionType") == "Call" else "PE"
                        strike_price = contract.get("strikePrice")
                        expiry_date = contract.get("expiryDate")
                        
                        if all([symbol, strike_price, expiry_date]):
                            from datetime import datetime, timedelta
                            to_date = datetime.now().strftime("%d-%m-%Y")
                            from_date = (datetime.now() - timedelta(days=30)).strftime("%d-%m-%Y")
                            
                            # Use caching to avoid repeated calls
                            cache_key = f"options_hist_{symbol}_{option_type}_{strike_price}_{expiry_date}"
                            if cache_key not in _cache or (time.time() - _cache[cache_key]['timestamp']) > _cache_timeout:
                                hist_data = nse_client.get_options_historical_data(
                                    symbol, option_type, strike_price, expiry_date, from_date, to_date
                                )
                                _cache[cache_key] = {'data': hist_data, 'timestamp': time.time()}
                            else:
                                hist_data = _cache[cache_key]['data']
                            
                            if "error" not in hist_data and hist_data.get("data"):
                                import pandas as pd
                                df = pd.DataFrame(hist_data["data"])
                                df['close'] = pd.to_numeric(df['FH_CLOSING_PRICE'], errors='coerce')
                                df['volume'] = pd.to_numeric(df['FH_TOT_TRADED_QTY'], errors='coerce')
                                
                                contract["historical_analysis"] = {
                                    "avg_volume": float(df['volume'].mean()),
                                    "price_volatility": float(df['close'].std()),
                                    "trend": "BULLISH" if df['close'].iloc[0] > df['close'].mean() else "BEARISH"
                                }
                    except Exception as e:
                        pass
                
                enhanced_contracts.append(contract)
            
            # Add remaining contracts without historical analysis
            enhanced_contracts.extend(contracts[len(enhanced_contracts):])
            training_data = {"volume": {"data": enhanced_contracts}}
            
            # Clean up old files before training
            _cleanup_old_files()
            
            # Auto-train model
            training_result = ml_model.train_with_historical_data(training_data)
            
            # Get ML predictions with balanced index representation
            predictions = ml_model.predict_recommendations(training_data)
            
            # Get ML predictions
            predictions = ml_model.predict_recommendations(training_data)
            
            # Add index-wise breakdown in recommendations - ensure all indices are represented
            predictions["recommendations_by_index"] = {}
            
            # Get all recommendations from training data grouped by index
            all_training_recs = []
            for contract in contracts:
                # Convert contract to recommendation format for consistency
                underlying = contract.get("underlying", "UNKNOWN")
                option_type = contract.get("optionType", "Call")
                price_change = contract.get("pChange", 0)
                volume = contract.get("numberOfContractsTraded", 0)
                
                # Use consistent AI scoring function
                ai_score = calculate_consistent_ai_score(volume, price_change, contract.get("lastPrice", 0))
                
                # Determine recommendation based on option type and price movement
                if option_type == "Call":
                    recommendation = "BUY CALL" if price_change > 0 else "SELL CALL"
                    trend = "BULLISH" if price_change > 0 else "BEARISH"
                else:  # Put
                    recommendation = "BUY PUT" if price_change > 0 else "SELL PUT"
                    trend = "BEARISH" if price_change > 0 else "BULLISH"
                
                # Calculate realistic stop-loss and targets
                current_price = contract.get("lastPrice", 0)
                if current_price > 0:
                    if "BUY" in recommendation:
                        stop_loss = current_price * 0.85  # 15% stop loss for buying
                        target = current_price * 1.25     # 25% target for buying
                    else:  # SELL
                        stop_loss = current_price * 1.15  # 15% stop loss for selling
                        target = current_price * 0.75     # 25% target for selling
                    
                    sl_percentage = ((stop_loss - current_price) / current_price) * 100
                    target_percentage = ((target - current_price) / current_price) * 100
                    risk_reward_ratio = abs(target_percentage / sl_percentage) if sl_percentage != 0 else 1.0
                else:
                    stop_loss = target = sl_percentage = target_percentage = risk_reward_ratio = None
                
                rec = {
                    "identifier": contract.get("identifier", ""),
                    "underlying": underlying,
                    "option_type": option_type,
                    "strike_price": contract.get("strikePrice", 0),
                    "ai_score": round(ai_score, 2),
                    "recommendation": recommendation,
                    "trend": trend,
                    "last_price": current_price,
                    "price_change": price_change,
                    "volume": volume,
                    "expiry_date": contract.get("expiryDate", ""),
                    "risk_level": "LOW" if ai_score > 70 else "MEDIUM" if ai_score > 50 else "HIGH",
                    "stop_loss": round(stop_loss, 2) if stop_loss else None,
                    "target": round(target, 2) if target else None,
                    "sl_percentage": round(sl_percentage, 2) if sl_percentage else None,
                    "target_percentage": round(target_percentage, 2) if target_percentage else None,
                    "risk_reward_ratio": round(risk_reward_ratio, 2) if risk_reward_ratio else None,
                    "trading_plan": {
                        "entry": f"{recommendation} at ₹{current_price}",
                        "stop_loss": f"₹{round(stop_loss, 2)} ({round(sl_percentage, 2)}%)" if stop_loss else "N/A",
                        "target": f"₹{round(target, 2)} ({round(target_percentage, 2)}%)" if target else "N/A",
                        "risk_reward": f"1:{round(risk_reward_ratio, 2)}" if risk_reward_ratio else "N/A"
                    } if current_price > 0 else None
                }
                all_training_recs.append(rec)
            
            # Group by index and take top 3 from each
            training_by_index = {}
            for rec in all_training_recs:
                underlying = rec.get("underlying", "UNKNOWN")
                if underlying not in training_by_index:
                    training_by_index[underlying] = []
                training_by_index[underlying].append(rec)
            
            # Sort each index by AI score and take top 3
            for index, index_recs in training_by_index.items():
                index_recs.sort(key=lambda x: (x.get("ai_score", 0), x.get("volume", 0)), reverse=True)
                predictions["recommendations_by_index"][index] = index_recs[:3]
            
            # Calculate confidence from ML predictions
            if "top_3_recommendations" in predictions:
                avg_score = sum(rec.get("ai_score", 50) for rec in predictions["top_3_recommendations"]) / len(predictions["top_3_recommendations"])
                predictions["confidence_level"] = "HIGH" if avg_score > 70 else "MEDIUM" if avg_score > 50 else "LOW"
            
            # Keep existing best_trades from ML model
            # ML model handles best_trades calculation
            
            # Fix trading strategy based on actual recommendations
            if "trading_strategy" in predictions:
                all_recs = []
                if "top_3_recommendations" in predictions:
                    all_recs.extend(predictions["top_3_recommendations"])
                if "recommendations_by_index" in predictions:
                    for idx_recs in predictions["recommendations_by_index"].values():
                        all_recs.extend(idx_recs)
                
                sell_put_count = len([r for r in all_recs if "SELL PUT" in r.get("recommendation", "")])
                buy_call_count = len([r for r in all_recs if "BUY CALL" in r.get("recommendation", "")])
                
                if sell_put_count > buy_call_count:
                    predictions["trading_strategy"] = "Focus on PUT selling"
                elif buy_call_count > sell_put_count:
                    predictions["trading_strategy"] = "Focus on CALL buying"
                else:
                    predictions["trading_strategy"] = "Mixed strategy"
            
            # Add enhanced info with index breakdown
            index_breakdown = {}
            for contract in contracts:
                underlying = contract.get("underlying", "UNKNOWN")
                if underlying not in index_breakdown:
                    index_breakdown[underlying] = {"count": 0, "volume": 0}
                index_breakdown[underlying]["count"] += 1
                index_breakdown[underlying]["volume"] += contract.get("numberOfContractsTraded", 0)
            
            predictions["training_info"] = training_result
            predictions["auto_trained"] = True
            predictions["data_sources"] = {
                "total_contracts": len(contracts),
                "indices_analyzed": list(index_breakdown.keys()),
                "index_breakdown": index_breakdown,
                "min_contracts_per_index": 3,
                "balanced_representation": True,
                "optimized_for_multi_index": True
            }
            
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