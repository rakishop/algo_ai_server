from fastapi import APIRouter
from nse_client import NSEClient
from data_processor import DataProcessor
from options_ai_analyzer import OptionsAIAnalyzer

def create_option_chain_routes(nse_client: NSEClient):
    router = APIRouter()
    processor = DataProcessor()
    
    @router.get("/api/v1/ai/option-chain-analysis")
    def analyze_option_chain(
        symbol: str = "NIFTY",
        expiry: str = None
    ):
        try:
            # Get option chain info first
            chain_info = nse_client.get_option_chain_info(symbol)
            if "error" in chain_info:
                return chain_info
            
            # Use first expiry if none provided
            if not expiry and "expiryDates" in chain_info:
                expiry = chain_info["expiryDates"][0]
            
            # Get option chain data
            chain_data = nse_client.get_option_chain(symbol, expiry)
            if "error" in chain_data:
                return chain_data
            
            # Extract and process option chain data
            processed_data = processor.extract_option_chain_data(chain_data)
            
            # Combine calls and puts for AI analysis
            all_options = processed_data["calls"] + processed_data["puts"]
            
            # Initialize AI analyzer
            ai_analyzer = OptionsAIAnalyzer()
            
            # Get AI recommendations
            ai_strategies = ai_analyzer.predict_optimal_strategies(all_options)
            
            # Calculate advanced metrics
            calls = processed_data["calls"]
            puts = processed_data["puts"]
            
            call_volume = sum(opt.get('totalTradedVolume', 0) for opt in calls)
            put_volume = sum(opt.get('totalTradedVolume', 0) for opt in puts)
            pcr = put_volume / call_volume if call_volume > 0 else 1
            
            call_oi = sum(opt.get('openInterest', 0) for opt in calls)
            put_oi = sum(opt.get('openInterest', 0) for opt in puts)
            oi_pcr = put_oi / call_oi if call_oi > 0 else 1
            
            # Find max pain
            strikes = sorted(set(opt.get('strikePrice', 0) for opt in all_options))
            max_pain_strike = 0
            min_pain = float('inf')
            
            for strike in strikes:
                call_pain = sum(max(0, strike - opt.get('strikePrice', 0)) * opt.get('openInterest', 0) 
                              for opt in calls if opt.get('strikePrice', 0) < strike)
                put_pain = sum(max(0, opt.get('strikePrice', 0) - strike) * opt.get('openInterest', 0) 
                             for opt in puts if opt.get('strikePrice', 0) > strike)
                total_pain = call_pain + put_pain
                
                if total_pain < min_pain:
                    min_pain = total_pain
                    max_pain_strike = strike
            
            return {
                "symbol": symbol,
                "expiry": expiry,
                "underlying_value": processed_data["underlying_value"],
                "timestamp": processed_data["timestamp"],
                "ai_strategies": ai_strategies,
                "option_chain_metrics": {
                    "total_strikes": processed_data["total_strikes"],
                    "call_volume": call_volume,
                    "put_volume": put_volume,
                    "volume_pcr": pcr,
                    "call_oi": call_oi,
                    "put_oi": put_oi,
                    "oi_pcr": oi_pcr,
                    "max_pain_strike": max_pain_strike
                },
                "available_expiries": chain_info.get("expiryDates", [])[:5],
                "analysis_method": "AI Option Chain Analysis"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    return router