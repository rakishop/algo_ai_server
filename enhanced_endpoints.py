from fastapi import APIRouter, HTTPException
from advanced_technical_analysis import AdvancedTechnicalAnalysis
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio

def create_enhanced_routes(nse_client, get_chart_data_func):
    router = APIRouter()
    technical_analyzer = AdvancedTechnicalAnalysis()
    
    @router.post("/api/v1/enhanced/comprehensive-analysis")
    def comprehensive_stock_analysis(
        tradingSymbol: str,
        chartPeriod: str = "I",
        timeInterval: int = 15
    ):
        """Complete technical analysis with all strategies"""
        try:
            # Get chart data
            chart_data = get_chart_data_func(tradingSymbol, chartPeriod, timeInterval)
            
            if chart_data.get("s") != "ok":
                return {"error": "Unable to fetch chart data"}
            
            # Calculate all indicators
            indicators = technical_analyzer.calculate_advanced_indicators(chart_data)
            
            if not indicators:
                return {"error": "Insufficient data for analysis"}
            
            # Run all strategies
            momentum_analysis = technical_analyzer.momentum_strategy(indicators, chart_data)
            mean_reversion_analysis = technical_analyzer.mean_reversion_strategy(indicators, chart_data)
            breakout_analysis = technical_analyzer.breakout_strategy(indicators, chart_data)
            
            # Current market data
            current_price = chart_data['c'][-1]
            prev_price = chart_data['c'][-2] if len(chart_data['c']) > 1 else current_price
            price_change = (current_price - prev_price) / prev_price * 100
            
            return {
                "symbol": tradingSymbol,
                "current_price": current_price,
                "price_change_pct": round(price_change, 2),
                "chart_period": chartPeriod,
                "time_interval": timeInterval,
                "technical_indicators": {
                    "rsi": round(indicators.get("rsi", 50), 2),
                    "macd": round(indicators.get("macd", 0), 4),
                    "macd_signal": round(indicators.get("macd_signal", 0), 4),
                    "bollinger_position": round(indicators.get("price_position", 50), 2),
                    "stochastic_k": round(indicators.get("stoch_k", 50), 2),
                    "williams_r": round(indicators.get("williams_r", -50), 2),
                    "atr": round(indicators.get("atr", 0), 2),
                    "volatility_pct": round(indicators.get("volatility_pct", 0), 2),
                    "momentum_score": round(indicators.get("momentum_score", 0), 2)
                },
                "strategy_analysis": {
                    "momentum": momentum_analysis,
                    "mean_reversion": mean_reversion_analysis,
                    "breakout": breakout_analysis
                },
                "support_resistance": {
                    "pivot_point": round(indicators.get("pivot_point", current_price), 2),
                    "resistance_1": round(indicators.get("resistance_1", current_price * 1.02), 2),
                    "support_1": round(indicators.get("support_1", current_price * 0.98), 2),
                    "bb_upper": round(indicators.get("bb_upper", current_price * 1.02), 2),
                    "bb_lower": round(indicators.get("bb_lower", current_price * 0.98), 2)
                },
                "analysis_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    @router.post("/api/v1/enhanced/multi-timeframe")
    def multi_timeframe_analysis(tradingSymbol: str):
        """Multi-timeframe technical analysis"""
        try:
            analysis = technical_analyzer.multi_timeframe_analysis(
                tradingSymbol, 
                get_chart_data_func
            )
            return analysis
            
        except Exception as e:
            return {"error": str(e)}
    
    @router.get("/api/v1/enhanced/market-scanner")
    def enhanced_market_scanner(
        strategy: str = "momentum",  # momentum, mean_reversion, breakout, all
        minConfidence: int = 70,
        maxResults: int = 30,
        chartPeriod: str = "I",
        timeInterval: int = 15
    ):
        """Advanced market scanner with strategy-specific filtering"""
        try:
            # Get trading statistics
            trading_stats = nse_client.get_stocks_traded()
            
            if "total" not in trading_stats or "data" not in trading_stats["total"]:
                return {"error": "Invalid trading statistics data"}
            
            stock_data = trading_stats["total"]["data"]
            
            # Filter for active stocks
            df = pd.DataFrame(stock_data)
            df_filtered = df[
                (df['symbol'].notna()) & 
                (df['pchange'].abs() >= 1.0) &  # Higher threshold for enhanced analysis
                (df['totalTradedVolume'] > 50)
            ].copy()
            
            # Sort by volume and volatility
            df_filtered['volatility_score'] = df_filtered['pchange'].abs() * np.log(df_filtered['totalTradedVolume'] + 1)
            df_filtered = df_filtered.sort_values('volatility_score', ascending=False)
            
            selected_stocks = df_filtered.head(min(maxResults * 2, 60)).to_dict('records')
            
            results = []
            
            # Parallel analysis
            def analyze_stock(stock):
                try:
                    symbol = stock['symbol']
                    chart_data = get_chart_data_func(symbol, chartPeriod, timeInterval)
                    
                    if chart_data.get("s") != "ok":
                        return None
                    
                    indicators = technical_analyzer.calculate_advanced_indicators(chart_data)
                    if not indicators:
                        return None
                    
                    # Run requested strategy
                    if strategy == "momentum" or strategy == "all":
                        momentum = technical_analyzer.momentum_strategy(indicators, chart_data)
                        if momentum.get("confidence", 0) >= minConfidence:
                            return {
                                "symbol": f"{symbol}-EQ",
                                "strategy": "momentum",
                                "decision": momentum["decision"],
                                "confidence": momentum["confidence"],
                                "score": momentum["score"],
                                "signals": momentum["signals"][:2],
                                "current_price": chart_data['c'][-1],
                                "price_change_pct": stock['pchange'],
                                "volume": stock['totalTradedVolume'],
                                "rsi": round(indicators.get("rsi", 50), 1),
                                "key_levels": momentum.get("key_levels", {})
                            }
                    
                    if strategy == "mean_reversion" or strategy == "all":
                        mean_rev = technical_analyzer.mean_reversion_strategy(indicators, chart_data)
                        if mean_rev.get("confidence", 0) >= minConfidence:
                            return {
                                "symbol": f"{symbol}-EQ",
                                "strategy": "mean_reversion",
                                "decision": mean_rev["decision"],
                                "confidence": mean_rev["confidence"],
                                "score": mean_rev["score"],
                                "signals": mean_rev["signals"][:2],
                                "current_price": chart_data['c'][-1],
                                "price_change_pct": stock['pchange'],
                                "volume": stock['totalTradedVolume'],
                                "bb_position": round(mean_rev.get("bb_position", 50), 1),
                                "mean_price": round(mean_rev.get("mean_price", chart_data['c'][-1]), 2)
                            }
                    
                    if strategy == "breakout" or strategy == "all":
                        breakout = technical_analyzer.breakout_strategy(indicators, chart_data)
                        if breakout.get("confidence", 0) >= minConfidence:
                            return {
                                "symbol": f"{symbol}-EQ",
                                "strategy": "breakout",
                                "decision": breakout["decision"],
                                "confidence": breakout["confidence"],
                                "score": breakout["score"],
                                "signals": breakout["signals"][:2],
                                "current_price": chart_data['c'][-1],
                                "price_change_pct": stock['pchange'],
                                "volume": stock['totalTradedVolume'],
                                "volume_ratio": round(breakout.get("volume_ratio", 1), 2),
                                "breakout_levels": breakout.get("breakout_levels", {})
                            }
                    
                    return None
                    
                except Exception:
                    return None
            
            # Process in parallel
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(analyze_stock, stock) for stock in selected_stocks]
                
                for future in futures:
                    try:
                        result = future.result(timeout=3)
                        if result:
                            results.append(result)
                    except:
                        continue
            
            # Sort by confidence
            results.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Categorize by decision
            buy_signals = [r for r in results if "BUY" in r["decision"]]
            sell_signals = [r for r in results if "SELL" in r["decision"]]
            
            return {
                "strategy_filter": strategy,
                "total_scanned": len(selected_stocks),
                "signals_found": len(results),
                "min_confidence": minConfidence,
                "chart_period": chartPeriod,
                "time_interval": timeInterval,
                "buy_signals": buy_signals[:maxResults//2],
                "sell_signals": sell_signals[:maxResults//2],
                "summary": {
                    "strong_buy": len([r for r in results if r["decision"] == "STRONG_BUY"]),
                    "buy": len([r for r in results if r["decision"] == "BUY"]),
                    "strong_sell": len([r for r in results if r["decision"] == "STRONG_SELL"]),
                    "sell": len([r for r in results if r["decision"] == "SELL"]),
                    "avg_confidence": round(np.mean([r["confidence"] for r in results]), 1) if results else 0
                },
                "scan_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    @router.post("/api/v1/enhanced/pattern-recognition")
    def pattern_recognition_analysis(
        tradingSymbol: str,
        chartPeriod: str = "I",
        timeInterval: int = 15
    ):
        """Advanced pattern recognition and candlestick analysis"""
        try:
            chart_data = get_chart_data_func(tradingSymbol, chartPeriod, timeInterval)
            
            if chart_data.get("s") != "ok":
                return {"error": "Unable to fetch chart data"}
            
            # Convert to numpy arrays
            open_prices = np.array(chart_data['o'], dtype=float)
            high_prices = np.array(chart_data['h'], dtype=float)
            low_prices = np.array(chart_data['l'], dtype=float)
            close_prices = np.array(chart_data['c'], dtype=float)
            
            if len(close_prices) < 10:
                return {"error": "Insufficient data for pattern analysis"}
            
            patterns = {}
            
            # Candlestick patterns (using talib)
            try:
                import talib
                
                # Bullish patterns
                patterns['hammer'] = bool(talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)[-1])
                patterns['doji'] = bool(talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)[-1])
                patterns['engulfing_bullish'] = bool(talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)[-1] > 0)
                patterns['morning_star'] = bool(talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)[-1])
                
                # Bearish patterns
                patterns['hanging_man'] = bool(talib.CDLHANGINGMAN(open_prices, high_prices, low_prices, close_prices)[-1])
                patterns['engulfing_bearish'] = bool(talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)[-1] < 0)
                patterns['evening_star'] = bool(talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)[-1])
                patterns['shooting_star'] = bool(talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)[-1])
                
            except ImportError:
                # Manual pattern detection if talib not available
                patterns = self._manual_pattern_detection(open_prices, high_prices, low_prices, close_prices)
            
            # Price action analysis
            current_price = close_prices[-1]
            prev_close = close_prices[-2]
            current_high = high_prices[-1]
            current_low = low_prices[-1]
            current_open = open_prices[-1]
            
            # Body and shadow analysis
            body_size = abs(current_price - current_open)
            upper_shadow = current_high - max(current_price, current_open)
            lower_shadow = min(current_price, current_open) - current_low
            total_range = current_high - current_low
            
            price_action = {
                "body_pct": (body_size / total_range * 100) if total_range > 0 else 0,
                "upper_shadow_pct": (upper_shadow / total_range * 100) if total_range > 0 else 0,
                "lower_shadow_pct": (lower_shadow / total_range * 100) if total_range > 0 else 0,
                "candle_type": "bullish" if current_price > current_open else "bearish",
                "gap": "up" if current_open > prev_close * 1.002 else "down" if current_open < prev_close * 0.998 else "none"
            }
            
            # Pattern significance
            bullish_patterns = sum([patterns.get('hammer', False), patterns.get('engulfing_bullish', False), 
                                  patterns.get('morning_star', False)])
            bearish_patterns = sum([patterns.get('hanging_man', False), patterns.get('engulfing_bearish', False),
                                  patterns.get('evening_star', False), patterns.get('shooting_star', False)])
            
            # Overall signal
            if bullish_patterns >= 2:
                signal = "STRONG_BULLISH"
                confidence = 85
            elif bullish_patterns >= 1:
                signal = "BULLISH"
                confidence = 70
            elif bearish_patterns >= 2:
                signal = "STRONG_BEARISH"
                confidence = 85
            elif bearish_patterns >= 1:
                signal = "BEARISH"
                confidence = 70
            else:
                signal = "NEUTRAL"
                confidence = 50
            
            return {
                "symbol": tradingSymbol,
                "current_price": current_price,
                "chart_period": chartPeriod,
                "time_interval": timeInterval,
                "candlestick_patterns": patterns,
                "price_action": price_action,
                "pattern_signal": signal,
                "confidence": confidence,
                "pattern_summary": {
                    "bullish_patterns_count": bullish_patterns,
                    "bearish_patterns_count": bearish_patterns,
                    "neutral_patterns": patterns.get('doji', False)
                },
                "analysis_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    @router.get("/api/v1/enhanced/sector-analysis")
    def sector_momentum_analysis(
        chartPeriod: str = "I",
        timeInterval: int = 15,
        minConfidence: int = 65
    ):
        """Sector-wise momentum and strength analysis"""
        try:
            # Get market data
            trading_stats = nse_client.get_stocks_traded()
            gainers = nse_client.get_gainers_data()
            losers = nse_client.get_losers_data()
            
            # Sector mapping (simplified)
            sector_keywords = {
                "Banking": ["BANK", "HDFC", "ICICI", "SBI", "AXIS", "KOTAK"],
                "IT": ["TCS", "INFY", "WIPRO", "HCL", "TECH"],
                "Pharma": ["SUN", "CIPLA", "REDDY", "LUPIN", "BIOCON"],
                "Auto": ["TATA", "BAJAJ", "HERO", "MARUTI", "MAHINDRA"],
                "Energy": ["RELIANCE", "ONGC", "IOC", "BPCL", "GAIL"],
                "Metals": ["TATA", "JSW", "HINDALCO", "VEDL", "STEEL"]
            }
            
            sector_analysis = {}
            
            for sector, keywords in sector_keywords.items():
                sector_stocks = []
                
                # Find stocks in sector
                if "total" in trading_stats and "data" in trading_stats["total"]:
                    for stock in trading_stats["total"]["data"]:
                        symbol = stock.get("symbol", "")
                        if any(keyword in symbol for keyword in keywords):
                            sector_stocks.append({
                                "symbol": symbol,
                                "price_change": stock.get("pchange", 0),
                                "volume": stock.get("totalTradedVolume", 0),
                                "market_cap": stock.get("totalMarketCap", 0)
                            })
                
                if sector_stocks:
                    # Calculate sector metrics
                    avg_change = np.mean([s["price_change"] for s in sector_stocks])
                    total_volume = sum([s["volume"] for s in sector_stocks])
                    positive_stocks = len([s for s in sector_stocks if s["price_change"] > 0])
                    
                    # Sector momentum score
                    momentum_score = (
                        avg_change * 0.4 +
                        (positive_stocks / len(sector_stocks) * 100 - 50) * 0.3 +
                        min(np.log(total_volume + 1) / 10, 10) * 0.3
                    )
                    
                    sector_analysis[sector] = {
                        "stock_count": len(sector_stocks),
                        "avg_price_change": round(avg_change, 2),
                        "positive_stocks_pct": round(positive_stocks / len(sector_stocks) * 100, 1),
                        "total_volume": total_volume,
                        "momentum_score": round(momentum_score, 2),
                        "top_performers": sorted(sector_stocks, key=lambda x: x["price_change"], reverse=True)[:3],
                        "sector_signal": "BULLISH" if momentum_score > 2 else "BEARISH" if momentum_score < -2 else "NEUTRAL"
                    }
            
            # Sort sectors by momentum
            sorted_sectors = sorted(sector_analysis.items(), key=lambda x: x[1]["momentum_score"], reverse=True)
            
            return {
                "sector_analysis": dict(sorted_sectors),
                "market_leaders": sorted_sectors[:3],
                "market_laggards": sorted_sectors[-3:],
                "overall_market_sentiment": {
                    "bullish_sectors": len([s for s in sector_analysis.values() if s["sector_signal"] == "BULLISH"]),
                    "bearish_sectors": len([s for s in sector_analysis.values() if s["sector_signal"] == "BEARISH"]),
                    "neutral_sectors": len([s for s in sector_analysis.values() if s["sector_signal"] == "NEUTRAL"])
                },
                "analysis_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    return router