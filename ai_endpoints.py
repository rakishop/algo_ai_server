from fastapi import FastAPI, Query, HTTPException
from typing import Optional, List, Dict, Any
from ml_analyzer import MLStockAnalyzer
from data_processor import DataProcessor
from nse_client import NSEClient
import json
import os

class AIEndpoints:
    def __init__(self, app: FastAPI, nse_client: NSEClient):
        self.app = app
        self.nse_client = nse_client
        self.processor = DataProcessor()
        self.ml_analyzer = MLStockAnalyzer()
        self.setup_ai_endpoints()
        self._train_models()
    
    def _train_models(self):
        """Train ML models with all JSON files"""
        try:
            json_files = [f for f in os.listdir('.') if f.endswith('.json')]
            all_data = {}
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        all_data[json_file] = data
                except Exception as e:
                    print(f"Error reading {json_file}: {e}")
            
            if all_data:
                df = self.ml_analyzer.prepare_dataset(all_data)
                if not df.empty:
                    self.ml_analyzer.train_models(df)
                    print(f"ML models trained successfully from {len(json_files)} JSON files!")
        except Exception as e:
            print(f"Error training models: {e}")
    
    def setup_ai_endpoints(self):
        """Setup AI-powered endpoints"""
        
        @self.app.get("/api/v1/ai/smart-picks")
        def get_smart_stock_picks(
            risk_level: str = Query("medium", regex="^(low|medium|high)$"),
            investment_amount: float = Query(100000, ge=1000),
            limit: int = Query(15, ge=1, le=50)
        ):
            """AI-powered stock recommendations based on risk profile"""
            try:
                # Get fresh data from NSE
                gainers_data = self.nse_client.get_gainers_data()
                volume_data = self.nse_client.get_volume_gainers()
                active_data = self.nse_client.get_most_active_securities()
                
                # Extract stocks directly from NSE response structure
                all_stocks = []
                
                # Extract from gainers data
                if isinstance(gainers_data, dict):
                    for key, section in gainers_data.items():
                        if isinstance(section, dict) and 'data' in section:
                            for stock in section['data']:
                                if isinstance(stock, dict) and stock.get('symbol'):
                                    all_stocks.append(stock)
                
                # Extract from volume data
                if isinstance(volume_data, dict) and 'data' in volume_data:
                    for stock in volume_data['data']:
                        if isinstance(stock, dict) and stock.get('symbol'):
                            all_stocks.append(stock)
                
                # Extract from active data
                if isinstance(active_data, dict) and 'data' in active_data:
                    for stock in active_data['data']:
                        if isinstance(stock, dict) and stock.get('symbol'):
                            all_stocks.append(stock)
                
                # Remove duplicates and add ML features
                unique_stocks = {}
                for stock in all_stocks:
                    symbol = stock.get("symbol")
                    if symbol and symbol not in unique_stocks and stock.get('ltp', 0) > 0:
                        features = self.ml_analyzer.extract_features(stock)
                        unique_stocks[symbol] = {**stock, **features}
                
                stocks_list = list(unique_stocks.values())
                
                # Apply ML analysis if model is trained
                if self.ml_analyzer.is_trained:
                    stocks_list = self.ml_analyzer.predict_clusters(stocks_list)
                
                # Filter based on risk level
                filtered_stocks = self._filter_by_risk_level(stocks_list, risk_level)
                
                # Calculate position sizes
                recommendations = self._calculate_position_sizes(filtered_stocks, investment_amount, limit)
                
                return {
                    "recommendations": recommendations,
                    "risk_level": risk_level,
                    "total_investment": investment_amount,
                    "diversification_count": len(recommendations),
                    "ai_insights": self.ml_analyzer.get_market_insights(stocks_list)
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/ai/anomaly-detection")
        def detect_market_anomalies(limit: int = Query(20, ge=1, le=100)):
            """Detect unusual market movements using AI"""
            try:
                # Get comprehensive market data
                gainers_data = self.nse_client.get_gainers_data()
                losers_data = self.nse_client.get_losers_data()
                volume_data = self.nse_client.get_volume_gainers()
                
                all_stocks = []
                all_stocks.extend(self.processor.extract_stock_data(gainers_data))
                all_stocks.extend(self.processor.extract_stock_data(losers_data))
                all_stocks.extend(self.processor.extract_stock_data(volume_data))
                
                # Process with ML
                processed_stocks = []
                for stock in all_stocks:
                    features = self.ml_analyzer.extract_features(stock)
                    processed_stocks.append({**stock, **features})
                
                if self.ml_analyzer.is_trained:
                    processed_stocks = self.ml_analyzer.predict_clusters(processed_stocks)
                    
                    # Filter anomalies
                    anomalies = [stock for stock in processed_stocks if stock.get('is_anomaly', False)]
                    
                    # Sort by significance
                    anomalies.sort(key=lambda x: abs(x.get('perChange', 0)), reverse=True)
                    
                    return {
                        "anomalies": anomalies[:limit],
                        "total_anomalies": len(anomalies),
                        "detection_method": "Isolation Forest ML Algorithm",
                        "significance_threshold": "Top anomalies by price change"
                    }
                else:
                    # Fallback to rule-based detection
                    anomalies = self._rule_based_anomaly_detection(processed_stocks)
                    return {
                        "anomalies": anomalies[:limit],
                        "total_anomalies": len(anomalies),
                        "detection_method": "Rule-based (ML model not trained)",
                        "note": "Train ML model for better anomaly detection"
                    }
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/ai/similar-stocks/{symbol}")
        def find_similar_stocks(
            symbol: str,
            limit: int = Query(10, ge=1, le=30)
        ):
            """Find stocks similar to the given symbol using AI clustering"""
            try:
                # Get market data
                gainers_data = self.nse_client.get_gainers_data()
                active_data = self.nse_client.get_most_active_securities()
                
                all_stocks = []
                all_stocks.extend(self.processor.extract_stock_data(gainers_data))
                all_stocks.extend(self.processor.extract_stock_data(active_data))
                
                # Process stocks
                processed_stocks = []
                for stock in all_stocks:
                    features = self.ml_analyzer.extract_features(stock)
                    processed_stocks.append({**stock, **features})
                
                if self.ml_analyzer.is_trained:
                    processed_stocks = self.ml_analyzer.predict_clusters(processed_stocks)
                    similar_stocks = self.ml_analyzer.find_similar_stocks(symbol.upper(), processed_stocks, limit)
                    
                    return {
                        "target_symbol": symbol.upper(),
                        "similar_stocks": similar_stocks,
                        "similarity_method": "ML Clustering (K-Means)",
                        "count": len(similar_stocks)
                    }
                else:
                    return {
                        "target_symbol": symbol.upper(),
                        "similar_stocks": [],
                        "similarity_method": "ML model not trained",
                        "note": "Train ML model for similarity analysis"
                    }
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/ai/market-sentiment")
        def analyze_market_sentiment():
            """Analyze overall market sentiment using AI"""
            try:
                # Get comprehensive data
                gainers_data = self.nse_client.get_gainers_data()
                losers_data = self.nse_client.get_losers_data()
                advance_data = self.nse_client.get_advance_decline()
                indices_data = self.nse_client.get_all_indices()
                
                # Process data
                gainers = self.processor.extract_stock_data(gainers_data)
                losers = self.processor.extract_stock_data(losers_data)
                
                # Calculate sentiment metrics
                sentiment_score = self._calculate_sentiment_score(gainers, losers, advance_data)
                
                # Get AI insights
                all_stocks = gainers + losers
                processed_stocks = []
                for stock in all_stocks:
                    features = self.ml_analyzer.extract_features(stock)
                    processed_stocks.append({**stock, **features})
                
                ai_insights = self.ml_analyzer.get_market_insights(processed_stocks)
                
                return {
                    "sentiment_score": sentiment_score,
                    "sentiment_label": self._get_sentiment_label(sentiment_score),
                    "market_breadth": advance_data,
                    "ai_insights": ai_insights,
                    "top_gainers_count": len(gainers),
                    "top_losers_count": len(losers),
                    "analysis_timestamp": self.processor.data_storage.get("sentiment", [{"timestamp": "N/A"}])[-1].get("timestamp", "N/A")
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/ai/portfolio-optimizer")
        def optimize_portfolio(
            symbols: str = Query(..., description="Comma-separated stock symbols"),
            investment_amount: float = Query(100000, ge=1000),
            risk_tolerance: str = Query("medium", regex="^(low|medium|high)$")
        ):
            """AI-powered portfolio optimization"""
            try:
                symbol_list = [s.strip().upper() for s in symbols.split(",")]
                
                # Get data for specified symbols
                gainers_data = self.nse_client.get_gainers_data()
                active_data = self.nse_client.get_most_active_securities()
                
                all_stocks = []
                all_stocks.extend(self.processor.extract_stock_data(gainers_data))
                all_stocks.extend(self.processor.extract_stock_data(active_data))
                
                # Filter for requested symbols
                portfolio_stocks = []
                for stock in all_stocks:
                    if stock.get("symbol") in symbol_list:
                        features = self.ml_analyzer.extract_features(stock)
                        portfolio_stocks.append({**stock, **features})
                
                if not portfolio_stocks:
                    return {
                        "error": "No data found for specified symbols",
                        "requested_symbols": symbol_list
                    }
                
                # Optimize allocation
                optimized_portfolio = self._optimize_allocation(portfolio_stocks, investment_amount, risk_tolerance)
                
                return {
                    "optimized_portfolio": optimized_portfolio,
                    "total_investment": investment_amount,
                    "risk_tolerance": risk_tolerance,
                    "diversification_score": len(optimized_portfolio),
                    "optimization_method": "Risk-adjusted allocation"
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/ai/momentum-analysis")
        def get_momentum_analysis(
            timeframe: str = Query("daily", regex="^(intraday|daily|weekly)$"),
            min_volume: int = Query(1000000, ge=100000),
            limit: int = Query(20, ge=1, le=50)
        ):
            """AI-powered momentum stock identification"""
            try:
                gainers_data = self.nse_client.get_gainers_data()
                volume_data = self.nse_client.get_volume_gainers()
                
                all_stocks = []
                all_stocks.extend(self.processor.extract_stock_data(gainers_data))
                all_stocks.extend(self.processor.extract_stock_data(volume_data))
                
                momentum_stocks = []
                for stock in all_stocks:
                    features = self.ml_analyzer.extract_features(stock)
                    price_change = abs(stock.get('perChange', 0))
                    volume = stock.get('trade_quantity', 0)
                    volatility = features.get('price_volatility', 0)
                    
                    momentum_score = (price_change * 0.4) + (min(volume/1000000, 10) * 0.3) + (volatility * 0.3)
                    momentum_score = min(momentum_score * 10, 100)
                    
                    if (volume >= min_volume and momentum_score > 60 and price_change > 2):
                        momentum_stocks.append({
                            **stock,
                            **features,
                            'momentum_score': momentum_score,
                            'trend_strength': "Strong Bullish" if price_change > 5 else "Moderate Bullish" if price_change > 2 else "Sideways"
                        })
                
                momentum_stocks.sort(key=lambda x: x['momentum_score'], reverse=True)
                
                return {
                    "momentum_stocks": momentum_stocks[:limit],
                    "timeframe": timeframe,
                    "min_volume_filter": min_volume,
                    "analysis_method": "AI Momentum Scoring",
                    "total_candidates": len(momentum_stocks)
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/ai/scalping-analysis")
        def get_scalping_analysis(
            volatility_threshold: float = Query(2.0, ge=1.0, le=10.0),
            volume_threshold: int = Query(5000000, ge=1000000),
            limit: int = Query(15, ge=1, le=30)
        ):
            """AI-powered scalping stock identification"""
            try:
                active_data = self.nse_client.get_most_active_securities()
                volume_data = self.nse_client.get_volume_gainers()
                
                all_stocks = []
                all_stocks.extend(self.processor.extract_stock_data(active_data))
                all_stocks.extend(self.processor.extract_stock_data(volume_data))
                
                scalping_stocks = []
                for stock in all_stocks:
                    features = self.ml_analyzer.extract_features(stock)
                    volume = stock.get('trade_quantity', 0)
                    volatility = features.get('price_volatility', 0)
                    price = stock.get('ltp', 0)
                    
                    liquidity_score = min(volume/1000000, 20) * 0.4
                    volatility_score = min(volatility, 10) * 0.4
                    price_score = min(price/100, 10) * 0.2
                    scalping_score = min((liquidity_score + volatility_score + price_score) * 5, 100)
                    
                    if (volume >= volume_threshold and volatility >= volatility_threshold and scalping_score > 70):
                        scalping_stocks.append({
                            **stock,
                            **features,
                            'scalping_score': scalping_score,
                            'liquidity_rating': "Excellent" if volume > 20000000 else "Good" if volume > 10000000 else "Fair"
                        })
                
                scalping_stocks.sort(key=lambda x: x['scalping_score'], reverse=True)
                
                return {
                    "scalping_opportunities": scalping_stocks[:limit],
                    "volatility_threshold": volatility_threshold,
                    "volume_threshold": volume_threshold,
                    "analysis_method": "AI Scalping Analysis",
                    "total_opportunities": len(scalping_stocks)
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/ai/options-strategies")
        def analyze_options_strategies(
            strategy_type: str = Query("all", regex="^(all|bullish|bearish|neutral)$"),
            limit: int = Query(25, ge=1, le=50)
        ):
            """AI-powered options trading opportunities"""
            try:
                active_stocks = self.nse_client.get_most_active_securities()
                underlying_stocks = self.processor.extract_stock_data(active_stocks)
                
                options_opportunities = []
                for stock in underlying_stocks:
                    features = self.ml_analyzer.extract_features(stock)
                    change = stock.get('perChange', 0)
                    volatility = features.get('price_volatility', 0)
                    
                    strategies = []
                    if strategy_type in ["all", "bullish"] and change > 1:
                        strategies.append({
                            "name": "Bull Call Spread",
                            "type": "Bullish",
                            "confidence_score": min(70 + change * 2, 95),
                            "risk_reward": 1.5,
                            "description": f"Buy ITM call, sell OTM call on {stock.get('symbol')}"
                        })
                    
                    if strategy_type in ["all", "bearish"] and change < -1:
                        strategies.append({
                            "name": "Bear Put Spread",
                            "type": "Bearish",
                            "confidence_score": min(70 + abs(change) * 2, 95),
                            "risk_reward": 1.4,
                            "description": f"Buy ITM put, sell OTM put on {stock.get('symbol')}"
                        })
                    
                    if strategy_type in ["all", "neutral"] and abs(change) < 2 and volatility > 3:
                        strategies.append({
                            "name": "Iron Condor",
                            "type": "Neutral",
                            "confidence_score": min(60 + volatility * 3, 90),
                            "risk_reward": 2.0,
                            "description": f"Sell OTM call/put spreads on {stock.get('symbol')}"
                        })
                    
                    for strategy in strategies:
                        if strategy['confidence_score'] > 65:
                            options_opportunities.append({
                                'underlying_symbol': stock.get('symbol'),
                                'underlying_price': stock.get('ltp', 0),
                                'strategy': strategy
                            })
                
                options_opportunities.sort(key=lambda x: x['strategy']['confidence_score'], reverse=True)
                
                return {
                    "options_opportunities": options_opportunities[:limit],
                    "strategy_filter": strategy_type,
                    "analysis_method": "AI Options Strategy Analysis",
                    "total_opportunities": len(options_opportunities)
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    def _filter_by_risk_level(self, stocks: List[Dict[str, Any]], risk_level: str) -> List[Dict[str, Any]]:
        """Filter stocks based on risk level"""
        if not stocks:
            return []
            
        if risk_level == "low":
            # Low risk: stable stocks with moderate changes
            return [s for s in stocks if abs(s.get('perChange', 0)) <= 5 and s.get('ltp', 0) > 0]
        elif risk_level == "medium":
            # Medium risk: moderate volatility - more inclusive
            return [s for s in stocks if s.get('ltp', 0) > 0]
        else:  # high risk
            # High risk: high volatility stocks
            return [s for s in stocks if abs(s.get('perChange', 0)) > 2 and s.get('ltp', 0) > 0]
    
    def _calculate_position_sizes(self, stocks: List[Dict[str, Any]], total_amount: float, limit: int) -> List[Dict[str, Any]]:
        """Calculate position sizes for portfolio"""
        if not stocks:
            return []
            
        selected_stocks = stocks[:limit]
        
        # Simple equal weight allocation (can be enhanced with more sophisticated methods)
        allocation_per_stock = total_amount / len(selected_stocks)
        
        recommendations = []
        for stock in selected_stocks:
            # Get effective price from multiple possible fields
            price = (stock.get('ltp', 0) or 
                    stock.get('open_price', 0) or 
                    stock.get('open', 0) or 
                    stock.get('close', 0) or 
                    stock.get('lastPrice', 0) or 
                    100)  # Default price if none available
            
            if price > 0:
                quantity = int(allocation_per_stock / price)
                actual_investment = quantity * price
                
                recommendations.append({
                    "symbol": stock.get("symbol"),
                    "price": price,
                    "quantity": quantity,
                    "investment_amount": actual_investment,
                    "weight_percentage": (actual_investment / total_amount) * 100,
                    "expected_return": stock.get("perChange", 0),
                    "risk_score": stock.get("price_volatility", 0)
                })
        
        return recommendations
    
    def _rule_based_anomaly_detection(self, stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rule-based anomaly detection fallback"""
        anomalies = []
        
        for stock in stocks:
            pchange = abs(stock.get('perChange', 0))
            volatility = stock.get('price_volatility', 0)
            volume = stock.get('trade_quantity', 0)
            
            # Define anomaly conditions
            is_anomaly = (
                pchange > 10 or  # Extreme price change
                volatility > 8 or  # High volatility
                volume > 50000000  # Unusually high volume
            )
            
            if is_anomaly:
                stock['anomaly_reason'] = []
                if pchange > 10:
                    stock['anomaly_reason'].append(f"Extreme price change: {pchange}%")
                if volatility > 8:
                    stock['anomaly_reason'].append(f"High volatility: {volatility}%")
                if volume > 50000000:
                    stock['anomaly_reason'].append(f"High volume: {volume}")
                
                anomalies.append(stock)
        
        return sorted(anomalies, key=lambda x: abs(x.get('perChange', 0)), reverse=True)
    
    def _calculate_sentiment_score(self, gainers: List, losers: List, advance_data: Dict) -> float:
        """Calculate market sentiment score"""
        try:
            # Extract advance/decline data
            advance_count = 0
            decline_count = 0
            
            if isinstance(advance_data, dict) and 'advance' in advance_data:
                advance_info = advance_data['advance']
                if isinstance(advance_info, dict) and 'count' in advance_info:
                    count_data = advance_info['count']
                    advance_count = count_data.get('Advances', 0)
                    decline_count = count_data.get('Declines', 0)
            
            # Calculate sentiment based on multiple factors
            gainer_strength = sum(abs(g.get('perChange', 0)) for g in gainers[:10])
            loser_strength = sum(abs(l.get('perChange', 0)) for l in losers[:10])
            
            # Normalize to -100 to +100 scale
            if advance_count + decline_count > 0:
                breadth_sentiment = ((advance_count - decline_count) / (advance_count + decline_count)) * 50
            else:
                breadth_sentiment = 0
                
            strength_sentiment = (gainer_strength - loser_strength) / max(gainer_strength + loser_strength, 1) * 50
            
            return round(breadth_sentiment + strength_sentiment, 2)
            
        except Exception:
            return 0.0
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score > 30:
            return "Very Bullish"
        elif score > 10:
            return "Bullish"
        elif score > -10:
            return "Neutral"
        elif score > -30:
            return "Bearish"
        else:
            return "Very Bearish"
    
    def _optimize_allocation(self, stocks: List[Dict[str, Any]], total_amount: float, risk_tolerance: str) -> List[Dict[str, Any]]:
        """Optimize portfolio allocation"""
        if not stocks:
            return []
        
        # Risk-based weighting
        risk_weights = {"low": 0.5, "medium": 1.0, "high": 1.5}
        risk_multiplier = risk_weights.get(risk_tolerance, 1.0)
        
        # Calculate weights based on performance and risk
        for stock in stocks:
            performance_score = abs(stock.get('perChange', 0))
            volatility_penalty = stock.get('price_volatility', 0) * 0.1
            stock['allocation_score'] = (performance_score - volatility_penalty) * risk_multiplier
        
        # Normalize weights
        total_score = sum(max(s['allocation_score'], 0.1) for s in stocks)
        
        portfolio = []
        for stock in stocks:
            weight = max(stock['allocation_score'], 0.1) / total_score
            allocation = total_amount * weight
            price = stock.get('ltp', 0)
            
            if price > 0:
                quantity = int(allocation / price)
                actual_investment = quantity * price
                
                portfolio.append({
                    "symbol": stock.get("symbol"),
                    "allocation_percentage": round(weight * 100, 2),
                    "investment_amount": actual_investment,
                    "quantity": quantity,
                    "price": price,
                    "risk_score": stock.get('price_volatility', 0)
                })
        
        return sorted(portfolio, key=lambda x: x['allocation_percentage'], reverse=True)
    
    def _calculate_momentum_score(self, stock: Dict[str, Any], features: Dict[str, Any]) -> float:
        """Calculate momentum score for a stock"""
        price_change = abs(stock.get('perChange', 0))
        volume = stock.get('trade_quantity', 0)
        volatility = features.get('price_volatility', 0)
        
        momentum_score = (price_change * 0.4) + (min(volume/1000000, 10) * 0.3) + (volatility * 0.3)
        return min(momentum_score * 10, 100)
    
    def _get_trend_strength(self, stock: Dict[str, Any]) -> str:
        """Assess trend strength"""
        change = stock.get('perChange', 0)
        if change > 5:
            return "Strong Bullish"
        elif change > 2:
            return "Moderate Bullish"
        elif change < -5:
            return "Strong Bearish"
        elif change < -2:
            return "Moderate Bearish"
        return "Sideways"
    
    def _assess_breakout_potential(self, stock: Dict[str, Any]) -> str:
        """Assess breakout potential"""
        volume = stock.get('trade_quantity', 0)
        change = stock.get('perChange', 0)
        
        if volume > 10000000 and change > 3:
            return "High"
        elif volume > 5000000 and change > 1.5:
            return "Medium"
        return "Low"
    
    def _calculate_scalping_score(self, stock: Dict[str, Any], features: Dict[str, Any]) -> float:
        """Calculate scalping suitability score"""
        volume = stock.get('trade_quantity', 0)
        volatility = features.get('price_volatility', 0)
        price = stock.get('ltp', 0)
        
        liquidity_score = min(volume/1000000, 20) * 0.4
        volatility_score = min(volatility, 10) * 0.4
        price_score = min(price/100, 10) * 0.2
        
        return min((liquidity_score + volatility_score + price_score) * 5, 100)
    
    def _assess_liquidity(self, stock: Dict[str, Any]) -> str:
        """Assess stock liquidity for scalping"""
        volume = stock.get('trade_quantity', 0)
        if volume > 20000000:
            return "Excellent"
        elif volume > 10000000:
            return "Good"
        elif volume > 5000000:
            return "Fair"
        return "Poor"
    
    def _calculate_intraday_range(self, stock: Dict[str, Any]) -> Dict[str, float]:
        """Calculate intraday trading range"""
        high = stock.get('dayHigh', stock.get('ltp', 0))
        low = stock.get('dayLow', stock.get('ltp', 0))
        current = stock.get('ltp', 0)
        
        if high > low and current > 0:
            range_percent = ((high - low) / current) * 100
            return {
                "range_percent": round(range_percent, 2),
                "high": high,
                "low": low,
                "current_position": round(((current - low) / (high - low)) * 100, 2) if high > low else 50
            }
        return {"range_percent": 0, "high": 0, "low": 0, "current_position": 50}
    
    def _identify_entry_zones(self, stock: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify optimal entry zones for scalping"""
        current_price = stock.get('ltp', 0)
        high = stock.get('dayHigh', current_price)
        low = stock.get('dayLow', current_price)
        
        zones = []
        if current_price > 0:
            # Support zone
            support = low + (current_price - low) * 0.2
            zones.append({
                "type": "Support",
                "price": round(support, 2),
                "action": "Buy",
                "confidence": "Medium"
            })
            
            # Resistance zone
            resistance = current_price + (high - current_price) * 0.8
            zones.append({
                "type": "Resistance",
                "price": round(resistance, 2),
                "action": "Sell",
                "confidence": "Medium"
            })
        
        return zones
    
    def _generate_options_strategies(self, stock: Dict[str, Any], features: Dict[str, Any], strategy_type: str) -> List[Dict[str, Any]]:
        """Generate options trading strategies"""
        strategies = []
        price = stock.get('ltp', 0)
        change = stock.get('perChange', 0)
        volatility = features.get('price_volatility', 0)
        
        if strategy_type in ["all", "bullish"] and change > 1:
            strategies.append({
                "name": "Bull Call Spread",
                "type": "Bullish",
                "confidence_score": min(70 + change * 2, 95),
                "risk_reward": 1.5,
                "pop": 65,
                "description": f"Buy ITM call, sell OTM call on {stock.get('symbol')}"
            })
        
        if strategy_type in ["all", "bearish"] and change < -1:
            strategies.append({
                "name": "Bear Put Spread",
                "type": "Bearish",
                "confidence_score": min(70 + abs(change) * 2, 95),
                "risk_reward": 1.4,
                "pop": 62,
                "description": f"Buy ITM put, sell OTM put on {stock.get('symbol')}"
            })
        
        if strategy_type in ["all", "neutral"] and abs(change) < 2 and volatility > 3:
            strategies.append({
                "name": "Iron Condor",
                "type": "Neutral",
                "confidence_score": min(60 + volatility * 3, 90),
                "risk_reward": 2.0,
                "pop": 70,
                "description": f"Sell OTM call/put spreads on {stock.get('symbol')}"
            })
        
        return strategies
    
    def _determine_market_outlook(self, stock: Dict[str, Any], features: Dict[str, Any]) -> str:
        """Determine market outlook for options trading"""
        change = stock.get('perChange', 0)
        volatility = features.get('price_volatility', 0)
        
        if change > 3 and volatility > 4:
            return "Strong Bullish with High Volatility"
        elif change > 1:
            return "Moderately Bullish"
        elif change < -3 and volatility > 4:
            return "Strong Bearish with High Volatility"
        elif change < -1:
            return "Moderately Bearish"
        elif volatility > 5:
            return "High Volatility - Range Bound"
        return "Low Volatility - Sideways"