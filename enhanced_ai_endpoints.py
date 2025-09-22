from fastapi import FastAPI, Query, HTTPException
from typing import List, Dict, Any
from ml_analyzer import MLStockAnalyzer
from data_processor import DataProcessor
from nse_client import NSEClient

class EnhancedAIEndpoints:
    def __init__(self, app: FastAPI, nse_client: NSEClient):
        self.app = app
        self.nse_client = nse_client
        self.processor = DataProcessor()
        self.ml_analyzer = MLStockAnalyzer()
        self.setup_enhanced_endpoints()
    
    def setup_enhanced_endpoints(self):
        @self.app.get("/api/v1/ai/momentum-stocks")
        def get_momentum_stocks(
            timeframe: str = Query("daily", regex="^(intraday|daily|weekly)$"),
            min_volume: int = Query(1000000, ge=100000),
            limit: int = Query(20, ge=1, le=50)
        ):
            try:
                gainers_data = self.nse_client.get_gainers_data()
                volume_data = self.nse_client.get_volume_gainers()
                
                all_stocks = []
                all_stocks.extend(self.processor.extract_stock_data(gainers_data))
                all_stocks.extend(self.processor.extract_stock_data(volume_data))
                
                momentum_stocks = []
                for stock in all_stocks:
                    features = self.ml_analyzer.extract_features(stock)
                    momentum_score = self._calculate_momentum_score(stock, features)
                    
                    if (stock.get('trade_quantity', 0) >= min_volume/2 and 
                        momentum_score > 30 and 
                        stock.get('perChange', 0) > 0.5):
                        
                        momentum_stocks.append({
                            **stock,
                            **features,
                            'momentum_score': momentum_score,
                            'trend_strength': self._get_trend_strength(stock),
                            'breakout_potential': self._assess_breakout_potential(stock)
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
        
        @self.app.get("/api/v1/ai/scalping-stocks")
        def get_scalping_opportunities(
            volatility_threshold: float = Query(2.0, ge=1.0, le=10.0),
            volume_threshold: int = Query(5000000, ge=1000000),
            limit: int = Query(15, ge=1, le=30)
        ):
            try:
                active_data = self.nse_client.get_most_active_securities()
                volume_data = self.nse_client.get_volume_gainers()
                
                all_stocks = []
                all_stocks.extend(self.processor.extract_stock_data(active_data))
                all_stocks.extend(self.processor.extract_stock_data(volume_data))
                
                scalping_stocks = []
                for stock in all_stocks:
                    features = self.ml_analyzer.extract_features(stock)
                    scalping_score = self._calculate_scalping_score(stock, features)
                    
                    if (stock.get('trade_quantity', 0) >= volume_threshold and 
                        features.get('price_volatility', 0) >= volatility_threshold and
                        scalping_score > 70):
                        
                        scalping_stocks.append({
                            **stock,
                            **features,
                            'scalping_score': scalping_score,
                            'liquidity_rating': self._assess_liquidity(stock)
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
        
        @self.app.get("/api/v1/ai/options-analysis")
        def analyze_options_opportunities(
            strategy_type: str = Query("all", regex="^(all|bullish|bearish|neutral)$"),
            limit: int = Query(25, ge=1, le=50)
        ):
            try:
                active_stocks = self.nse_client.get_most_active_securities()
                underlying_stocks = self.processor.extract_stock_data(active_stocks)
                
                options_opportunities = []
                for stock in underlying_stocks:
                    features = self.ml_analyzer.extract_features(stock)
                    strategies = self._generate_options_strategies(stock, features, strategy_type)
                    
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
    
    def _calculate_momentum_score(self, stock: Dict[str, Any], features: Dict[str, Any]) -> float:
        price_change = abs(stock.get('perChange', 0))
        volume = stock.get('trade_quantity', 0)
        volatility = features.get('price_volatility', 0)
        
        momentum_score = (price_change * 0.4) + (min(volume/1000000, 10) * 0.3) + (volatility * 0.3)
        return min(momentum_score * 10, 100)
    
    def _get_trend_strength(self, stock: Dict[str, Any]) -> str:
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
        volume = stock.get('trade_quantity', 0)
        change = stock.get('perChange', 0)
        
        if volume > 10000000 and change > 3:
            return "High"
        elif volume > 5000000 and change > 1.5:
            return "Medium"
        return "Low"
    
    def _calculate_scalping_score(self, stock: Dict[str, Any], features: Dict[str, Any]) -> float:
        volume = stock.get('trade_quantity', 0)
        volatility = features.get('price_volatility', 0)
        price = stock.get('ltp', 0)
        
        liquidity_score = min(volume/1000000, 20) * 0.4
        volatility_score = min(volatility, 10) * 0.4
        price_score = min(price/100, 10) * 0.2
        
        return min((liquidity_score + volatility_score + price_score) * 5, 100)
    
    def _assess_liquidity(self, stock: Dict[str, Any]) -> str:
        volume = stock.get('trade_quantity', 0)
        if volume > 20000000:
            return "Excellent"
        elif volume > 10000000:
            return "Good"
        elif volume > 5000000:
            return "Fair"
        return "Poor"
    
    def _generate_options_strategies(self, stock: Dict[str, Any], features: Dict[str, Any], strategy_type: str) -> List[Dict[str, Any]]:
        strategies = []
        change = stock.get('perChange', 0)
        volatility = features.get('price_volatility', 0)
        
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
        
        return strategies