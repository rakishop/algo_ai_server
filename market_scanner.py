from typing import Dict, List, Optional
from nse_client import NSEClient
from ml_analyzer import MLStockAnalyzer
from advanced_indicators import AdvancedTechnicalIndicators
import pandas as pd
from datetime import datetime

class MarketScanner:
    """Advanced market scanner for finding trading opportunities"""
    
    def __init__(self):
        self.nse_client = NSEClient()
        self.ml_analyzer = MLStockAnalyzer()
        self.indicators = AdvancedTechnicalIndicators()
    
    def scan_breakout_stocks(self, volume_threshold: float = 1.5, price_change_threshold: float = 3.0) -> Dict:
        """Scan for breakout stocks with high volume and price movement"""
        try:
            # Get market data
            active_data = self.nse_client.get_most_active_securities()
            volume_data = self.nse_client.get_volume_gainers()
            gainers_data = self.nse_client.get_gainers_data()
            
            breakout_stocks = []
            all_stocks = []
            
            # Combine all data sources
            for data_source in [active_data, volume_data, gainers_data]:
                if "data" in data_source:
                    all_stocks.extend(data_source["data"])
            
            # Remove duplicates
            unique_stocks = {}
            for stock in all_stocks:
                symbol = stock.get('symbol')
                if symbol and symbol not in unique_stocks:
                    unique_stocks[symbol] = stock
            
            for stock in unique_stocks.values():
                features = self.ml_analyzer.extract_features(stock)
                
                # Breakout criteria
                price_change = abs(features.get('perChange', 0))
                volume_ratio = features.get('volume', 0) / max(features.get('turnover', 1) / max(features.get('ltp', 1), 1), 1)
                
                if price_change >= price_change_threshold and volume_ratio >= volume_threshold:
                    breakout_score = min((price_change / 10) * 40 + (volume_ratio / 3) * 60, 100)
                    
                    breakout_stocks.append({
                        **stock,
                        **features,
                        'breakout_score': breakout_score,
                        'volume_ratio': volume_ratio,
                        'breakout_type': 'BULLISH' if features.get('perChange', 0) > 0 else 'BEARISH'
                    })
            
            # Sort by breakout score
            breakout_stocks.sort(key=lambda x: x['breakout_score'], reverse=True)
            
            return {
                "scan_type": "Breakout Scanner",
                "total_scanned": len(unique_stocks),
                "breakout_opportunities": breakout_stocks[:15],
                "criteria": {
                    "min_volume_ratio": volume_threshold,
                    "min_price_change": price_change_threshold
                },
                "scan_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def scan_momentum_stocks(self, rsi_range: tuple = (30, 70), min_volume: int = 1000000) -> Dict:
        """Scan for momentum stocks with good RSI levels"""
        try:
            active_data = self.nse_client.get_most_active_securities()
            gainers_data = self.nse_client.get_gainers_data()
            
            momentum_stocks = []
            all_stocks = []
            
            for data_source in [active_data, gainers_data]:
                if "data" in data_source:
                    all_stocks.extend(data_source["data"])
            
            unique_stocks = {}
            for stock in all_stocks:
                symbol = stock.get('symbol')
                if symbol and symbol not in unique_stocks:
                    unique_stocks[symbol] = stock
            
            for stock in unique_stocks.values():
                features = self.ml_analyzer.extract_features(stock)
                volume = features.get('trade_quantity', 0)
                
                if volume >= min_volume:
                    # Simulate RSI calculation (simplified)
                    price_change = features.get('perChange', 0)
                    rsi_estimate = 50 + (price_change * 2)  # Simplified RSI estimation
                    rsi_estimate = max(0, min(100, rsi_estimate))
                    
                    if rsi_range[0] <= rsi_estimate <= rsi_range[1]:
                        momentum_score = min(
                            (volume / 1000000) * 30 + 
                            abs(price_change) * 10 + 
                            (70 - abs(rsi_estimate - 50)) * 0.8, 
                            100
                        )
                        
                        momentum_stocks.append({
                            **stock,
                            **features,
                            'momentum_score': momentum_score,
                            'estimated_rsi': rsi_estimate,
                            'momentum_type': 'STRONG' if momentum_score > 70 else 'MODERATE'
                        })
            
            momentum_stocks.sort(key=lambda x: x['momentum_score'], reverse=True)
            
            return {
                "scan_type": "Momentum Scanner",
                "total_scanned": len(unique_stocks),
                "momentum_opportunities": momentum_stocks[:15],
                "criteria": {
                    "rsi_range": rsi_range,
                    "min_volume": min_volume
                },
                "scan_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def scan_reversal_candidates(self, oversold_threshold: float = -5.0, volume_spike: float = 2.0) -> Dict:
        """Scan for potential reversal candidates"""
        try:
            losers_data = self.nse_client.get_losers_data()
            volume_data = self.nse_client.get_volume_gainers()
            active_data = self.nse_client.get_most_active_securities()
            
            reversal_stocks = []
            
            # Extract all losers from different categories
            all_losers = []
            if isinstance(losers_data, dict):
                for category in ['NIFTY', 'BANKNIFTY', 'NIFTYNEXT50', 'FOSec', 'allSec']:
                    if category in losers_data and 'data' in losers_data[category]:
                        all_losers.extend(losers_data[category]['data'])
            
            # Focus on losers with high volume
            if all_losers:
                for stock in all_losers:
                    features = self.ml_analyzer.extract_features(stock)
                    price_change = features.get('perChange', 0)
                    
                    if price_change <= oversold_threshold:
                        # Check if it's in volume gainers or active securities (potential reversal signal)
                        symbol = stock.get('symbol')
                        volume_spike_detected = False
                        high_activity = False
                        
                        # Check volume gainers
                        if "data" in volume_data:
                            for vol_stock in volume_data["data"]:
                                if vol_stock.get('symbol') == symbol:
                                    volume_spike_detected = True
                                    break
                        
                        # Also check active securities for high trading activity
                        if "data" in active_data:
                            for active_stock in active_data["data"]:
                                if active_stock.get('symbol') == symbol:
                                    high_activity = True
                                    break
                        
                        # Accept if either volume spike or high activity is detected
                        if volume_spike_detected or high_activity:
                            reversal_score = min(
                                abs(price_change) * 10 + 
                                (features.get('trade_quantity', 0) / 1000000) * 20 + 
                                30 + (10 if volume_spike_detected else 0) + (5 if high_activity else 0),
                                100
                            )
                            
                            reversal_stocks.append({
                                **stock,
                                **features,
                                'reversal_score': reversal_score,
                                'reversal_signal': 'BULLISH_REVERSAL',
                                'volume_spike': volume_spike_detected,
                                'high_activity': high_activity
                            })
            
            # If no strict reversals found, include some moderate losers with good volume
            if len(reversal_stocks) < 3:
                moderate_threshold = max(oversold_threshold / 2, -2.5)  # More lenient threshold
                
                if all_losers:
                    for stock in all_losers:
                        if stock.get('symbol') not in [s.get('symbol') for s in reversal_stocks]:
                            features = self.ml_analyzer.extract_features(stock)
                            price_change = features.get('perChange', 0)
                            volume = features.get('trade_quantity', 0)
                            
                            if moderate_threshold <= price_change <= oversold_threshold and volume > 500000:
                                reversal_score = min(
                                    abs(price_change) * 8 + 
                                    (volume / 1000000) * 15 + 
                                    20,  # Lower base score for moderate candidates
                                    100
                                )
                                
                                reversal_stocks.append({
                                    **stock,
                                    **features,
                                    'reversal_score': reversal_score,
                                    'reversal_signal': 'MODERATE_REVERSAL',
                                    'volume_spike': False,
                                    'high_activity': False
                                })
            
            reversal_stocks.sort(key=lambda x: x['reversal_score'], reverse=True)
            
            return {
                "scan_type": "Reversal Scanner",
                "reversal_candidates": reversal_stocks[:10],
                "criteria": {
                    "max_price_change": oversold_threshold,
                    "volume_spike_required": volume_spike
                },
                "scan_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def scan_gap_opportunities(self, min_gap_percent: float = 2.0) -> Dict:
        """Scan for gap up/down opportunities"""
        try:
            active_data = self.nse_client.get_most_active_securities()
            gap_stocks = []
            
            if "data" in active_data:
                for stock in active_data["data"]:
                    features = self.ml_analyzer.extract_features(stock)
                    gap_percent = features.get('gap_up_down', 0)
                    
                    if abs(gap_percent) >= min_gap_percent:
                        gap_score = min(abs(gap_percent) * 15 + (features.get('trade_quantity', 0) / 1000000) * 10, 100)
                        
                        gap_stocks.append({
                            **stock,
                            **features,
                            'gap_score': gap_score,
                            'gap_type': 'GAP_UP' if gap_percent > 0 else 'GAP_DOWN',
                            'gap_percent': gap_percent
                        })
            
            gap_stocks.sort(key=lambda x: x['gap_score'], reverse=True)
            
            return {
                "scan_type": "Gap Scanner",
                "gap_opportunities": gap_stocks[:15],
                "criteria": {
                    "min_gap_percent": min_gap_percent
                },
                "scan_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def comprehensive_market_scan(self) -> Dict:
        """Perform comprehensive market scan with multiple strategies"""
        try:
            results = {
                "scan_timestamp": datetime.now().isoformat(),
                "market_overview": {},
                "opportunities": {}
            }
            
            # Get market overview
            active_data = self.nse_client.get_most_active_securities()
            gainers_data = self.nse_client.get_gainers_data()
            losers_data = self.nse_client.get_losers_data()
            
            # Market statistics
            total_gainers = len(gainers_data.get("data", []))
            total_losers = len(losers_data.get("data", []))
            
            results["market_overview"] = {
                "total_gainers": total_gainers,
                "total_losers": total_losers,
                "market_sentiment": "BULLISH" if total_gainers > total_losers else "BEARISH" if total_losers > total_gainers else "NEUTRAL",
                "gainer_loser_ratio": total_gainers / max(total_losers, 1)
            }
            
            # Run all scans
            results["opportunities"]["breakouts"] = self.scan_breakout_stocks()["breakout_opportunities"][:5]
            results["opportunities"]["momentum"] = self.scan_momentum_stocks()["momentum_opportunities"][:5]
            results["opportunities"]["reversals"] = self.scan_reversal_candidates()["reversal_candidates"][:5]
            results["opportunities"]["gaps"] = self.scan_gap_opportunities()["gap_opportunities"][:5]
            
            # Summary
            total_opportunities = sum(len(opp) for opp in results["opportunities"].values())
            results["summary"] = {
                "total_opportunities_found": total_opportunities,
                "scan_types_completed": 4,
                "recommendation": "Focus on breakout and momentum stocks in current market conditions"
            }
            
            return results
            
        except Exception as e:
            return {"error": str(e)}