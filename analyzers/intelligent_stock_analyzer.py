#!/usr/bin/env python3
"""
Intelligent Stock Analyzer
Analyzes stocks using AI/ML models and sends notifications for best opportunities.
"""
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from config import settings
from nse_client import NSEClient
from market_scanner import MarketScanner
from ml_analyzer import MLStockAnalyzer

@dataclass
class StockOpportunity:
    symbol: str
    current_price: float
    price_change: float
    volume: int
    ai_score: float
    recommendation: str
    confidence: float
    reasons: List[str]
    stop_loss: float
    target: float
    risk_reward_ratio: float
    sector: str

class IntelligentStockAnalyzer:
    def __init__(self):
        self.nse_client = NSEClient()
        self.market_scanner = MarketScanner()
        self.ml_analyzer = MLStockAnalyzer()
        self.previous_data_file = "previous_stock_analysis.json"
        self.min_confidence_threshold = 75.0
        self.min_ai_score_threshold = 70.0
        self.last_analysis_time = None
        self.analysis_interval = 900  # 15 minutes
        self.urgent_analysis_cooldown = 300  # 5 minutes
        self.last_urgent_analysis = None
        
    def fetch_all_stock_data(self) -> Dict:
        """Fetch comprehensive stock market data"""
        print("Fetching comprehensive stock market data...")
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'active_securities': {},
            'gainers': {},
            'losers': {},
            'scanner_results': {},
            'indices': {}
        }
        
        try:
            # 1. Active Securities
            data['active_securities'] = self.nse_client.get_most_active_securities()
            
            # 2. Gainers & Losers
            data['gainers'] = self.nse_client.get_gainers_data()
            data['losers'] = self.nse_client.get_losers_data()
            
            # 3. Market Scanner Results
            data['scanner_results'] = self.market_scanner.comprehensive_market_scan()
            
            # 4. Indices Data
            data['indices'] = self.nse_client.get_all_indices()
            
            print(f"Successfully fetched stock data from {len([k for k, v in data.items() if v and k != 'timestamp'])} sources")
            return data
            
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return data
    
    def extract_opportunities_from_data(self, data: Dict) -> List[StockOpportunity]:
        """Extract stock trading opportunities from all fetched data"""
        opportunities = []
        
        # Extract from scanner results (primary source)
        if 'scanner_results' in data and 'opportunities' in data['scanner_results']:
            for category, stocks in data['scanner_results']['opportunities'].items():
                for stock in stocks:
                    opportunity = self._create_opportunity_from_stock(stock, category)
                    if opportunity:
                        opportunities.append(opportunity)
        
        # Extract from active securities
        if 'active_securities' in data and 'data' in data['active_securities']:
            for stock in data['active_securities']['data']:
                opportunity = self._create_opportunity_from_stock(stock, 'active')
                if opportunity:
                    opportunities.append(opportunity)
        
        # Extract from gainers
        if 'gainers' in data and 'data' in data['gainers']:
            for stock in data['gainers']['data']:
                opportunity = self._create_opportunity_from_stock(stock, 'gainer')
                if opportunity:
                    opportunities.append(opportunity)
        
        print(f"Extracted {len(opportunities)} potential stock opportunities")
        return opportunities
    
    def _create_opportunity_from_stock(self, stock: Dict, category: str) -> Optional[StockOpportunity]:
        """Create opportunity from stock data"""
        try:
            symbol = stock.get('symbol', '')
            current_price = stock.get('ltp', stock.get('lastPrice', 0))
            price_change = stock.get('perChange', stock.get('pChange', 0))
            volume = stock.get('totalTradedVolume', stock.get('volume', 0))
            
            if not symbol or current_price <= 0 or volume < 100000:
                return None
            
            # Calculate AI score
            ai_score, is_ml_used = self._calculate_ai_score(volume, price_change, current_price)
            if ai_score < self.min_ai_score_threshold:
                return None
            
            # Get recommendation
            recommendation = self._get_recommendation(price_change, ai_score)
            confidence = self._calculate_confidence(ai_score, volume, abs(price_change))
            
            if confidence < self.min_confidence_threshold:
                return None
            
            # Calculate levels
            stop_loss, target = self._calculate_levels(current_price, recommendation)
            risk_reward_ratio = abs((target - current_price) / (current_price - stop_loss)) if stop_loss != current_price else 1.0
            
            # Generate reasons
            reasons = self._generate_reasons(volume, price_change, ai_score, is_ml_used, category)
            
            return StockOpportunity(
                symbol=symbol,
                current_price=float(current_price),
                price_change=float(price_change),
                volume=int(volume),
                ai_score=ai_score,
                recommendation=recommendation,
                confidence=confidence,
                reasons=reasons,
                stop_loss=stop_loss,
                target=target,
                risk_reward_ratio=risk_reward_ratio,
                sector=stock.get('industry', stock.get('sector', 'Unknown'))
            )
        except Exception as e:
            return None
    
    def _calculate_ai_score(self, volume: int, price_change: float, current_price: float) -> Tuple[float, bool]:
        """Calculate AI score using ML model + rule-based fallback"""
        try:
            # Try ML model first
            ml_features = self.ml_analyzer.extract_features({
                'totalTradedVolume': volume,
                'perChange': price_change,
                'ltp': current_price
            })
            
            if ml_features and hasattr(self.ml_analyzer, 'predict_score'):
                ml_score = self.ml_analyzer.predict_score(ml_features)
                if ml_score is not None:
                    return ml_score, True  # ML model used
        except Exception as e:
            print(f"Warning: ML model failed, using rule-based: {e}")
        
        # Fallback to rule-based scoring
        volume_score = min((volume / 100000) * 20, 40)
        price_score = min(abs(price_change) * 2, 30)
        liquidity_score = min((current_price * volume) / 1000000, 20)
        base_score = 10
        
        ai_score = volume_score + price_score + liquidity_score + base_score
        return max(30, min(ai_score, 95)), False  # Rule-based used
    
    def _get_recommendation(self, price_change: float, ai_score: float) -> str:
        """Get trading recommendation"""
        if price_change > 0:
            return "BUY" if ai_score > 80 else "HOLD"
        else:
            return "SELL" if ai_score > 80 else "AVOID"
    
    def _calculate_confidence(self, ai_score: float, volume: int, price_change: float) -> float:
        """Calculate confidence level"""
        score_factor = (ai_score / 100) * 40
        volume_factor = min((volume / 1000000) * 30, 30)
        momentum_factor = min(price_change / 5 * 30, 30)
        
        confidence = score_factor + volume_factor + momentum_factor
        return max(50, min(confidence, 100))
    
    def _calculate_levels(self, current_price: float, recommendation: str) -> Tuple[float, float]:
        """Calculate stop loss and target levels"""
        if recommendation == "BUY":
            stop_loss = current_price * 0.92  # 8% stop loss
            target = current_price * 1.15     # 15% target
        elif recommendation == "SELL":
            stop_loss = current_price * 1.08  # 8% stop loss
            target = current_price * 0.85     # 15% target
        else:  # HOLD/AVOID
            stop_loss = current_price * 0.95  # 5% stop loss
            target = current_price * 1.08     # 8% target
        
        return round(stop_loss, 2), round(target, 2)
    
    def _generate_reasons(self, volume: int, price_change: float, ai_score: float, 
                         is_ml_used: bool = False, category: str = '') -> List[str]:
        """Generate reasons for the opportunity"""
        reasons = []
        
        # AI/ML specific reasons
        if is_ml_used:
            if ai_score > 95:
                reasons.append("AI model detects exceptional stock opportunity")
            elif ai_score > 90:
                reasons.append("AI model predicts strong stock performance")
            elif ai_score > 80:
                reasons.append("AI model identifies high-probability stock trade")
            else:
                reasons.append("AI model recommends this stock")
        
        # Volume analysis
        if volume > 50000000:
            reasons.append("Massive trading volume surge")
        elif volume > 20000000:
            reasons.append("Exceptional volume activity")
        elif volume > 10000000:
            reasons.append("High volume activity")
        elif volume > 5000000:
            reasons.append("Above average volume")
        elif volume > 1000000:
            reasons.append("Moderate volume activity")
        
        # Price momentum
        if abs(price_change) > 10:
            reasons.append("Strong price momentum")
        elif abs(price_change) > 5:
            reasons.append("Good price movement")
        elif abs(price_change) > 2:
            reasons.append("Positive price action")
        
        # Category-based reasons
        if category == 'breakout':
            reasons.append("Breakout pattern detected")
        elif category == 'momentum':
            reasons.append("Momentum signal confirmed")
        elif category == 'reversal':
            reasons.append("Reversal opportunity identified")
        elif category == 'gainer':
            reasons.append("Top market gainer")
        
        # Score-based reasons (rule-based only)
        if not is_ml_used:
            if ai_score > 85:
                reasons.append("Excellent technical score")
            elif ai_score > 75:
                reasons.append("High technical score")
        
        return reasons or ["Meets minimum criteria"]
    
    def should_analyze_urgent(self, current_data: Dict) -> bool:
        """Check if urgent analysis needed for stocks"""
        if not self.is_market_open():
            return False
        
        # Check cooldown
        if (self.last_urgent_analysis and 
            time.time() - self.last_urgent_analysis < self.urgent_analysis_cooldown):
            return False
        
        # Check for unusual stock activity
        sources = [
            ('active_securities', 'data'),
            ('gainers', 'data'),
            ('losers', 'data')
        ]
        
        for source_key, data_key in sources:
            if source_key in current_data and data_key in current_data[source_key]:
                stocks = current_data[source_key][data_key][:10]  # Check top 10
                for stock in stocks:
                    volume = stock.get('totalTradedVolume', stock.get('volume', 0))
                    price_change = abs(stock.get('perChange', stock.get('pChange', 0)))
                    
                    # Try AI surge detection first
                    current_price = stock.get('ltp', stock.get('lastPrice', 0))
                    is_ai_surge = self._detect_ai_surge(volume, price_change, current_price)
                    
                    # Rule-based fallback
                    is_rule_surge = volume > 100000000 or price_change > 15
                    
                    if is_ai_surge or is_rule_surge:
                        detection_method = "AI MODEL" if is_ai_surge else "RULE-BASED"
                        print(f"🚨 URGENT STOCK ({detection_method}): {stock.get('symbol')} - Volume: {volume:,}, Change: {price_change:.1f}%")
                        return True
        
        return False
    
    def _detect_ai_surge(self, volume: int, price_change: float, current_price: float) -> bool:
        """Use AI model to detect stock surges"""
        try:
            ai_score, is_ml_used = self._calculate_ai_score(volume, price_change, current_price)
            
            if is_ml_used:
                # AI detected surge
                if ai_score > 90 and (volume > 20000000 or abs(price_change) > 8):
                    return True
                if ai_score > 95:
                    return True
            
            return False
        except Exception as e:
            return False
    
    def compare_with_previous_analysis(self, current_opportunities: List[StockOpportunity]) -> List[StockOpportunity]:
        """Compare with previous stock analysis"""
        try:
            with open(self.previous_data_file, 'r') as f:
                previous_data = json.load(f)
                previous_opportunities = previous_data.get('opportunities', [])
        except (FileNotFoundError, json.JSONDecodeError):
            previous_opportunities = []
        
        best_opportunities = []
        
        for current_opp in current_opportunities:
            is_new_or_improved = True
            
            for prev_opp in previous_opportunities:
                if current_opp.symbol == prev_opp.get('symbol'):
                    prev_confidence = prev_opp.get('confidence', 0)
                    if current_opp.confidence <= prev_confidence + 10:
                        is_new_or_improved = False
                    break
            
            if is_new_or_improved:
                best_opportunities.append(current_opp)
        
        best_opportunities.sort(key=lambda x: (x.confidence, x.ai_score), reverse=True)
        
        print(f"Found {len(best_opportunities)} new/improved stock opportunities out of {len(current_opportunities)} total")
        return best_opportunities[:5]  # Top 5 only
    
    def save_current_analysis(self, opportunities: List[StockOpportunity]):
        """Save current stock analysis"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'opportunities': [
                {
                    'symbol': opp.symbol,
                    'current_price': opp.current_price,
                    'price_change': opp.price_change,
                    'volume': opp.volume,
                    'ai_score': opp.ai_score,
                    'recommendation': opp.recommendation,
                    'confidence': opp.confidence,
                    'reasons': opp.reasons,
                    'stop_loss': opp.stop_loss,
                    'target': opp.target,
                    'risk_reward_ratio': opp.risk_reward_ratio,
                    'sector': opp.sector
                }
                for opp in opportunities
            ]
        }
        
        try:
            with open(self.previous_data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Error saving stock analysis: {e}")
    
    def send_websocket_notification(self, opportunities: List[StockOpportunity], is_urgent: bool = False) -> bool:
        """Send WebSocket notification for stock opportunities"""
        try:
            from websockets.websocket_streaming import manager
            import asyncio
            
            ws_data = {
                "type": "stock_alert" if not is_urgent else "urgent_stock_alert",
                "timestamp": datetime.now().isoformat(),
                "alert_count": len(opportunities),
                "opportunities": [
                    {
                        "symbol": opp.symbol,
                        "current_price": opp.current_price,
                        "price_change": opp.price_change,
                        "volume": opp.volume,
                        "recommendation": opp.recommendation,
                        "ai_score": opp.ai_score,
                        "confidence": opp.confidence,
                        "target": opp.target,
                        "stop_loss": opp.stop_loss,
                        "sector": opp.sector,
                        "reasons": opp.reasons[:2]
                    }
                    for opp in opportunities[:5]
                ]
            }
            
            connection_count = len(manager.active_connections)
            
            if connection_count > 0:
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    success = loop.run_until_complete(manager.broadcast_json(ws_data))
                    loop.close()
                    
                    if success:
                        print(f"✅ Stock WebSocket alert SENT to {connection_count} connections")
                        return True
                    else:
                        print(f"❌ Stock WebSocket alert failed to send")
                        return False
                except Exception as e:
                    print(f"❌ Stock WebSocket async error: {e}")
                    return False
            else:
                print("📭 No active WebSocket connections")
                manager.pending_messages.append(ws_data)
                return True
                
        except Exception as e:
            print(f"❌ Stock WebSocket notification failed: {e}")
            return False
    
    def send_telegram_notification(self, opportunities: List[StockOpportunity], is_urgent: bool = False) -> bool:
        """Send stock notifications to Telegram and WebSocket"""
        if not opportunities:
            print("No stock opportunities to send")
            return False
        
        try:
            alert_type = "🚨 URGENT STOCK ALERTS" if is_urgent else "📈 INTELLIGENT STOCK ALERTS"
            message = f"{alert_type} - {datetime.now().strftime('%H:%M')}\n\n"
            
            if is_urgent:
                message += f"⚡ AI/SYSTEM DETECTED UNUSUAL STOCK ACTIVITY!\n"
            
            message += f"📊 Found {len(opportunities)} HIGH-CONFIDENCE stock opportunities:\n\n"
            
            for i, opp in enumerate(opportunities, 1):
                message += f"{i}. {opp.symbol} ({opp.sector})\n"
                message += f"   💰 {opp.recommendation} at ₹{opp.current_price}\n"
                message += f"   📈 Change: {opp.price_change:+.1f}% | Vol: {opp.volume:,}\n"
                message += f"   🎯 Target: ₹{opp.target} | SL: ₹{opp.stop_loss}\n"
                message += f"   🤖 AI Score: {opp.ai_score:.0f} | Confidence: {opp.confidence:.0f}%\n"
                message += f"   💡 {', '.join(opp.reasons[:2])}\n\n"
            
            message += "⚡ Only BEST stock opportunities with 75%+ confidence\n"
            message += "🔄 Next analysis in 15 minutes"
            
            # Send to Telegram
            telegram_success = False
            if settings.telegram_bot_token:
                url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
                chat_id = settings.telegram_chat_id or "-1002981590794"
                
                data = {"chat_id": chat_id, "text": message}
                response = requests.post(url, data=data, timeout=10)
                
                if response.status_code == 200 and response.json().get('ok'):
                    print(f"Telegram stock alert sent with {len(opportunities)} opportunities")
                    telegram_success = True
                else:
                    print(f"Telegram failed: {response.text}")
            
            # Send to WebSocket
            websocket_success = self.send_websocket_notification(opportunities, is_urgent)
            
            return telegram_success or websocket_success
                
        except Exception as e:
            print(f"Error sending stock notifications: {e}")
            return False
    
    def is_market_open(self) -> bool:
        """Check if market is open"""
        now = datetime.now()
        
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_start <= now <= market_end
    
    def run_intelligent_analysis(self, is_urgent: bool = False) -> bool:
        """Run complete intelligent stock analysis"""
        if not is_urgent and (not self.is_market_open() or 
                             (self.last_analysis_time and 
                              time.time() - self.last_analysis_time < self.analysis_interval)):
            return False
        
        analysis_type = "URGENT" if is_urgent else "SCHEDULED"
        print(f"Starting {analysis_type} intelligent stock analysis...")
        start_time = time.time()
        
        try:
            # 1. Fetch all data
            all_data = self.fetch_all_stock_data()
            
            # 2. Extract opportunities
            opportunities = self.extract_opportunities_from_data(all_data)
            
            if not opportunities:
                print("No stock opportunities found meeting criteria")
                if not is_urgent:
                    self.last_analysis_time = time.time()
                return False
            
            # 3. Compare with previous analysis
            best_opportunities = self.compare_with_previous_analysis(opportunities)
            
            if not best_opportunities:
                print("No new or improved stock opportunities found")
                if not is_urgent:
                    self.last_analysis_time = time.time()
                return False
            
            # 4. Save current analysis
            self.save_current_analysis(opportunities)
            
            # 5. Send notifications
            success = self.send_telegram_notification(best_opportunities, is_urgent)
            
            if not is_urgent:
                self.last_analysis_time = time.time()
            else:
                self.last_urgent_analysis = time.time()
            
            analysis_time = time.time() - start_time
            print(f"{analysis_type} stock analysis completed in {analysis_time:.1f}s - {'Sent' if success else 'Failed'} notification")
            return success
            
        except Exception as e:
            print(f"{analysis_type} stock analysis failed: {e}")
            if not is_urgent:
                self.last_analysis_time = time.time()
            return False
    
    def run_continuous_monitoring(self) -> bool:
        """Continuous monitoring for unusual stock activity"""
        if not self.is_market_open():
            return False
        
        if (self.last_urgent_analysis and 
            time.time() - self.last_urgent_analysis < self.urgent_analysis_cooldown):
            return False
        
        try:
            # Quick data fetch for monitoring
            current_data = {
                'active_securities': self.nse_client.get_most_active_securities(),
                'gainers': self.nse_client.get_gainers_data()
            }
            
            if self.should_analyze_urgent(current_data):
                return self.run_intelligent_analysis(is_urgent=True)
            
            return False
            
        except Exception as e:
            print(f"Stock monitoring error: {e}")
            return False

def main():
    """Main function for standalone execution"""
    analyzer = IntelligentStockAnalyzer()
    
    print("Starting Intelligent Stock Analyzer")
    print("=" * 60)
    
    while True:
        try:
            # Run scheduled analysis
            analyzer.run_intelligent_analysis()
            
            # Run continuous monitoring
            analyzer.run_continuous_monitoring()
            
            # Wait 2 minutes before next check
            print("Waiting 2 minutes before next check...")
            time.sleep(120)
            
        except KeyboardInterrupt:
            print("\nStopping stock analyzer...")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()