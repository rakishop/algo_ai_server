#!/usr/bin/env python3
"""
Intelligent Derivative Analyzer
Fetches all related APIs, analyzes data, compares with previous data,
and sends recommendations only when finding the best opportunities.
"""
import json
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from config import settings
from nse_client import NSEClient
import numpy as np
from ws.websocket_streaming import manager

@dataclass
class DerivativeOpportunity:
    symbol: str
    option_type: str
    strike_price: float
    expiry_date: str
    current_price: float
    price_change: float
    volume: int
    open_interest: int
    ai_score: float
    recommendation: str
    confidence: float
    reasons: List[str]
    stop_loss: float
    target: float
    risk_reward_ratio: float

class IntelligentDerivativeAnalyzer:
    def __init__(self):
        self.nse_client = NSEClient()
        self.previous_data_file = "previous_derivative_analysis.json"
        self.analysis_history_file = "derivative_analysis_history.json"
        self.min_confidence_threshold = 75.0
        self.min_ai_score_threshold = 70.0
        self.last_analysis_time = None
        self.analysis_interval = 900  # 15 minutes
        self.urgent_analysis_cooldown = 300  # 5 minutes between urgent analyses
        self.last_urgent_analysis = None
        
    def fetch_all_derivative_data(self) -> Dict:
        """Fetch data from all derivative-related APIs"""
        print("Fetching comprehensive derivative data...")
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'market_snapshot': {},
            'active_underlyings': {},
            'oi_spurts': {},
            'option_chains': {},
            'equity_snapshot': {},
            'futures_data': {}
        }
        
        try:
            # 1. Market Snapshot
            data['market_snapshot'] = self.nse_client.get_derivatives_snapshot()
            
            # 2. Active Underlyings
            data['active_underlyings'] = self.nse_client.get_most_active_underlying()
            
            # 3. OI Spurts
            data['oi_spurts'] = {
                'underlyings': self.nse_client.get_oi_spurts_underlyings(),
                'contracts': self.nse_client.get_oi_spurts_contracts()
            }
            
            # 4. Equity Snapshot (derivatives)
            data['equity_snapshot'] = self.nse_client.get_derivatives_equity_snapshot(50)
            
            # 5. Option Chains for major indices
            major_indices = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
            for symbol in major_indices:
                try:
                    contract_info = self.nse_client.get_option_chain_info(symbol)
                    if "error" not in contract_info and "records" in contract_info:
                        expiry_dates = contract_info["records"].get("expiryDates", [])
                        if expiry_dates:
                            # Find next valid expiry (not expired)
                            valid_expiry = self._get_next_valid_expiry(expiry_dates)
                            if valid_expiry:
                                option_chain = self.nse_client.get_option_chain(symbol, valid_expiry)
                                if "error" not in option_chain:
                                    data['option_chains'][symbol] = {
                                        'contract_info': contract_info,
                                        'option_chain': option_chain,
                                        'expiry': valid_expiry
                                    }
                                    print(f"Using expiry {valid_expiry} for {symbol}")
                except Exception as e:
                    print(f"Warning: Error fetching option chain for {symbol}: {e}")
            
            print(f"Successfully fetched data from {len([k for k, v in data.items() if v and k != 'timestamp'])} sources")
            return data
            
        except Exception as e:
            print(f"Error fetching derivative data: {e}")
            return data
    
    def extract_opportunities_from_data(self, data: Dict) -> List[DerivativeOpportunity]:
        """Extract trading opportunities from all fetched data"""
        opportunities = []
        
        # Extract from equity snapshot
        if 'equity_snapshot' in data and 'volume' in data['equity_snapshot']:
            contracts = data['equity_snapshot']['volume'].get('data', [])
            for contract in contracts:
                opportunity = self._create_opportunity_from_contract(contract)
                if opportunity:
                    opportunities.append(opportunity)
        
        # Extract from option chains
        if 'option_chains' in data:
            for symbol, chain_data in data['option_chains'].items():
                if 'option_chain' in chain_data and 'records' in chain_data['option_chain']:
                    records = chain_data['option_chain']['records']
                    underlying_value = records.get('underlyingValue', 0)
                    
                    if 'data' in records:
                        for strike_data in records['data']:
                            # Process Call options
                            if 'CE' in strike_data:
                                ce_data = strike_data['CE']
                                opportunity = self._create_opportunity_from_option_data(
                                    ce_data, symbol, underlying_value, "Call"
                                )
                                if opportunity:
                                    opportunities.append(opportunity)
                            
                            # Process Put options
                            if 'PE' in strike_data:
                                pe_data = strike_data['PE']
                                opportunity = self._create_opportunity_from_option_data(
                                    pe_data, symbol, underlying_value, "Put"
                                )
                                if opportunity:
                                    opportunities.append(opportunity)
        
        print(f"Extracted {len(opportunities)} potential opportunities")
        return opportunities
    
    def _create_opportunity_from_contract(self, contract: Dict) -> Optional[DerivativeOpportunity]:
        """Create opportunity from contract data"""
        try:
            # Check if contract is expired
            expiry_date = contract.get('expiryDate', '')
            if expiry_date and self._is_expired(expiry_date):
                return None
            
            volume = contract.get('numberOfContractsTraded', 0)
            price_change = contract.get('pChange', 0)
            current_price = contract.get('lastPrice', 0)
            
            if volume < 100000 or abs(price_change) < 5 or current_price <= 0:
                return None
            
            ai_score, is_ml_used = self._calculate_ai_score(volume, price_change, current_price)
            if ai_score < self.min_ai_score_threshold:
                return None
            
            option_type = contract.get('optionType', 'Call')
            recommendation = self._get_recommendation(option_type, price_change, 0, contract.get('strikePrice', 0), volume, contract.get('openInterest', 0), current_price)
            confidence = self._calculate_confidence(ai_score, volume, abs(price_change))
            
            if confidence < self.min_confidence_threshold:
                return None
            
            stop_loss, target = self._calculate_levels(current_price, recommendation)
            risk_reward_ratio = abs((target - current_price) / (current_price - stop_loss)) if stop_loss != current_price else 1.0
            
            reasons = self._generate_reasons(volume, price_change, ai_score, is_ml_used)
            
            return DerivativeOpportunity(
                symbol=contract.get('underlying', ''),
                option_type=option_type,
                strike_price=contract.get('strikePrice', 0),
                expiry_date=contract.get('expiryDate', ''),
                current_price=current_price,
                price_change=price_change,
                volume=volume,
                open_interest=contract.get('openInterest', 0),
                ai_score=ai_score,
                recommendation=recommendation,
                confidence=confidence,
                reasons=reasons,
                stop_loss=stop_loss,
                target=target,
                risk_reward_ratio=risk_reward_ratio
            )
        except Exception as e:
            return None
    
    def _create_opportunity_from_option_data(self, option_data: Dict, symbol: str, 
                                           underlying_value: float, option_type: str) -> Optional[DerivativeOpportunity]:
        """Create opportunity from option chain data"""
        try:
            # Check if option is expired
            expiry_date = option_data.get('expiryDate', '')
            if expiry_date and self._is_expired(expiry_date):
                return None
            
            volume = option_data.get('totalTradedVolume', 0)
            price_change = option_data.get('pChange', 0)
            current_price = option_data.get('lastPrice', 0)
            
            if volume < 50000 or abs(price_change) < 10 or current_price <= 0:
                return None
            
            ai_score, is_ml_used = self._calculate_ai_score(volume, price_change, current_price)
            if ai_score < self.min_ai_score_threshold:
                return None
            
            recommendation = self._get_recommendation(option_type, price_change, underlying_value, option_data.get('strikePrice', 0), volume, option_data.get('openInterest', 0), current_price)
            confidence = self._calculate_confidence(ai_score, volume, abs(price_change))
            
            if confidence < self.min_confidence_threshold:
                return None
            
            stop_loss, target = self._calculate_levels(current_price, recommendation)
            risk_reward_ratio = abs((target - current_price) / (current_price - stop_loss)) if stop_loss != current_price else 1.0
            
            reasons = self._generate_reasons(volume, price_change, ai_score, is_ml_used)
            
            return DerivativeOpportunity(
                symbol=symbol,
                option_type=option_type,
                strike_price=option_data.get('strikePrice', 0),
                expiry_date=option_data.get('expiryDate', ''),
                current_price=current_price,
                price_change=price_change,
                volume=volume,
                open_interest=option_data.get('openInterest', 0),
                ai_score=ai_score,
                recommendation=recommendation,
                confidence=confidence,
                reasons=reasons,
                stop_loss=stop_loss,
                target=target,
                risk_reward_ratio=risk_reward_ratio
            )
        except Exception as e:
            return None
    
    def _calculate_ai_score(self, volume: int, price_change: float, current_price: float) -> Tuple[float, bool]:
        """Calculate AI score using ML model + rule-based fallback"""
        try:
            # Try ML model first
            ml_score = self._get_ml_prediction(volume, price_change, current_price)
            if ml_score is not None:
                return ml_score, True  # ML model used
        except Exception as e:
            print(f"Warning: ML model failed, using rule-based: {e}")
        
        # Fallback to rule-based scoring
        volume_score = min((volume / 1000) * 10, 40)
        price_score = min(abs(price_change) * 1.5, 30)
        liquidity_score = min((current_price * volume) / 100000, 20)
        base_score = 10
        
        ai_score = volume_score + price_score + liquidity_score + base_score
        return max(30, min(ai_score, 95)), False  # Rule-based used
    
    def _get_ml_prediction(self, volume: int, price_change: float, current_price: float) -> Optional[float]:
        """Get prediction from trained ML model"""
        try:
            from routes.ml_derivatives_model import DerivativesMLModel
            
            # Initialize ML model
            ml_model = DerivativesMLModel()
            
            if not ml_model.is_trained:
                return None
            
            # Prepare features for ML model
            features = np.array([[
                volume / 1000000,  # Volume ratio
                abs(price_change) / 100,  # Price momentum
                0,  # OI ratio (placeholder)
                0,  # Moneyness (placeholder)
                (current_price * volume) / 1000000,  # Liquidity score
                1,  # Option type (placeholder)
                0   # Premium turnover (placeholder)
            ]])
            
            # Get ML prediction
            X_scaled = ml_model.scaler.transform(features)
            ml_probability = ml_model.model.predict_proba(X_scaled)[0][1]
            
            # Convert probability to score (30-95 range)
            ml_score = 30 + (ml_probability * 65)
            
           # print(f"ML Score: {ml_score:.1f} (probability: {ml_probability:.3f})")
            return ml_score
            
        except Exception as e:
            return None
    
    def _get_recommendation(self, option_type: str, price_change: float, underlying_value: float = 0, strike_price: float = 0, volume: int = 0, open_interest: int = 0, current_price: float = 0) -> str:
        """Get trading recommendation based on multiple parameters"""
        # Calculate moneyness
        if underlying_value > 0 and strike_price > 0:
            moneyness = underlying_value / strike_price
        else:
            moneyness = 1.0
        
        # Volume analysis
        high_volume = volume > 1000000
        exceptional_volume = volume > 5000000
        
        # Price momentum analysis
        strong_momentum = abs(price_change) > 20
        moderate_momentum = abs(price_change) > 10
        
        # Liquidity analysis
        high_liquidity = current_price * volume > 50000000
        
        if option_type == "Call":
            # ITM Call (moneyness > 1.0)
            if moneyness > 1.02:
                if price_change > 0 and high_volume:
                    return "BUY CALL" if strong_momentum else "BUY CALL"
                elif price_change < -15 and exceptional_volume:
                    return "SELL CALL"
                else:
                    return "HOLD CALL"
            # OTM Call (moneyness < 1.0)
            elif moneyness < 0.98:
                if price_change > 15 and exceptional_volume:
                    return "BUY CALL"
                elif price_change < 0:
                    return "AVOID CALL"
                else:
                    return "WATCH CALL"
            # ATM Call
            else:
                if strong_momentum and high_volume:
                    return "BUY CALL" if price_change > 0 else "SELL CALL"
                elif price_change < -10:
                    return "AVOID CALL (Theta Decay)"
                else:
                    return "NEUTRAL"
        
        else:  # Put
            # ITM Put (moneyness < 1.0)
            if moneyness < 0.98:
                if price_change > 0 and high_volume:
                    return "BUY PUT" if strong_momentum else "BUY PUT"
                elif price_change < -15 and exceptional_volume:
                    return "SELL PUT"
                else:
                    return "HOLD PUT"
            # OTM Put (moneyness > 1.0)
            elif moneyness > 1.02:
                if price_change > 15 and exceptional_volume:
                    return "BUY PUT"
                elif price_change < 0:
                    return "AVOID PUT"
                else:
                    return "WATCH PUT"
            # ATM Put
            else:
                if strong_momentum and high_volume:
                    return "BUY PUT" if price_change > 0 else "SELL PUT"
                elif price_change < -10:
                    return "AVOID PUT (Theta Decay)"
                else:
                    return "NEUTRAL"
    
    def _calculate_confidence(self, ai_score: float, volume: int, price_change: float) -> float:
        """Calculate confidence level"""
        score_factor = (ai_score / 100) * 40
        volume_factor = min((volume / 1000000) * 30, 30)
        momentum_factor = min(price_change / 10 * 30, 30)
        
        confidence = score_factor + volume_factor + momentum_factor
        return max(50, min(confidence, 100))
    
    def _calculate_levels(self, current_price: float, recommendation: str) -> Tuple[float, float]:
        """Calculate stop loss and target levels"""
        if "BUY" in recommendation:
            stop_loss = current_price * 0.85  # 15% stop loss
            target = current_price * 1.30     # 30% target
        else:
            stop_loss = current_price * 1.15  # 15% stop loss
            target = current_price * 0.70     # 30% target
        
        return round(stop_loss, 2), round(target, 2)
    
    def _is_expired(self, expiry_date_str: str) -> bool:
        """Check if expiry date is expired (allow trading on expiry day during market hours)"""
        try:
            expiry_date = datetime.strptime(expiry_date_str, "%d-%b-%Y").date()
            current_date = datetime.now().date()
            
            # If expiry is in the past, it's expired
            if expiry_date < current_date:
                return True
            
            # If expiry is today, check if market is still open
            if expiry_date == current_date:
                return not self.is_market_open()
            
            # Future expiry is valid
            return False
        except ValueError:
            return False
    
    def _get_next_valid_expiry(self, expiry_dates: List[str]) -> Optional[str]:
        """Get next valid (non-expired) expiry date"""
        current_date = datetime.now().date()
        market_open = self.is_market_open()
        
        for expiry_str in expiry_dates:
            try:
                # Parse expiry date (format: "30-Sep-2025")
                expiry_date = datetime.strptime(expiry_str, "%d-%b-%Y").date()
                
                # If expiry is in the future, it's valid
                if expiry_date > current_date:
                    return expiry_str
                
                # If expiry is today, check if market is still open
                if expiry_date == current_date and market_open:
                    return expiry_str
                    
            except ValueError:
                continue
        
        return None
    
    def _generate_reasons(self, volume: int, price_change: float, ai_score: float, is_ml_used: bool = False) -> List[str]:
        """Generate reasons for the opportunity"""
        reasons = []
        
        # AI/ML specific reasons
        if is_ml_used:
            if ai_score > 95:
                reasons.append("AI model detects exceptional surge opportunity")
            elif ai_score > 90:
                reasons.append("AI model predicts exceptional opportunity")
            elif ai_score > 80:
                reasons.append("AI model identifies high-probability trade")
            else:
                reasons.append("AI model recommends this opportunity")
        
        # Volume analysis
        if volume > 10000000:
            reasons.append("Massive volume surge detected")
        elif volume > 5000000:
            reasons.append("Exceptional volume activity")
        elif volume > 2000000:
            reasons.append("High volume activity")
        elif volume > 1000000:
            reasons.append("Above average volume")
        elif volume > 500000:
            reasons.append("Moderate volume activity")
        
        # Price momentum
        if abs(price_change) > 50:
            reasons.append("Strong price momentum")
        elif abs(price_change) > 20:
            reasons.append("Good price movement")
        
        # Score-based reasons
        if not is_ml_used:  # Only for rule-based
            if ai_score > 85:
                reasons.append("Excellent technical score")
            elif ai_score > 75:
                reasons.append("High technical score")
        
        return reasons or ["Meets minimum criteria"]
    
    def compare_with_previous_analysis(self, current_opportunities: List[DerivativeOpportunity]) -> List[DerivativeOpportunity]:
        """Compare current opportunities with previous analysis"""
        try:
            with open(self.previous_data_file, 'r') as f:
                previous_data = json.load(f)
                previous_opportunities = previous_data.get('opportunities', [])
        except (FileNotFoundError, json.JSONDecodeError):
            previous_opportunities = []
        
        # Find new or significantly improved opportunities
        best_opportunities = []
        
        for current_opp in current_opportunities:
            is_new_or_improved = True
            
            for prev_opp in previous_opportunities:
                if (current_opp.symbol == prev_opp.get('symbol') and 
                    current_opp.strike_price == prev_opp.get('strike_price') and
                    current_opp.option_type == prev_opp.get('option_type')):
                    
                    # Check if significantly improved
                    prev_confidence = prev_opp.get('confidence', 0)
                    if current_opp.confidence <= prev_confidence + 10:  # Must be 10% better
                        is_new_or_improved = False
                    break
            
            if is_new_or_improved:
                best_opportunities.append(current_opp)
        
        # Sort by confidence and take top opportunities
        best_opportunities.sort(key=lambda x: (x.confidence, x.ai_score), reverse=True)
        
        print(f"Found {len(best_opportunities)} new/improved opportunities out of {len(current_opportunities)} total")
        return best_opportunities[:8]  # Top 8 for better strategy options
    
    def save_current_analysis(self, opportunities: List[DerivativeOpportunity]):
        """Save current analysis for future comparison"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'opportunities': [
                {
                    'symbol': opp.symbol,
                    'option_type': opp.option_type,
                    'strike_price': opp.strike_price,
                    'expiry_date': opp.expiry_date,
                    'current_price': opp.current_price,
                    'price_change': opp.price_change,
                    'volume': opp.volume,
                    'open_interest': opp.open_interest,
                    'ai_score': opp.ai_score,
                    'recommendation': opp.recommendation,
                    'confidence': opp.confidence,
                    'reasons': opp.reasons,
                    'stop_loss': opp.stop_loss,
                    'target': opp.target,
                    'risk_reward_ratio': opp.risk_reward_ratio
                }
                for opp in opportunities
            ]
        }
        
        try:
            with open(self.previous_data_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Also save to history
            self._save_to_history(data)
        except Exception as e:
            print(f"Warning: Error saving analysis: {e}")
    
    def _save_to_history(self, data: Dict):
        """Save analysis to history file"""
        try:
            try:
                with open(self.analysis_history_file, 'r') as f:
                    history = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                history = []
            
            history.append(data)
            
            # Keep only last 50 analyses
            if len(history) > 50:
                history = history[-50:]
            
            with open(self.analysis_history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"Warning: Error saving to history: {e}")
    
    def send_websocket_notification(self, analysis_data: Dict, websocket_manager=None) -> bool:
        """Send WebSocket notification using prepared analysis data"""
        try:
            # Use provided manager or import from ws module
            if websocket_manager:
                manager = websocket_manager
                print(f"Using provided manager {id(manager)} with {len(manager.active_connections)} active connections")
                print(f"Manager type: {type(manager)}")
            else:
                from ws.websocket_streaming import manager
                print(f"Using imported manager {id(manager)} with {len(manager.active_connections)} active connections")
                print(f"Manager type: {type(manager)}")
            
            # Debug: Check if manager has the connections attribute
            print(f"Manager has active_connections attr: {hasattr(manager, 'active_connections')}")
            if hasattr(manager, 'active_connections'):
                print(f"active_connections type: {type(manager.active_connections)}")
                print(f"active_connections content: {manager.active_connections}")
            
            import asyncio
            
            ws_data = {
                "type": "derivative_alert",
                "timestamp": datetime.now().isoformat(),
                "alert_count": analysis_data["alert_count"],
                "market_sentiment": analysis_data["market_sentiment"],
                "recommended_strategy": analysis_data["best_strategy"],
                "call_volume": analysis_data["call_volume"],
                "put_volume": analysis_data["put_volume"],
                "weighted_momentum": analysis_data["weighted_momentum"],
                "opportunities": [
                    {
                        "symbol": opp.symbol,
                        "option_type": opp.option_type,
                        "strike_price": opp.strike_price,
                        "current_price": opp.current_price,
                        "recommendation": opp.recommendation,
                        "ai_score": opp.ai_score,
                        "confidence": opp.confidence,
                        "target": opp.target,
                        "stop_loss": opp.stop_loss,
                        "reasons": opp.reasons[:2]
                    }
                    for opp in analysis_data["opportunities"][:5]
                ]
            }
            
            
            print(f"Complete WebSocket payload:")
       
            
            connection_count = len(manager.active_connections)
            print(f"Active WebSocket connections ({connection_count}):")
            for i, conn in enumerate(manager.active_connections, 1):
                print(f"  {i}. {conn.client.host}:{conn.client.port}")
            
            if connection_count > 0:
                try:
                    # Use existing event loop instead of creating new one
                    try:
                        loop = asyncio.get_running_loop()
                        task = loop.create_task(manager.broadcast_json(ws_data))
                        # Don't wait for completion in sync context
                        print(f"WebSocket broadcast task created for {connection_count} connections")
                        return True
                    except RuntimeError:
                        # No running loop, create new one
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        success = loop.run_until_complete(manager.broadcast_json(ws_data))
                        loop.close()
                        
                        if success:
                            print(f"WebSocket alert SENT to {connection_count} connections")
                            return True
                        else:
                            print(f"WebSocket alert failed to send")
                            return False
                        
                except Exception as e:
                    print(f"WebSocket async error: {e}")
                    return False
            else:
                print(f"No active WebSocket connections (manager has {len(manager.active_connections)} connections)")
                manager.pending_messages.append(ws_data)
                print(f"Added to pending messages queue (now {len(manager.pending_messages)} pending)")
                return True
                
        except Exception as e:
            print(f"WebSocket notification failed: {e}")
            return False
    
    def send_telegram_notification(self, opportunities: List[DerivativeOpportunity], is_urgent: bool = False, websocket_manager=None) -> bool:
        """Send intelligent notification to Telegram and WebSocket"""
        if not opportunities:
            print("No opportunities to send")
            return False
        
        telegram_success = False
        websocket_success = False
        
        try:
            # Prepare analysis data once for both Telegram and WebSocket
            analysis_data = self._prepare_analysis_data(opportunities)
            market_sentiment = analysis_data["market_sentiment"]
            best_strategy = analysis_data["best_strategy"]
            call_volume = analysis_data["call_volume"]
            put_volume = analysis_data["put_volume"]
            call_oi = analysis_data["call_oi"]
            put_oi = analysis_data["put_oi"]
            weighted_momentum = analysis_data["weighted_momentum"]
            avg_liquidity = analysis_data["avg_liquidity"]
            
            # Send to Telegram
            alert_type = "URGENT DERIVATIVE ALERTS" if is_urgent else "INTELLIGENT DERIVATIVE ALERTS"
            message = f"{alert_type} - {datetime.now().strftime('%H:%M')}\n\n"
            
            if is_urgent:
                message += f"AI/SYSTEM DETECTED UNEXPECTED MARKET ACTIVITY!\n"
            
            message += f"Found {len(opportunities)} HIGH-CONFIDENCE opportunities:\n"
            message += f"Market Sentiment: {market_sentiment}\n"
            message += f"Analysis: Vol C/P {call_volume:,}/{put_volume:,} | OI C/P {call_oi:,}/{put_oi:,}\n"
            message += f"Momentum: {weighted_momentum:+.1f}% | Liquidity: Rs{avg_liquidity/1000000:.1f}M\n\n"
            
            for i, opp in enumerate(opportunities, 1):
                message += f"{i}. {opp.symbol} {opp.option_type}\n"
                message += f"   Strike: {opp.strike_price} | Expiry: {opp.expiry_date}\n"
                message += f"   {opp.recommendation} at Rs{opp.current_price} | Market: {market_sentiment}\n"
                message += f"   Change: {opp.price_change:+.1f}% | Vol: {opp.volume:,}\n"
                message += f"   Target: Rs{opp.target} | SL: Rs{opp.stop_loss}\n"
                message += f"   AI Score: {opp.ai_score:.0f} | Confidence: {opp.confidence:.0f}%\n"
                message += f"   {', '.join(opp.reasons[:2])}\n\n"
            
            message += f"RECOMMENDED STRATEGY: {best_strategy}\n\n"
            message += "Only BEST opportunities with 75%+ confidence\n"
            message += "Next analysis in 15 minutes"
            
            # Send to Telegram
            if settings.telegram_bot_token:
                url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
                chat_id = settings.telegram_chat_id or "-1002981590794"
                
                data = {"chat_id": chat_id, "text": message}
                response = requests.post(url, data=data, timeout=10)
                
                if response.status_code == 200 and response.json().get('ok'):
                    print(f"Telegram alert sent with {len(opportunities)} opportunities")
                    telegram_success = True
                else:
                    print(f"Telegram failed: {response.text}")
            else:
                print("Warning: Telegram bot token not configured")
            
            # Send to WebSocket using same analysis data
            print(f"About to send WebSocket with strategy: {analysis_data.get('best_strategy', 'MISSING')}")
            websocket_success = self.send_websocket_notification(analysis_data, websocket_manager)
            
            return telegram_success or websocket_success
                
        except Exception as e:
            print(f"Error sending notifications: {e}")
            return False
    
    def is_market_open(self) -> bool:
        """Check if market is open"""
        now = datetime.now()
        
        # Skip weekends
        if now.weekday() >= 5:
            return False
        
        # Market hours: 9:00 AM to 3:30 PM
        market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_start <= now <= market_end
        #return True
    
    def should_analyze(self) -> bool:
        """Check if analysis should be performed"""
        if not self.is_market_open():
            return False
        
        if self.last_analysis_time is None:
            return True
        
        time_since_last = time.time() - self.last_analysis_time
        return time_since_last >= self.analysis_interval
    
    def should_analyze_urgent(self, current_data: Dict) -> bool:
        """Check if urgent analysis needed using AI model + rule-based fallback"""
        if not self.is_market_open():
            return False
        
        # Check for volume spikes or unusual price movements
        if 'equity_snapshot' in current_data and 'volume' in current_data['equity_snapshot']:
            contracts = current_data['equity_snapshot']['volume'].get('data', [])
            for contract in contracts[:10]:  # Check top 10
                volume = contract.get('numberOfContractsTraded', 0)
                price_change = abs(contract.get('pChange', 0))
                current_price = contract.get('lastPrice', 0)
                
                # Try AI model for surge detection first
                is_ai_surge = self._detect_ai_surge(volume, price_change, current_price)
                
                # Rule-based fallback for extreme cases
                is_rule_surge = volume > 15000000 or price_change > 25
                
                if is_ai_surge or is_rule_surge:
                    detection_method = "AI MODEL" if is_ai_surge else "RULE-BASED"
                    print(f"URGENT ({detection_method}): {contract.get('underlying')} - Volume: {volume:,}, Change: {price_change:.1f}%")
                    return True
        
        return False
    
    def _detect_ai_surge(self, volume: int, price_change: float, current_price: float) -> bool:
        """Use AI model to detect volume/price surges"""
        try:
            # Get AI score for current data
            ai_score, is_ml_used = self._calculate_ai_score(volume, price_change, current_price)
            
            if is_ml_used:
                # AI model detected surge if score > 90 with high volume/price activity
                if ai_score > 90 and (volume > 5000000 or abs(price_change) > 15):
                    return True
                # Lower threshold for exceptional AI confidence
                if ai_score > 95:
                    return True
            
            return False
            
        except Exception as e:
            return False
    
    def train_ml_model_with_current_data(self, opportunities: List[DerivativeOpportunity]):
        """Train ML model with current opportunities data"""
        try:
            from routes.ml_derivatives_model import DerivativesMLModel
            
            ml_model = DerivativesMLModel()
            
            # Convert opportunities to training format
            training_data = {
                "volume": {
                    "data": [
                        {
                            "numberOfContractsTraded": opp.volume,
                            "pChange": opp.price_change,
                            "lastPrice": opp.current_price,
                            "openInterest": opp.open_interest,
                            "strikePrice": opp.strike_price,
                            "underlyingValue": opp.current_price * 100,  # Estimate
                            "optionType": opp.option_type,
                            "premiumTurnover": opp.volume * opp.current_price,
                            "underlying": opp.symbol,
                            "expiryDate": opp.expiry_date
                        }
                        for opp in opportunities
                    ]
                }
            }
            
            # Train model
            result = ml_model.train_with_historical_data(training_data)
            print(f"ML model training: {result.get('status', 'unknown')}")
            
        except Exception as e:
            print(f"Warning: ML model training failed: {e}")
    
    def run_intelligent_analysis(self, is_urgent: bool = False, websocket_manager=None, force: bool = False) -> bool:
        """Run complete intelligent analysis with ML integration"""
        if not is_urgent and not self.should_analyze():
            print(f"Skipping analysis - Market closed or too soon (last: {self.last_analysis_time})")
            return False
        
        analysis_type = "URGENT" if is_urgent else "SCHEDULED"
        print(f"Starting {analysis_type} intelligent derivative analysis with ML...")
        start_time = time.time()
        
        try:
            # 1. Fetch all data
            all_data = self.fetch_all_derivative_data()
            
            # 2. Extract opportunities
            opportunities = self.extract_opportunities_from_data(all_data)
            
            if not opportunities:
                print("No opportunities found meeting criteria")
                if not is_urgent:
                    self.last_analysis_time = time.time()
                return False
            
            # 3. Train ML model with current data (background)
            self.train_ml_model_with_current_data(opportunities)
            
            # 4. Compare with previous analysis (or force top opportunities)
            if force:
                best_opportunities = sorted(opportunities, key=lambda x: x.confidence, reverse=True)[:3]
                print(f"FORCE MODE: Using top {len(best_opportunities)} opportunities")
            else:
                best_opportunities = self.compare_with_previous_analysis(opportunities)
            
            if not best_opportunities:
                print("No new or improved opportunities found")
                if not is_urgent:
                    self.last_analysis_time = time.time()
                return False
            
            # 5. Save current analysis
            self.save_current_analysis(opportunities)
            
            # 6. Send notification with urgency flag and websocket manager
            success = self.send_telegram_notification(best_opportunities, is_urgent, websocket_manager)
            
            if not is_urgent:
                self.last_analysis_time = time.time()
            else:
                self.last_urgent_analysis = time.time()
            
            analysis_time = time.time() - start_time
            
            print(f"{analysis_type} analysis completed in {analysis_time:.1f}s - {'Sent' if success else 'Failed'} notification")
            return success
            
        except Exception as e:
            print(f"{analysis_type} analysis failed: {e}")
            if not is_urgent:
                self.last_analysis_time = time.time()
            return False
    
    def _generate_best_strategy(self, opportunities: List[DerivativeOpportunity], market_sentiment: str, weighted_momentum: float, call_volume: int, put_volume: int) -> str:
        """Generate one best strategy with lot sizes, breakevens, and risk warnings"""
        if not opportunities:
            return "NO STRATEGY - Insufficient data"
        
        # Get strikes and lot size info
        strikes = sorted(set(opp.strike_price for opp in opportunities))
        best_opp = max(opportunities, key=lambda x: x.confidence)
        lot_size = self._get_lot_size(best_opp.symbol)
        
        if "EXPIRY DAY" in market_sentiment:
            return "AVOID TRADING - Expiry day time decay"
        elif "SIDEWAYS" in market_sentiment:
            if len(strikes) >= 4:
                strikes_sorted = sorted(strikes)
                put_buy = strikes_sorted[0]    # Lowest strike
                put_sell = strikes_sorted[1]   # Second lowest
                call_sell = strikes_sorted[-2] # Second highest
                call_buy = strikes_sorted[-1]  # Highest strike
                
                # Calculate net credit (sell premium - buy premium)
                sell_premium = sum(opp.current_price for opp in opportunities if opp.strike_price in [put_sell, call_sell])
                buy_premium = sum(opp.current_price for opp in opportunities if opp.strike_price in [put_buy, call_buy])
                net_credit = sell_premium - buy_premium
                target_profit = net_credit * 0.5
                max_loss = min(put_sell - put_buy, call_buy - call_sell) - net_credit
                
                # Breakeven points
                upper_breakeven = call_buy + net_credit
                lower_breakeven = put_buy - net_credit
                
                return f"IRON CONDOR: BUY {put_buy} PUT + SELL {put_sell} PUT + SELL {call_sell} CALL + BUY {call_buy} CALL | Lot: {lot_size} | Credit: Rs{net_credit:.0f} | Target: Rs{target_profit:.0f} | Max Loss: Rs{max_loss:.0f} | Breakeven: {lower_breakeven:.0f}-{upper_breakeven:.0f} | HIGH MARGIN REQUIRED"
            elif len(strikes) >= 2:
                put_strike = min(strikes)
                call_strike = max(strikes)
                put_premium = next((opp.current_price for opp in opportunities if opp.strike_price == put_strike and opp.option_type == "Put"), 0)
                call_premium = next((opp.current_price for opp in opportunities if opp.strike_price == call_strike and opp.option_type == "Call"), 0)
                net_credit = put_premium + call_premium
                target_profit = net_credit * 0.5
                
                # Breakeven points for Short Strangle
                upper_breakeven = call_strike + net_credit
                lower_breakeven = put_strike - net_credit
                
                # Margin requirement (approximate)
                margin_req = max(call_strike * 0.15, put_strike * 0.15) * lot_size
                
                # Get expiry from opportunities
                expiry = next((opp.expiry_date for opp in opportunities), "")
                
                # Calculate max profit and loss
                max_profit = net_credit  # Maximum profit is the credit received
                target_50_percent = net_credit * 0.5  # 50% profit target
                stop_loss_200_percent = net_credit * 2  # 200% loss
                max_profit_per_lot = max_profit * lot_size
                max_loss_per_lot = stop_loss_200_percent * lot_size
                
                return f"SHORT STRANGLE: SELL {put_strike} PUT + SELL {call_strike} CALL | Expiry: {expiry} | Lot: {lot_size} | Credit: Rs{net_credit:.0f} | Target: Rs{target_50_percent:.0f} | Max Profit: Rs{max_profit_per_lot:.0f} | Stop Loss: Rs{stop_loss_200_percent:.0f} | Max Loss: Rs{max_loss_per_lot:.0f} | Breakeven: {lower_breakeven:.1f}-{upper_breakeven:.1f} | Margin: ~Rs{margin_req/1000:.0f}K"
            else:
                call_put_premium = best_opp.current_price * 2
                target_profit = call_put_premium * 0.5
                upper_breakeven = best_opp.strike_price + call_put_premium
                lower_breakeven = best_opp.strike_price - call_put_premium
                margin_req = best_opp.strike_price * 0.15 * lot_size
                # Calculate max profit and loss for straddle
                max_profit = call_put_premium  # Maximum profit is the credit received
                target_50_percent = call_put_premium * 0.5
                stop_loss_200_percent = call_put_premium * 2
                max_profit_per_lot = max_profit * lot_size
                max_loss_per_lot = stop_loss_200_percent * lot_size
                
                return f"SHORT STRADDLE: SELL {best_opp.strike_price} CALL+PUT | Expiry: {best_opp.expiry_date} | Lot: {lot_size} | Credit: Rs{call_put_premium:.0f} | Target: Rs{target_50_percent:.0f} | Max Profit: Rs{max_profit_per_lot:.0f} | Stop Loss: Rs{stop_loss_200_percent:.0f} | Max Loss: Rs{max_loss_per_lot:.0f} | Breakeven: {lower_breakeven:.0f}-{upper_breakeven:.0f} | Margin: ~Rs{margin_req/1000:.0f}K"
        elif "STRONG BEARISH" in market_sentiment:
            if len(strikes) >= 2:
                buy_strike = max(strikes)
                sell_strike = min(strikes)
                net_debit = best_opp.current_price * 0.7
                max_profit = buy_strike - sell_strike - net_debit
                return f"BEAR PUT SPREAD: BUY {buy_strike} PUT + SELL {sell_strike} PUT | Debit: Rs{net_debit:.0f} | Max Profit: Rs{max_profit:.0f} | SL: 50%"
            else:
                return f"LONG PUT: BUY {best_opp.strike_price} PUT | Entry: Rs{best_opp.current_price} | Target: Rs{best_opp.target} | SL: Rs{best_opp.stop_loss}"
        elif "STRONG BULLISH" in market_sentiment:
            if len(strikes) >= 2:
                buy_strike = min(strikes)
                sell_strike = max(strikes)
                net_debit = best_opp.current_price * 0.7
                max_profit = sell_strike - buy_strike - net_debit
                return f"BULL CALL SPREAD: BUY {buy_strike} CALL + SELL {sell_strike} CALL | Debit: Rs{net_debit:.0f} | Max Profit: Rs{max_profit:.0f} | SL: 50%"
            else:
                return f"LONG CALL: BUY {best_opp.strike_price} CALL | Entry: Rs{best_opp.current_price} | Target: Rs{best_opp.target} | SL: Rs{best_opp.stop_loss}"
        elif "BEARISH BIAS" in market_sentiment:
            if len(strikes) >= 2:
                buy_strike = max(strikes)
                sell_strike = min(strikes)
                buy_premium = next((opp.current_price for opp in opportunities if opp.strike_price == buy_strike), best_opp.current_price)
                sell_premium = next((opp.current_price for opp in opportunities if opp.strike_price == sell_strike), best_opp.current_price * 0.5)
                net_debit = buy_premium - sell_premium
                max_profit = buy_strike - sell_strike - net_debit
                max_loss = net_debit
                return f"BEAR PUT SPREAD: BUY {buy_strike} PUT + SELL {sell_strike} PUT | Entry Debit: Rs{net_debit:.0f} | Max Profit: Rs{max_profit:.0f} | Max Loss: Rs{max_loss:.0f}"
            else:
                return f"LONG PUT: BUY {best_opp.strike_price} PUT | Entry: Rs{best_opp.current_price} | Target: Rs{best_opp.target} | Max Loss: Rs{best_opp.current_price}"
        elif "BULLISH BIAS" in market_sentiment:
            if len(strikes) >= 2:
                buy_strike = min(strikes)
                sell_strike = max(strikes)
                buy_premium = next((opp.current_price for opp in opportunities if opp.strike_price == buy_strike), best_opp.current_price)
                sell_premium = next((opp.current_price for opp in opportunities if opp.strike_price == sell_strike), best_opp.current_price * 0.5)
                net_debit = buy_premium - sell_premium
                max_profit = sell_strike - buy_strike - net_debit
                max_loss = net_debit
                return f"BULL CALL SPREAD: BUY {buy_strike} CALL + SELL {sell_strike} CALL | Entry Debit: Rs{net_debit:.0f} | Max Profit: Rs{max_profit:.0f} | Max Loss: Rs{max_loss:.0f}"
            else:
                return f"LONG CALL: BUY {best_opp.strike_price} CALL | Entry: Rs{best_opp.current_price} | Target: Rs{best_opp.target} | Max Loss: Rs{best_opp.current_price}"
        else:
            call_put_cost = best_opp.current_price * 2
            breakeven_up = best_opp.strike_price + call_put_cost
            breakeven_down = best_opp.strike_price - call_put_cost
            return f"LONG STRADDLE: BUY {best_opp.strike_price} CALL+PUT | Entry Cost: Rs{call_put_cost:.0f} | Breakeven: {breakeven_down:.0f}-{breakeven_up:.0f} | Max Loss: Rs{call_put_cost:.0f}"
    
    def _prepare_analysis_data(self, opportunities: List[DerivativeOpportunity]) -> Dict:
        """Prepare analysis data once for both Telegram and WebSocket"""
        call_volume = sum(opp.volume for opp in opportunities if opp.option_type == "Call")
        put_volume = sum(opp.volume for opp in opportunities if opp.option_type == "Put")
        call_oi = sum(opp.open_interest for opp in opportunities if opp.option_type == "Call")
        put_oi = sum(opp.open_interest for opp in opportunities if opp.option_type == "Put")
        
        total_volume = sum(opp.volume for opp in opportunities)
        weighted_momentum = sum(opp.price_change * opp.volume for opp in opportunities) / total_volume if total_volume > 0 else 0
        
        total_liquidity = sum(opp.current_price * opp.volume for opp in opportunities)
        avg_liquidity = total_liquidity / len(opportunities) if opportunities else 0
        
        # Check if both calls and puts are falling
        falling_calls = sum(1 for opp in opportunities if opp.option_type == "Call" and opp.price_change < -10)
        falling_puts = sum(1 for opp in opportunities if opp.option_type == "Put" and opp.price_change < -10)
        total_calls = sum(1 for opp in opportunities if opp.option_type == "Call")
        total_puts = sum(1 for opp in opportunities if opp.option_type == "Put")
        
        # Market sentiment logic
        if (falling_calls > total_calls * 0.7 and falling_puts > total_puts * 0.7):
            market_sentiment = "SIDEWAYS (Theta Decay)"
        elif weighted_momentum < -15:
            market_sentiment = "STRONG BEARISH"
        elif weighted_momentum < -5:
            market_sentiment = "BEARISH BIAS"
        elif weighted_momentum > 15:
            market_sentiment = "STRONG BULLISH"
        elif weighted_momentum > 5:
            market_sentiment = "BULLISH BIAS"
        else:
            if call_volume > put_volume * 2 and call_oi > put_oi * 1.5:
                market_sentiment = "CAUTIOUS BULLISH"
            elif put_volume > call_volume * 2 and put_oi > call_oi * 1.5:
                market_sentiment = "CAUTIOUS BEARISH"
            else:
                market_sentiment = "NEUTRAL/MIXED"
        
        best_strategy = self._generate_best_strategy(opportunities, market_sentiment, weighted_momentum, call_volume, put_volume)
        
        print(f"Generated strategy: {best_strategy}")
        print(f"Market sentiment: {market_sentiment}")
        
        return {
            "market_sentiment": market_sentiment,
            "best_strategy": best_strategy,
            "call_volume": call_volume,
            "put_volume": put_volume,
            "call_oi": call_oi,
            "put_oi": put_oi,
            "weighted_momentum": round(weighted_momentum, 2),
            "avg_liquidity": avg_liquidity,
            "alert_count": len(opportunities),
            "opportunities": opportunities
        }
    
    def _get_lot_size(self, symbol: str) -> int:
        """Get lot size from Kotak API"""
        response = requests.get(
            "https://neo.kotaksecurities.com/api/masterscrip/expired-scrip-info",
            timeout=5
        )
        data = response.json().get("data", [])
        for item in data:
            if item.get("symbol") == symbol.upper():
                return int(item.get("lotSize"))
        return None
    
    def run_continuous_monitoring(self) -> bool:
        """Continuous monitoring for unexpected volume/price changes"""
        if not self.is_market_open():
            return False
        
        # Check cooldown for urgent analysis
        if (self.last_urgent_analysis and 
            time.time() - self.last_urgent_analysis < self.urgent_analysis_cooldown):
            return False
        
        try:
            # Quick data fetch for monitoring
            current_data = {
                'equity_snapshot': self.nse_client.get_derivatives_equity_snapshot(20)
            }
            
            # Check if urgent analysis needed
            if self.should_analyze_urgent(current_data):
                return self.run_intelligent_analysis(is_urgent=True)
            
            return False
            
        except Exception as e:
            print(f"Monitoring error: {e}")
            return False

def main():
    """Main function for standalone execution"""
    analyzer = IntelligentDerivativeAnalyzer()
    
    print("Starting Intelligent Derivative Analyzer")
    print("=" * 60)
    
    while True:
        try:
            analyzer.run_intelligent_analysis()
            
            # Wait 5 minutes before next check
            print("Waiting 5 minutes before next check...")
            time.sleep(300)
            
        except KeyboardInterrupt:
            print("\nStopping analyzer...")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()