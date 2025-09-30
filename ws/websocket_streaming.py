from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
from datetime import datetime, time
from typing import Dict, List
from nse_client import NSEClient
import os
from dotenv import load_dotenv
load_dotenv()

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[str, List[WebSocket]] = {}
        self.nse_client = NSEClient()
        self.last_data_hash = None
        self.derivatives_task = None
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_NEWS_CHANNEL_ID')
        self.pending_messages = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        # Remove from all subscriptions
        for symbol in list(self.subscriptions.keys()):
            if websocket in self.subscriptions[symbol]:
                self.subscriptions[symbol].remove(websocket)
                if not self.subscriptions[symbol]:
                    del self.subscriptions[symbol]
    
    async def subscribe_symbol(self, websocket: WebSocket, symbol: str):
        if symbol not in self.subscriptions:
            self.subscriptions[symbol] = []
        if websocket not in self.subscriptions[symbol]:
            self.subscriptions[symbol].append(websocket)
    
    async def broadcast_to_symbol(self, symbol: str, data: dict):
        if symbol in self.subscriptions:
            for websocket in self.subscriptions[symbol][:]:
                try:
                    await websocket.send_text(json.dumps(data))
                except:
                    self.subscriptions[symbol].remove(websocket)
    
    async def broadcast_json(self, data: dict) -> bool:
        """Broadcast JSON data to all connected WebSocket clients"""
        if not self.active_connections:
            return False
        
        success_count = 0
        for websocket in self.active_connections[:]:
            try:
                await websocket.send_text(json.dumps(data))
                success_count += 1
            except Exception as e:
                print(f"Failed to send to websocket: {e}")
                self.active_connections.remove(websocket)
        
        return success_count > 0
    
    def is_market_open(self):
        """Check if market is open (9:15 AM to 3:30 PM IST)"""
        now = datetime.now()
        current_time = now.time()
        weekday = now.weekday()
        
        # Monday=0, Sunday=6
        if weekday >= 5:  # Saturday or Sunday
            return False
            
        market_open = time(9, 15)
        market_close = time(15, 30)
        
        return market_open <= current_time <= market_close
    
    async def broadcast_derivatives_analysis(self):
        """Broadcast derivatives and options analysis every 3 minutes during market hours"""
        try:
            while True:
                if self.is_market_open() and self.active_connections:
                    try:
                        # Call method directly
                        try:
                            from routes.ml_derivatives_model import DerivativesMLModel
                            ml_model = DerivativesMLModel()
                            
                            # Get derivatives data and train model
                            derivatives_data = self.nse_client.get_derivatives_equity_snapshot(30)
                            if "error" not in derivatives_data:
                                training_result = ml_model.train_with_historical_data(derivatives_data)
                                recommendations = ml_model.predict_recommendations(derivatives_data)
                            else:
                                recommendations = derivatives_data
                        except Exception as e:
                            recommendations = {"error": f"Method call failed: {str(e)}"}
                        
                            # Extract options analysis from recommendations
                            options_analysis = None
                            if "error" not in recommendations and "top_3_recommendations" in recommendations:
                                options_analysis = {
                                    "best_trades_summary": recommendations.get("top_3_recommendations", [])[:5],
                                    "market_sentiment": recommendations.get("market_sentiment", "NEUTRAL"),
                                    "confidence_level": recommendations.get("confidence_level", "MEDIUM"),
                                    "model_status": recommendations.get("model_status", "UNKNOWN")
                                }
                            
                            # Broadcast to all connected clients
                            message = {
                                "type": "derivatives_analysis",
                                "timestamp": datetime.now().isoformat(),
                                "derivatives_recommendations": recommendations,
                                "options_analysis": options_analysis
                            }
                            
                            for websocket in self.active_connections[:]:
                                try:
                                    await websocket.send_text(json.dumps(message))
                                except:
                                    self.active_connections.remove(websocket)
                            
                            print(f"✅ Derivatives analysis broadcasted at {datetime.now().strftime('%H:%M:%S')}")
                            
                            # Send any pending messages from intelligent analyzer
                            if self.pending_messages:
                                for pending_msg in self.pending_messages:
                                    for websocket in self.active_connections[:]:
                                        try:
                                            await websocket.send_text(json.dumps(pending_msg))
                                        except:
                                            self.active_connections.remove(websocket)
                                print(f"✅ Sent {len(self.pending_messages)} pending derivative alerts")
                                self.pending_messages.clear()
                            
                            # Send to Telegram using existing handler
                            if "error" not in recommendations and "top_3_recommendations" in recommendations:
                                from telegram_handler import TelegramHandler
                                tg = TelegramHandler()
                                msg = f"🔔 Derivative Alert\n{recommendations['top_3_recommendations'][0]['recommendation']}\n{datetime.now().strftime('%H:%M:%S')}"
                                tg.send_message(self.chat_id, msg)
                            
                    
                    except Exception as e:
                        print(f"❌ Derivatives analysis error: {e}")
                
                await asyncio.sleep(180)  # 3 minutes
                
        except asyncio.CancelledError:
            print("Derivatives analysis cancelled")
            raise
    
    async def start_market_stream(self):
        """Stream live market data every 30 seconds"""
        try:
            # Start derivatives analysis task
            self.derivatives_task = asyncio.create_task(self.broadcast_derivatives_analysis())
            
            while True:
                try:
                    # Get live market data
                    active_data = self.nse_client.get_most_active_securities()
                    gainers_data = self.nse_client.get_gainers_data()
                    
                    # Check if data changed (ignore timestamps)
                    def extract_core_data(data):
                        if not data or 'data' not in data:
                            return []
                        return [(item.get('symbol'), item.get('ltp'), item.get('pChange')) 
                               for item in data['data'][:10]]
                    
                    current_data = (extract_core_data(active_data), extract_core_data(gainers_data))
                    if current_data == self.last_data_hash:
                        await asyncio.sleep(30)
                        continue
                    
                    self.last_data_hash = current_data
                    timestamp = datetime.now().isoformat()
                    
                    # Broadcast only if data changed
                    for websocket in self.active_connections[:]:
                        try:
                            await websocket.send_text(json.dumps({
                                "type": "market_update",
                                "timestamp": timestamp,
                                "active_securities": active_data,
                                "gainers": gainers_data
                            }))
                        except:
                            self.active_connections.remove(websocket)
                    
                    await asyncio.sleep(30)  # Update every 30 seconds
                except Exception as e:
                    print(f"Stream error: {e}")
                    await asyncio.sleep(60)
        except asyncio.CancelledError:
            print("Market stream cancelled")
            if self.derivatives_task:
                self.derivatives_task.cancel()
            raise

manager = WebSocketManager()