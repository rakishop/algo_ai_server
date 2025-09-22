from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
from datetime import datetime
from typing import Dict, List
from nse_client import NSEClient

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[str, List[WebSocket]] = {}
        self.nse_client = NSEClient()
        self.last_data_hash = None
        
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
    
    async def start_market_stream(self):
        """Stream live market data every 30 seconds"""
        try:
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
            raise

manager = WebSocketManager()