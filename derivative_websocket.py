import websocket
import json
import threading
import time
from datetime import datetime
from intelligent_derivative_analyzer import IntelligentDerivativeAnalyzer

class IntelligentDerivativeWebSocket:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.ws = None
        self.analyzer = IntelligentDerivativeAnalyzer()
        self.last_analysis = time.time()
        self.analysis_interval = 900  # 15 minutes
        
    def on_message(self, ws, message):
        """Process WebSocket message and trigger intelligent analysis if needed"""
        try:
            data = json.loads(message)
            current_time = time.time()
            
            # Only run intelligent analysis every 15 minutes during market hours
            if (current_time - self.last_analysis >= self.analysis_interval and 
                self.analyzer.is_market_open()):
                
                print(f"🧠 WebSocket triggered intelligent analysis at {datetime.now().strftime('%H:%M:%S')}")
                self.run_intelligent_analysis()
                self.last_analysis = current_time
            
        except Exception as e:
            print(f"⚠️ WebSocket message processing error: {e}")
    
    def run_intelligent_analysis(self):
        """Run comprehensive intelligent analysis"""
        try:
            result = self.analyzer.run_intelligent_analysis()
            if result:
                print("✅ WebSocket: Intelligent analysis completed and notification sent")
            else:
                print("📊 WebSocket: Analysis completed - no new opportunities")
        except Exception as e:
            print(f"❌ WebSocket: Intelligent analysis failed: {e}")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        print(f"❌ WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        print(f"🔌 WebSocket connection closed: {close_status_code} - {close_msg}")
    
    def on_open(self, ws):
        """Handle WebSocket open"""
        print(f"🔗 Intelligent Derivative WebSocket connected at {datetime.now().strftime('%H:%M:%S')}")
        print("🧠 Will run intelligent analysis every 15 minutes during market hours")
    
    def connect(self, ws_url):
        """Connect to WebSocket with intelligent analysis"""
        try:
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            self.ws.run_forever()
        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
    
    def start_background_analysis(self):
        """Start background intelligent analysis thread"""
        def background_worker():
            while True:
                try:
                    if self.analyzer.should_analyze():
                        print("🔄 Background: Running scheduled intelligent analysis")
                        self.run_intelligent_analysis()
                    
                    # Check every 5 minutes
                    time.sleep(300)
                    
                except Exception as e:
                    print(f"❌ Background analysis error: {e}")
                    time.sleep(60)
        
        thread = threading.Thread(target=background_worker, daemon=True)
        thread.start()
        print("🚀 Background intelligent analysis thread started")

# Backward compatibility
class DerivativeWebSocket(IntelligentDerivativeWebSocket):
    """Legacy class for backward compatibility"""
    def __init__(self, bot_token, chat_id):
        super().__init__(bot_token, chat_id)
        print("⚠️ Using legacy DerivativeWebSocket - consider upgrading to IntelligentDerivativeWebSocket")

if __name__ == "__main__":
    # Standalone execution for testing
    from config import settings
    
    if settings.telegram_bot_token:
        intelligent_ws = IntelligentDerivativeWebSocket(
            settings.telegram_bot_token, 
            settings.telegram_chat_id or "-1002981590794"
        )
        
        # Start background analysis
        intelligent_ws.start_background_analysis()
        
        print("🧠 Intelligent Derivative WebSocket ready")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n🛑 Stopping intelligent derivative analysis...")
    else:
        print("❌ Telegram bot token not configured")

# Usage examples:
# intelligent_ws = IntelligentDerivativeWebSocket("YOUR_BOT_TOKEN", "YOUR_CHAT_ID")
# intelligent_ws.start_background_analysis()  # Start background analysis
# intelligent_ws.connect("wss://your-derivative-websocket-url")  # Connect to WebSocket

# For standalone intelligent analysis without WebSocket:
# analyzer = IntelligentDerivativeAnalyzer()
# analyzer.run_intelligent_analysis()