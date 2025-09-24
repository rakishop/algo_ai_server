import requests
import time
import threading
from datetime import datetime

class KeepAlive:
    def __init__(self, url):
        self.url = url
        self.running = False
        
    def ping_server(self):
        """Ping server to keep it alive"""
        while self.running:
            try:
                response = requests.get(f"{self.url}/", timeout=30)
                print(f"✅ Keep-alive ping: {response.status_code} at {datetime.now().strftime('%H:%M:%S')}")
            except Exception as e:
                print(f"❌ Keep-alive failed: {e}")
            
            time.sleep(840)  # 14 minutes (Render sleeps after 15min)
    
    def start(self):
        """Start keep-alive service"""
        self.running = True
        thread = threading.Thread(target=self.ping_server, daemon=True)
        thread.start()
        print(f"🔄 Keep-alive started for {self.url}")
        
    def stop(self):
        """Stop keep-alive service"""
        self.running = False