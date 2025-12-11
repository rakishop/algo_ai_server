import asyncio
import json
from datetime import datetime
from real_news_fetcher import RealNewsFetcher
from ws.websocket_streaming import manager

class ExchangeAnnouncementMonitor:
    def __init__(self):
        self.fetcher = RealNewsFetcher()
        self.last_announcements = set()
        self.running = False
    
    async def check_new_announcements(self):
        """Check for new NSE and BSE announcements and broadcast via WebSocket"""
        try:
            total_new = 0
            
            # Get NSE announcements
            nse_announcements = self.fetcher.scrape_nse_announcements()
            if nse_announcements:
                total_new += await self._process_announcements(nse_announcements, "NSE")
            
            # Get BSE announcements
            bse_announcements = self.fetcher.scrape_bse_announcements()
            if bse_announcements:
                total_new += await self._process_announcements(bse_announcements, "BSE")
            
            return total_new
            
        except Exception as e:
            print(f"❌ NSE/BSE announcement check failed: {e}")
            return 0
    
    async def _process_announcements(self, announcements, source):
        """Process announcements for a specific source (NSE or BSE)"""
        current_announcements = set()
        new_announcements = []
        
        for announcement in announcements:
            # Create unique ID based on source, symbol and subject
            announcement_id = f"{source}-{announcement.get('symbol', announcement.get('company', ''))}-{announcement.get('subject', '')}"
            current_announcements.add(announcement_id)
            
            # Check if this is a new announcement
            if announcement_id not in self.last_announcements:
                announcement['source'] = source
                new_announcements.append(announcement)
        
        # Update last announcements for this source
        self.last_announcements = {aid for aid in self.last_announcements if not aid.startswith(f"{source}-")}
        self.last_announcements.update(current_announcements)
        
        # Broadcast new announcements via WebSocket
        if new_announcements and manager.active_connections:
            for announcement in new_announcements:
                websocket_message = {
                    "type": "exchange_announcement",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "title": announcement.get('title', ''),
                        "symbol": announcement.get('symbol', announcement.get('company', '')),
                        "subject": announcement.get('subject', ''),
                        "source": source,
                        "announcement_time": announcement.get('timestamp', ''),
                        "priority": "high" if any(keyword in announcement.get('subject', '').lower() 
                                               for keyword in ['dividend', 'bonus', 'split', 'merger', 'acquisition']) else "normal"
                    }
                }
                
                # Broadcast to all connected WebSocket clients
                success = await manager.broadcast_json(websocket_message)
                if success:
                    print(f"📢 {source} Announcement broadcasted: {announcement.get('symbol', announcement.get('company', ''))} - {announcement.get('subject', '')[:50]}...")
        
        return len(new_announcements)
    
    async def start_monitoring(self):
        """Start continuous monitoring of NSE announcements"""
        self.running = True
        print("🔍 Starting NSE & BSE announcement monitoring...")
        
        # Wait for WebSocket connections
        await asyncio.sleep(2)
        
        # Initial load - broadcast current announcements on startup
        try:
            # Process NSE announcements
            nse_initial = self.fetcher.scrape_nse_announcements()
            if nse_initial:
                new_count = await self._process_announcements(nse_initial, "NSE")
                print(f"📋 Broadcasted {new_count} NSE announcements on startup")
            
            # Process BSE announcements
            bse_initial = self.fetcher.scrape_bse_announcements()
            if bse_initial:
                new_count = await self._process_announcements(bse_initial, "BSE")
                print(f"📋 Broadcasted {new_count} BSE announcements on startup")
            
            print(f"📋 Total announcements loaded: {len(self.last_announcements)}")
        except Exception as e:
            print(f"⚠️ Initial announcement load failed: {e}")
        
        while self.running:
            try:
                new_count = await self.check_new_announcements()
                if new_count > 0:
                    print(f"✅ Found {new_count} new NSE/BSE announcements at {datetime.now().strftime('%H:%M:%S')}")
                
                # Check every 2 minutes
                await asyncio.sleep(120)
                
            except Exception as e:
                print(f"❌ NSE/BSE monitoring error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    def stop_monitoring(self):
        """Stop the monitoring"""
        self.running = False
        print("🛑 NSE & BSE announcement monitoring stopped")

# Global monitor instance
exchange_monitor = ExchangeAnnouncementMonitor()

async def start_exchange_announcement_monitor():
    """Function to start NSE & BSE announcement monitoring"""
    try:
        await exchange_monitor.start_monitoring()
    except asyncio.CancelledError:
        print("🛑 Exchange announcement monitor cancelled")
        exchange_monitor.stop_monitoring()
        raise
    except Exception as e:
        print(f"❌ Exchange announcement monitor error: {e}")
        exchange_monitor.stop_monitoring()

def stop_exchange_monitor():
    """Function to stop exchange monitoring"""
    exchange_monitor.stop_monitoring()