import json
from datetime import datetime, timedelta
from nse_client import NSEClient
from config import settings
import requests

class SmartAlerts:
    def __init__(self):
        self.nse = NSEClient()
        self.sent_alerts_file = "sent_alerts.json"
        
    def load_sent_alerts(self):
        """Load previously sent alerts"""
        try:
            with open(self.sent_alerts_file, 'r') as f:
                return json.load(f)
        except:
            return {"breakouts": [], "timestamp": datetime.now().isoformat()}
    
    def save_sent_alerts(self, alerts_data):
        """Save sent alerts"""
        try:
            with open(self.sent_alerts_file, 'w') as f:
                json.dump(alerts_data, f)
        except Exception as e:
            print(f"Error saving alerts: {e}")
        
    def is_market_open(self):
        """Check if market is open (9:15 AM to 3:30 PM on weekdays)"""
        now = datetime.now()
        
        # Skip weekends
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        # Market hours: 9:00 AM to 3:30 PM
        market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_start <= now <= market_end
    
    def breakout_alerts(self):
        """52-week high/low breakout alerts - only during market hours"""
        if not self.is_market_open():
            return
            
        high_data = self.nse.get_52week_high_stocks_data()
        stocks = high_data.get('data', [])[:5]
        
        if not stocks:
            return
            
        # Load previously sent alerts
        sent_data = self.load_sent_alerts()
        sent_breakouts = set(sent_data.get('breakouts', []))
        
        # Filter new breakouts only
        new_breakouts = []
        for stock in stocks:
            symbol = stock.get('symbol')
            if symbol and symbol not in sent_breakouts:
                new_breakouts.append(stock)
                sent_breakouts.add(symbol)
        
        if new_breakouts:
            message = f"🚀 BREAKOUT ALERT - {datetime.now().strftime('%H:%M')}\n\n"
            for i, stock in enumerate(new_breakouts, 1):
                message += f"{i}. {stock.get('symbol')} - NEW 52W HIGH\n"
                message += f"   💰 ₹{stock.get('ltp', 0):.1f} (+{stock.get('perChange', 0):.1f}%)\n\n"
            
            self.send_telegram(message)
            
            # Save updated alerts
            sent_data['breakouts'] = list(sent_breakouts)
            sent_data['timestamp'] = datetime.now().isoformat()
            self.save_sent_alerts(sent_data)
    
    def unusual_options_activity(self):
        """Detect unusual options volume/OI - only during market hours"""
        if not self.is_market_open():
            return
            
        derivatives = self.nse.get_derivatives_snapshot()
        if not derivatives.get('data'):
            return
            
        unusual = []
        for option in derivatives['data'][:20]:
            volume = option.get('numberOfContractsTraded', 0)
            oi_change = option.get('changeinOpenInterest', 0)
            
            if volume > 50000 or abs(oi_change) > 10000:
                unusual.append(option)
        
        if unusual:
            message = f"⚡ UNUSUAL OPTIONS - {datetime.now().strftime('%H:%M')}\n\n"
            for i, opt in enumerate(unusual[:3], 1):
                message += f"{i}. {opt.get('underlying')} {opt.get('strikePrice')} {opt.get('optionType')}\n"
                message += f"   📊 Vol: {opt.get('numberOfContractsTraded', 0):,}\n"
                message += f"   🔄 OI Δ: {opt.get('changeinOpenInterest', 0):+,}\n\n"
            
            self.send_telegram(message)
    
    def market_sentiment_alert(self):
        """Overall market sentiment based on advance/decline - only during market hours"""
        if not self.is_market_open():
            return
            
        advance_data = self.nse.get_advance_decline()
        if not advance_data.get('advances'):
            return
            
        advances = advance_data.get('advances', 0)
        declines = advance_data.get('declines', 0)
        
        if advances + declines > 0:
            advance_ratio = advances / (advances + declines)
            
            if advance_ratio > 0.75:
                sentiment = "🟢 BULLISH"
            elif advance_ratio < 0.25:
                sentiment = "🔴 BEARISH"
            else:
                return  # Neutral, no alert
            
            message = f"📊 MARKET SENTIMENT - {datetime.now().strftime('%H:%M')}\n\n"
            message += f"{sentiment}\n"
            message += f"Advances: {advances} | Declines: {declines}\n"
            message += f"Ratio: {advance_ratio:.1%}"
            
            self.send_telegram(message)
    
    def send_telegram(self, message):
        """Send message to Telegram"""
        try:
            url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
            data = {"chat_id": settings.telegram_chat_id, "text": message}
            requests.post(url, data=data, timeout=10)
        except Exception as e:
            print(f"Alert send failed: {e}")

def run_smart_alerts():
    """Function for scheduler"""
    alerts = SmartAlerts()
    alerts.breakout_alerts()
    alerts.unusual_options_activity()
    alerts.market_sentiment_alert()