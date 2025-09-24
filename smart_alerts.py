import json
from datetime import datetime, timedelta
from nse_client import NSEClient
from config import settings
import requests

class SmartAlerts:
    def __init__(self):
        self.nse = NSEClient()
        
    def breakout_alerts(self):
        """52-week high/low breakout alerts"""
        high_data = self.nse.get_52week_high_stocks_data()
        stocks = high_data.get('data', [])[:5]
        
        if stocks:
            message = f"🚀 BREAKOUT ALERT - {datetime.now().strftime('%H:%M')}\n\n"
            for i, stock in enumerate(stocks, 1):
                message += f"{i}. {stock.get('symbol')} - NEW 52W HIGH\n"
                message += f"   💰 ₹{stock.get('ltp', 0):.1f} (+{stock.get('perChange', 0):.1f}%)\n\n"
            
            self.send_telegram(message)
    
    def unusual_options_activity(self):
        """Detect unusual options volume/OI"""
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
        """Overall market sentiment based on advance/decline"""
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