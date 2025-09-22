import requests
import time
from config import settings
from datetime import datetime
from nse_client import NSEClient

def send_stock_alert():
    try:
        nse = NSEClient()
        active_data = nse.get_most_active_securities()
        
        gainers = [stock for stock in active_data.get('data', []) if stock.get('pChange', 0) > 0]
        gainers = sorted(gainers, key=lambda x: x.get('pChange', 0), reverse=True)[:5]
        
        losers = [stock for stock in active_data.get('data', []) if stock.get('pChange', 0) < 0]
        losers = sorted(losers, key=lambda x: x.get('pChange', 0))[:4]
        
        message = f"🚀 AI BREAKOUT ALERT - {datetime.now().strftime('%H:%M')}\n\n"
        message += "📈 TOP 5 BREAKOUT GAINERS\n"
        for i, stock in enumerate(gainers, 1):
            breakout_type = "Daily" if abs(stock.get('pChange', 0)) > 4 else "15min"
            message += f"{i}. {stock['symbol']} | ₹{stock['lastPrice']:.1f} | +{stock['pChange']:.1f}% {breakout_type}\n"
        
        message += "\n📉 TOP 4 BREAKOUT LOSERS\n"
        for i, stock in enumerate(losers, 1):
            breakout_type = "Daily" if abs(stock.get('pChange', 0)) > 4 else "15min"
            message += f"{i}. {stock['symbol']} | ₹{stock['lastPrice']:.1f} | {stock['pChange']:.1f}% {breakout_type}\n"
        
        message += "\n💡 Only breakout stocks with high volume"
        
        # Send to group
        group_id = "-4915028182"
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
        data = {"chat_id": group_id, "text": message}
        response = requests.post(url, data=data)
        
        status = "✓ Sent" if response.json().get('ok') else "✗ Failed"
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {status}")
        
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Error: {e}")

print("Starting auto stock alerts every 30 minutes...")
print("Press Ctrl+C to stop")

while True:
    send_stock_alert()
    time.sleep(1800)  # 30 minutes = 1800 seconds