import requests
import time
from config import settings
from datetime import datetime
from nse_client import NSEClient

def send_stock_alert():
    try:
        print(f"Starting stock alert at {datetime.now().strftime('%H:%M:%S')}")
        
        nse = NSEClient()
        active_data = nse.get_most_active_securities()
        
        if not active_data or not active_data.get('data'):
            print("No market data available")
            return False
        
        gainers = [stock for stock in active_data.get('data', []) if stock.get('pChange', 0) > 0]
        gainers = sorted(gainers, key=lambda x: x.get('pChange', 0), reverse=True)[:5]
        
        losers = [stock for stock in active_data.get('data', []) if stock.get('pChange', 0) < 0]
        losers = sorted(losers, key=lambda x: x.get('pChange', 0))[:4]
        
        if not gainers and not losers:
            print("No significant price movements found")
            return False
        
        message = f"AI BREAKOUT ALERT - {datetime.now().strftime('%H:%M')}\n\n"
        
        if gainers:
            message += "TOP 5 BREAKOUT GAINERS\n"
            for i, stock in enumerate(gainers, 1):
                breakout_type = "Daily" if abs(stock.get('pChange', 0)) > 4 else "15min"
                message += f"{i}. {stock['symbol']} | Rs{stock['lastPrice']:.1f} | +{stock['pChange']:.1f}% {breakout_type}\n"
        
        if losers:
            message += "\nTOP 4 BREAKOUT LOSERS\n"
            for i, stock in enumerate(losers, 1):
                breakout_type = "Daily" if abs(stock.get('pChange', 0)) > 4 else "15min"
                message += f"{i}. {stock['symbol']} | Rs{stock['lastPrice']:.1f} | {stock['pChange']:.1f}% {breakout_type}\n"
        
        message += "\nOnly breakout stocks with high volume"
        
        # Use chat_id from settings
        chat_id = settings.telegram_chat_id or "-1002981590794"
        
        if not settings.telegram_bot_token:
            print("Telegram bot token not configured")
            return False
        
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
        data = {"chat_id": chat_id, "text": message}
        
        print(f"Sending to Telegram chat: {chat_id}")
        response = requests.post(url, data=data, timeout=10)
        
        if response.status_code == 200 and response.json().get('ok'):
            print(f"Alert sent successfully at {datetime.now().strftime('%H:%M:%S')}")
            return True
        else:
            print(f"Failed to send alert: {response.text}")
            return False
        
    except Exception as e:
        print(f"Error in send_stock_alert: {e}")
        return False

# This function is now called by the scheduler in main.py
# To run standalone for testing:
if __name__ == "__main__":
    print("Starting auto stock alerts every 30 minutes...")
    print("Press Ctrl+C to stop")
    
    while True:
        try:
            send_stock_alert()
            print("Next alert in 30 minutes...")
            time.sleep(1800)  # 30 minutes = 1800 seconds
        except KeyboardInterrupt:
            print("\nStopping alerts...")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            time.sleep(60)  # Wait 1 minute before retrying