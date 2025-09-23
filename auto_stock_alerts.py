import requests
import time
import json
import os
from config import settings
from datetime import datetime
from nse_client import NSEClient

# File to store previous stocks
PREVIOUS_STOCKS_FILE = "previous_stocks.json"

def load_previous_stocks():
    """Load previously detected stocks"""
    try:
        if os.path.exists(PREVIOUS_STOCKS_FILE):
            with open(PREVIOUS_STOCKS_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return {"gainers": [], "losers": []}

def save_current_stocks(gainers, losers):
    """Save current stocks for next comparison"""
    try:
        data = {
            "gainers": [stock['symbol'] for stock in gainers],
            "losers": [stock['symbol'] for stock in losers],
            "timestamp": datetime.now().isoformat()
        }
        with open(PREVIOUS_STOCKS_FILE, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error saving stocks: {e}")

def find_new_stocks(current_stocks, previous_stocks):
    """Find new stocks not in previous list"""
    current_symbols = [stock['symbol'] for stock in current_stocks]
    return [stock for stock in current_stocks if stock['symbol'] not in previous_stocks]

def is_market_open():
    """Check if market is open (9:15 AM to 3:30 PM on weekdays)"""
    now = datetime.now()
    
    # Skip weekends
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    
    # Market hours: 9:15 AM to 3:30 PM
    market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_start <= now <= market_end

def send_stock_alert():
    try:
        # Check if market is open first
        if not is_market_open():
            print(f"Market closed at {datetime.now().strftime('%H:%M:%S')} - skipping check")
            return False
        
        print(f"Checking for new stocks at {datetime.now().strftime('%H:%M:%S')}")
        
        nse = NSEClient()
        active_data = nse.get_most_active_securities()
        
        if not active_data or not active_data.get('data'):
            print("No market data available")
            return False
        
        # Get current top stocks
        current_gainers = [stock for stock in active_data.get('data', []) if stock.get('pChange', 0) > 2]
        current_gainers = sorted(current_gainers, key=lambda x: x.get('pChange', 0), reverse=True)[:10]
        
        current_losers = [stock for stock in active_data.get('data', []) if stock.get('pChange', 0) < -2]
        current_losers = sorted(current_losers, key=lambda x: x.get('pChange', 0))[:10]
        
        # Load previous stocks
        previous_data = load_previous_stocks()
        
        # Find new stocks
        new_gainers = find_new_stocks(current_gainers, previous_data.get('gainers', []))
        new_losers = find_new_stocks(current_losers, previous_data.get('losers', []))
        
        # Only send alert if new stocks found
        if not new_gainers and not new_losers:
            print("No new stocks detected - skipping alert")
            # Still save current stocks for next comparison
            save_current_stocks(current_gainers[:5], current_losers[:4])
            return False
        
        print(f"New stocks found: {len(new_gainers)} gainers, {len(new_losers)} losers")
        
        # Prepare message with new stocks only
        message = f"NEW STOCK ALERT - {datetime.now().strftime('%H:%M')}\n\n"
        
        if new_gainers:
            message += f"NEW GAINERS ({len(new_gainers)})\n"
            for i, stock in enumerate(new_gainers[:5], 1):
                message += f"{i}. {stock['symbol']} | Rs{stock['lastPrice']:.1f} | +{stock['pChange']:.1f}%\n"
        
        if new_losers:
            message += f"\nNEW LOSERS ({len(new_losers)})\n"
            for i, stock in enumerate(new_losers[:4], 1):
                message += f"{i}. {stock['symbol']} | Rs{stock['lastPrice']:.1f} | {stock['pChange']:.1f}%\n"
        
        message += "\nOnly new breakout stocks detected"
        
        # Send to Telegram
        chat_id = settings.telegram_chat_id or "-1002981590794"
        
        if not settings.telegram_bot_token:
            print("Telegram bot token not configured")
            return False
        
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
        data = {"chat_id": chat_id, "text": message}
        
        print(f"Sending new stock alert to Telegram")
        response = requests.post(url, data=data, timeout=10)
        
        if response.status_code == 200 and response.json().get('ok'):
            print(f"New stock alert sent at {datetime.now().strftime('%H:%M:%S')}")
            # Save current stocks after successful alert
            save_current_stocks(current_gainers[:5], current_losers[:4])
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