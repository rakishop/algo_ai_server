import requests
from config import settings
from datetime import datetime
from nse_client import NSEClient
from ai_stock_selector import AIStockSelector

# Get real market data
nse = NSEClient()
ai_selector = AIStockSelector()

# Fetch real data
gainers_data = nse.get_gainers_data()
losers_data = nse.get_losers_data()

# Get AI-selected stocks
top_gainers, top_losers = ai_selector.select_top_stocks(gainers_data, losers_data)

# Format message
message = f"🚀 AI BREAKOUT ALERT - {datetime.now().strftime('%H:%M')}\n\n"
message += "📈 TOP 5 BREAKOUT GAINERS\n"
for i, stock in enumerate(top_gainers[:5], 1):
    breakout_type = "Daily" if abs(stock.get('pChange', 0)) > 4 else "15min"
    message += f"{i}. {stock['symbol']} | ₹{stock['ltp']:.1f} | +{stock['pChange']:.1f}% {breakout_type}\n"

message += "\n📉 TOP 4 BREAKOUT LOSERS\n"
for i, stock in enumerate(top_losers[:4], 1):
    breakout_type = "Daily" if abs(stock.get('pChange', 0)) > 4 else "15min"
    message += f"{i}. {stock['symbol']} | ₹{stock['ltp']:.1f} | {stock['pChange']:.1f}% {breakout_type}\n"

message += "\n💡 Only breakout stocks with high volume"

# Send to Telegram
url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
data = {"chat_id": settings.telegram_chat_id, "text": message}
response = requests.post(url, data=data)
print("✓ Real stock alert sent" if response.json().get('ok') else "✗ Failed")