import requests
from config import settings
from datetime import datetime

# Sample stock data
gainers = [
    {"symbol": "RELIANCE", "ltp": 2450.5, "pChange": 5.2},
    {"symbol": "TCS", "ltp": 3890.0, "pChange": 3.8},
    {"symbol": "INFY", "ltp": 1650.25, "pChange": 4.1},
    {"symbol": "HDFC", "ltp": 1580.75, "pChange": 2.9},
    {"symbol": "ICICIBANK", "ltp": 950.30, "pChange": 3.5}
]

losers = [
    {"symbol": "BAJFINANCE", "ltp": 6800.0, "pChange": -4.2},
    {"symbol": "MARUTI", "ltp": 10500.5, "pChange": -3.1},
    {"symbol": "ASIANPAINT", "ltp": 3200.25, "pChange": -2.8},
    {"symbol": "TITAN", "ltp": 3150.0, "pChange": -3.6}
]

message = f"🚀 AI BREAKOUT ALERT - {datetime.now().strftime('%H:%M')}\n\n"
message += "📈 TOP 5 BREAKOUT GAINERS\n"
for i, stock in enumerate(gainers[:5], 1):
    breakout_type = "Daily" if abs(stock['pChange']) > 4 else "15min"
    message += f"{i}. {stock['symbol']} | ₹{stock['ltp']:.1f} | +{stock['pChange']:.1f}% {breakout_type}\n"

message += "\n📉 TOP 4 BREAKOUT LOSERS\n"
for i, stock in enumerate(losers[:4], 1):
    breakout_type = "Daily" if abs(stock['pChange']) > 4 else "15min"
    message += f"{i}. {stock['symbol']} | ₹{stock['ltp']:.1f} | {stock['pChange']:.1f}% {breakout_type}\n"

message += "\n💡 Only breakout stocks with high volume"

url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
data = {"chat_id": settings.telegram_chat_id, "text": message}
response = requests.post(url, data=data)
print("✓ Stock alert sent" if response.json().get('ok') else "✗ Failed")