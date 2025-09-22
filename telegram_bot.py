import requests
import asyncio
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class TelegramBot:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    async def send_message(self, message: str):
        try:
            url = f"{self.base_url}/sendMessage"
            data = {"chat_id": self.chat_id, "text": message}
            response = requests.post(url, data=data)
            return response.json()
        except Exception as e:
            print(f"Telegram error: {e}")
            return None

    def format_stock_alert(self, gainers, losers):
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
        return message

# Initialize bot from environment variables
telegram_bot = TelegramBot(
    os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN"),
    os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")
)