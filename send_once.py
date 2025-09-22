import requests
from config import settings

url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
data = {"chat_id": settings.telegram_chat_id, "text": "🚀 AI BREAKOUT ALERT - Test Message"}
response = requests.post(url, data=data)
print("✓ Sent" if response.json().get('ok') else "✗ Failed")