import requests
from config import settings

# Replace with your channel ID (starts with -100)
channel_id = "@your_channel_username"  # or "-1001234567890"

message = "🤖 Test message to channel from AI Stock Bot"
url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
data = {"chat_id": channel_id, "text": message}

response = requests.post(url, data=data)
result = response.json()

if result.get('ok'):
    print("✓ Message sent to channel")
else:
    print(f"✗ Failed: {result}")