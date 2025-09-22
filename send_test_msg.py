import requests
from config import settings

# Send test message to current chat/group
message = "🤖 Test message from AI Stock Bot"
url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
data = {"chat_id": settings.telegram_chat_id, "text": message}

response = requests.post(url, data=data)
result = response.json()

if result.get('ok'):
    print("✓ Test message sent successfully")
else:
    print(f"✗ Failed: {result}")