import requests
from config import settings

# Try sending to a channel using @username format
channel_username = input("Enter your channel username (with @): ")

message = "🤖 Hello from AI Stock Bot! Channel setup successful."
url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
data = {"chat_id": channel_username, "text": message}

response = requests.post(url, data=data)
result = response.json()

if result.get('ok'):
    print(f"✓ Message sent to channel {channel_username}")
    print(f"Channel ID: {result['result']['chat']['id']}")
else:
    print(f"✗ Failed: {result.get('description', 'Unknown error')}")
    print("Make sure:")
    print("1. Bot is added as admin to the channel")
    print("2. Channel username is correct (with @)")
    print("3. Channel exists and is accessible")