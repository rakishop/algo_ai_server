import requests
from config import settings

# Set webhook URL for your bot
webhook_url = f"{settings.server_url}/api/v1/telegram/webhook"
bot_token = settings.telegram_bot_token

url = f"https://api.telegram.org/bot{bot_token}/setWebhook"
data = {"url": webhook_url}

response = requests.post(url, data=data)
result = response.json()

if result.get('ok'):
    print(f"✓ Webhook set successfully: {webhook_url}")
else:
    print(f"✗ Failed: {result}")
    print("Note: Webhook only works with HTTPS URLs in production")