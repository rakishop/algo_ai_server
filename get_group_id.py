import requests
from config import settings

# Get updates to find group ID
url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/getUpdates"
response = requests.get(url)
data = response.json()

print("Recent chats:")
for update in data.get('result', []):
    if 'message' in update:
        chat = update['message']['chat']
        print(f"Type: {chat['type']}, ID: {chat['id']}, Title: {chat.get('title', 'N/A')}")