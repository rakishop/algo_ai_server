import requests
from config import settings

# Get updates to find channel ID
url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/getUpdates"
response = requests.get(url)
data = response.json()

print("Recent chats and channels:")
channel_id = None
for update in data.get('result', []):
    if 'channel_post' in update:
        chat = update['channel_post']['chat']
        channel_id = chat['id']
        print(f"CHANNEL: {chat['title']} | ID: {chat['id']}")
    elif 'message' in update:
        chat = update['message']['chat']
        print(f"{chat['type'].upper()}: {chat.get('title', chat.get('first_name', 'N/A'))} | ID: {chat['id']}")

# Update .env file if channel found
if channel_id:
    print(f"\nUpdating .env with channel ID: {channel_id}")
    
    # Read current .env
    with open('.env', 'r') as f:
        lines = f.readlines()
    
    # Update TELEGRAM_CHAT_ID
    with open('.env', 'w') as f:
        for line in lines:
            if line.startswith('TELEGRAM_CHAT_ID='):
                f.write(f'TELEGRAM_CHAT_ID={channel_id}\n')
            else:
                f.write(line)
    
    print("✓ .env updated with channel ID")
else:
    print("\nNo channel found. Create a channel and add your bot as admin first.")