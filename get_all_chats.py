import requests
from config import settings

def get_all_chats():
    """Get all channels and groups the bot has access to"""
    try:
        # Get recent updates to see all chats
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/getUpdates"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            chats = {}
            
            print("ALL ACCESSIBLE CHATS:")
            print("=" * 50)
            
            for update in data.get('result', []):
                chat = None
                
                # Check different message types
                if 'message' in update:
                    chat = update['message']['chat']
                elif 'channel_post' in update:
                    chat = update['channel_post']['chat']
                elif 'edited_message' in update:
                    chat = update['edited_message']['chat']
                elif 'edited_channel_post' in update:
                    chat = update['edited_channel_post']['chat']
                
                if chat:
                    chat_id = chat['id']
                    if chat_id not in chats:
                        chats[chat_id] = {
                            'id': chat_id,
                            'title': chat.get('title', chat.get('first_name', 'Unknown')),
                            'type': chat['type'],
                            'username': chat.get('username', 'No username')
                        }
            
            # Display all unique chats
            for chat_id, info in chats.items():
                chat_type = info['type'].upper()
                title = info['title']
                username = f"@{info['username']}" if info['username'] != 'No username' else 'No username'
                
                print(f"{chat_type}: {title}")
                print(f"  ID: {chat_id}")
                print(f"  Username: {username}")
                print("-" * 30)
            
            print(f"\nTotal chats found: {len(chats)}")
            
            # Also try to get bot info
            bot_url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/getMe"
            bot_response = requests.get(bot_url)
            if bot_response.status_code == 200:
                bot_info = bot_response.json()
                if bot_info.get('ok'):
                    bot_data = bot_info['result']
                    print(f"\nBot Info:")
                    print(f"  Name: {bot_data.get('first_name', 'Unknown')}")
                    print(f"  Username: @{bot_data.get('username', 'Unknown')}")
                    print(f"  ID: {bot_data.get('id', 'Unknown')}")
            
            return chats
            
        else:
            print(f"Failed to get updates: HTTP {response.status_code}")
            return {}
            
    except Exception as e:
        print(f"Error: {e}")
        return {}

if __name__ == "__main__":
    get_all_chats()