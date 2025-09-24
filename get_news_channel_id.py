import requests
from config import settings

def get_channel_id():
    """Get channel ID for MyAlgoFax NEWS channel"""
    try:
        # First, send a test message to get channel info
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/getUpdates"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            print("Recent chats:")
            
            for update in data.get('result', []):
                if 'message' in update:
                    chat = update['message']['chat']
                    print(f"Chat: {chat.get('title', 'No title')} | ID: {chat['id']} | Type: {chat['type']}")
                elif 'channel_post' in update:
                    chat = update['channel_post']['chat']
                    print(f"Channel: {chat.get('title', 'No title')} | ID: {chat['id']} | Type: {chat['type']}")
        
        # Test with common channel username
        test_channels = ["@MyAlgoFaxNews", "@myalgofaxnews", "@MyAlgoFax_News"]
        
        for channel in test_channels:
            try:
                test_url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
                test_data = {"chat_id": channel, "text": "Test message - getting channel ID"}
                test_response = requests.post(test_url, data=test_data)
                
                if test_response.status_code == 200:
                    result = test_response.json()
                    if result.get('ok'):
                        chat_id = result['result']['chat']['id']
                        print(f"Found channel: {channel} = {chat_id}")
                        return chat_id
                    else:
                        print(f"Failed {channel}: {result.get('description', 'Failed')}")
                else:
                    print(f"Failed {channel}: HTTP {test_response.status_code}")
            except Exception as e:
                print(f"Failed {channel}: {e}")
        
        return None
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    print("Finding MyAlgoFax NEWS channel ID...")
    channel_id = get_channel_id()
    
    if channel_id:
        print(f"\nUse this ID: {channel_id}")
        print(f"Add to .env: TELEGRAM_NEWS_CHANNEL_ID={channel_id}")
    else:
        print("\nChannel not found. Make sure:")
        print("1. Bot is added to the channel as admin")
        print("2. Channel username is correct")
        print("3. Channel is public or bot has access")