import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_all_chat_ids():
    """Get all chat IDs from Telegram bot"""
    
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    
    if not bot_token:
        print("TELEGRAM_BOT_TOKEN not found in .env")
        return
    
    # Get updates from Telegram
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    
    try:
        response = requests.get(url)
        data = response.json()
        print(data)
        if not data.get('ok'):
            print(f"Error: {data}")
            return
        
        chat_ids = set()
        
       
        
    except Exception as e:
        print(f"Error getting chat IDs: {e}")

if __name__ == "__main__":
    get_all_chat_ids()