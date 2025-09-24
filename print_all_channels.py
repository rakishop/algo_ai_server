import os
from dotenv import load_dotenv

load_dotenv()

def print_all_channels():
    """Print all Telegram channel IDs from environment"""
    
    print("=== TELEGRAM CHANNELS ===")
    
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    news_channel_id = os.getenv('TELEGRAM_NEWS_CHANNEL_ID')
    
    print(f"BOT_TOKEN: {bot_token}")
    print(f"CHAT_ID: {chat_id}")
    print(f"NEWS_CHANNEL_ID: {news_channel_id}")
    
    print("\n=== CHANNEL USAGE ===")
    print(f"Regular messages -> {chat_id}")
    print(f"News with photos -> {news_channel_id}")

if __name__ == "__main__":
    print_all_channels()