import requests
from datetime import datetime
from config import settings

def create_and_send_summary(new_news):
    """Create and send news summary for after-hours"""
    try:
        if not new_news:
            return False
        
        message = f"📰 NEWS SUMMARY - {datetime.now().strftime('%H:%M')}\n\n"
        
        for i, news in enumerate(new_news[:5], 1):  # Top 5 news items
            title = news['title']
            source = news['source'].replace('_', ' ').title()
            
            message += f"{i}. {title}\n"
            message += f"   Source: {source}\n\n"
        
        message += f"Total {len(new_news)} new items"
        
        # Send to telegram
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
        data = {
            "chat_id": settings.telegram_chat_id,
            "text": message
        }
        
        response = requests.post(url, data=data, timeout=10)
        
        if response.status_code == 200 and response.json().get('ok'):
            print(f"News summary sent at {datetime.now().strftime('%H:%M:%S')}")
            return True
        else:
            print(f"Failed to send summary: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error creating summary: {e}")
        return False