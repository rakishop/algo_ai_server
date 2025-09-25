import time
import json
import os
from datetime import datetime
import pytz
from real_news_fetcher import RealNewsFetcher
from news_telegram_alert import send_news_to_telegram
import requests
from config import settings

class InstantNewsMonitor:
    def __init__(self):
        self.fetcher = RealNewsFetcher()
        self.seen_news_file = "seen_news.json"
        self.check_interval = 15  # Check every 15 seconds for real-time alerts
        self.last_summary_time = None
        self.first_message_sent = False
        self.ist = pytz.timezone('Asia/Kolkata')
    
    def get_current_time(self):
        """Get current time in IST"""
        return datetime.now(self.ist)
        
    def load_seen_news(self):
        """Load previously seen news - clear old cache"""
        # Always start fresh - don't load old cache
        return set()
    
    def save_seen_news(self, seen_news):
        """Save seen news titles"""
        try:
            with open(self.seen_news_file, 'w') as f:
                json.dump(list(seen_news), f)
        except Exception as e:
            print(f"Error saving seen news: {e}")
    
    def send_instant_alert(self, new_news):
        """Send instant alert for new news with photos"""
        try:
            from news_photo_handler import NewsPhotoHandler
            handler = NewsPhotoHandler()
            
            for news in new_news[:3]:  # Send top 3 news items individually
                # Check if news has images
                if news.get('has_images') and news.get('images'):
                    # Send with photo
                    result = handler.send_news_with_photo(news)

                else:
                    # Send as text with full content
                    title = news['title']
                    content = news.get('summary', '') or news.get('description', '')
                    source = news['source'].replace('_', ' ').title()
                    link = news.get('link', '')
                    
                    message = f"🚨 BREAKING NEWS - {datetime.now().strftime('%H:%M')}\n\n"
                    message += f"📰 {title}\n\n"
                    if content and content != title:
                        message += f"{content}\n\n"
                    message += f"📡 Source: {source}\n"
                    if link:
                        message += f"🔗 {link}\n"
                    message += "\n⚡ Breaking news alert"
                    
                    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
                    data = {
                        "chat_id": settings.telegram_chat_id,
                        "text": message,
                        "parse_mode": "HTML"
                    }
                    
                    response = requests.post(url, data=data, timeout=10)
                    

            
            return True
                
        except Exception as e:

            return False
    
    def send_2hour_summary(self):
        """Send 2-hour news summary with full news and photos"""
        try:
            current_time = datetime.now()
            
            # Check if 2 hours passed since last summary
            if self.last_summary_time and (current_time - self.last_summary_time).total_seconds() < 7200:
                return False
            
            news_data = self.fetcher.get_all_news()
            
            if not news_data or not news_data.get('news'):
                return False
            
            # Get all news from last 2 hours
            recent_news = news_data['news'][:10]  # Top 10 news items
            
            if not recent_news:
                return False
            
            # Send header message
            header_msg = f"📰 2-HOUR NEWS SUMMARY - {current_time.strftime('%H:%M')}\n\n📊 {len(recent_news)} news items"
            
            url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
            data = {
                "chat_id": settings.telegram_chat_id,
                "text": header_msg,
                "parse_mode": "HTML"
            }
            requests.post(url, data=data, timeout=10)
            
            # Send each news item with full content and photos
            from news_photo_handler import NewsPhotoHandler
            handler = NewsPhotoHandler()
            
            for i, news in enumerate(recent_news, 1):
                # Check if news has images
                if news.get('has_images') and news.get('images'):
                    # Send with photo
                    result = handler.send_news_with_photo(news)

                else:
                    # Send full text news
                    title = news['title']
                    content = news.get('summary', '') or news.get('description', '')
                    source = news['source'].replace('_', ' ').title()
                    link = news.get('link', '')
                    
                    message = f"📰 {i}/10 - {title}\n\n"
                    if content and content != title:
                        message += f"{content}\n\n"
                    message += f"📡 Source: {source}\n"
                    if link:
                        message += f"🔗 {link}"
                    
                    data = {
                        "chat_id": settings.telegram_chat_id,
                        "text": message,
                        "parse_mode": "HTML"
                    }
                    
                    response = requests.post(url, data=data, timeout=10)
                    

            
            self.last_summary_time = current_time

            return True
                
        except Exception as e:

            return False
    
    def send_first_message(self):
        """Send first message at 9:05 AM"""
        try:
            now = self.get_current_time()
            if now.hour == 9 and now.minute >= 5 and not self.first_message_sent:
                message = f"🌅 MARKET OPEN - {now.strftime('%H:%M')}\n\n📈 Real-time news monitoring started\n⚡ Breaking news alerts active"
                
                url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
                data = {
                    "chat_id": settings.telegram_chat_id,
                    "text": message,
                    "parse_mode": "HTML"
                }
                
                response = requests.post(url, data=data, timeout=10)
                
                if response.status_code == 200 and response.json().get('ok'):
                    self.first_message_sent = True
                    return True
        except Exception as e:
            print(f"Error sending first message: {e}")
        return False
    
    def monitor_news(self):
        """Monitor for new news and send instant alerts"""

        seen_news = self.load_seen_news()
        
        while True:
            try:
                # Get current news
                news_data = self.fetcher.get_all_news()
                
                if news_data and news_data.get('news'):
                    current_titles = {news['title'] for news in news_data['news']}
                    
                    # Find new news
                    new_titles = current_titles - seen_news
                    
                    if new_titles:
                        # Get full news objects for new titles
                        new_news = [news for news in news_data['news'] if news['title'] in new_titles]
                        
                        # Filter only real-time news (last 5 minutes)
                        recent_news = []
                        current_time = self.get_current_time()
                        
                        for news in new_news:
                            # Check if news is really new (within last 5 minutes)
                            news_time = None
                            if news.get('published'):
                                try:
                                    from dateutil import parser
                                    news_time = parser.parse(news['published'])
                                except:
                                    pass
                            
                            # If no timestamp or very recent (last 5 minutes), consider it new
                            if not news_time or (current_time - news_time.replace(tzinfo=None)).total_seconds() <= 300:
                                if news['title'] not in seen_news:
                                    recent_news.append(news)
                        
                        new_news = recent_news
                        

                        
                        # Check if market hours or after hours
                        now = self.get_current_time()
                        if 9 <= now.hour <= 15:  # Market hours - real-time alerts only
                            self.send_instant_alert(new_news)
                        # Off-market hours - don't send individual news
                        
                        # Update seen news
                        seen_news.update(new_titles)
                        self.save_seen_news(seen_news)
                    else:
                        pass
                
                # Send first message at 9:05 AM
                self.send_first_message()
                
                # Different intervals based on market hours
                now = self.get_current_time()
                if 9 <= now.hour <= 15:  # Market hours - check every 15 seconds
                    time.sleep(self.check_interval)
                else:  # Off-market hours - send 2-hour summary
                    self.send_2hour_summary()
                    time.sleep(7200)  # Check every 2 hours
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in news monitoring: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

def start_instant_news_monitor():
    """Function to start monitoring"""
    monitor = InstantNewsMonitor()
    monitor.monitor_news()

if __name__ == "__main__":
    start_instant_news_monitor()