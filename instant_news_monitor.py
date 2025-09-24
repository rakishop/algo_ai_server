import time
import json
import os
from datetime import datetime
from real_news_fetcher import RealNewsFetcher
from news_telegram_alert import send_news_to_telegram
import requests
from config import settings

class InstantNewsMonitor:
    def __init__(self):
        self.fetcher = RealNewsFetcher()
        self.seen_news_file = "seen_news.json"
        self.check_interval = 30  # Check every 30 seconds for faster alerts
        
    def load_seen_news(self):
        """Load previously seen news"""
        try:
            if os.path.exists(self.seen_news_file):
                with open(self.seen_news_file, 'r') as f:
                    return set(json.load(f))
        except:
            pass
        return set()
    
    def save_seen_news(self, seen_news):
        """Save seen news titles"""
        try:
            with open(self.seen_news_file, 'w') as f:
                json.dump(list(seen_news), f)
        except Exception as e:
            print(f"Error saving seen news: {e}")
    
    def send_instant_alert(self, new_news):
        """Send instant alert for new news"""
        try:
            message = f"BREAKING NEWS - {datetime.now().strftime('%H:%M')}\n\n"
            
            for i, news in enumerate(new_news[:3], 1):  # Top 3 new items
                title = news['title'][:80] + "..." if len(news['title']) > 80 else news['title']
                summary = news.get('summary', '')[:150] + "..." if len(news.get('summary', '')) > 150 else news.get('summary', '')
                source = news['source'].replace('_', ' ').title()
                
                title = news['title']
                content = news.get('summary', '') or news.get('description', '')
                
                message += f"{i}. {title}\n"
                if content and content != title:
                    message += f"   {content}\n"
                message += f"   Source: {source}\n\n"
            
            message += "Breaking news alert"
            
            # Send to News Channel
            news_channel = settings.telegram_news_channel_id or "@MyAlgoFaxNews"
            url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
            data = {"chat_id": news_channel, "text": message}
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200 and response.json().get('ok'):
                print(f"Instant news alert sent at {datetime.now().strftime('%H:%M:%S')}")
                return True
            else:
                print(f"Failed to send instant alert: {response.text}")
                return False
                
        except Exception as e:
            print(f"Error sending instant alert: {e}")
            return False
    
    def monitor_news(self):
        """Monitor for new news and send instant alerts"""
        print("Starting instant news monitoring...")
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
                        
                        print(f"Found {len(new_news)} new news items")
                        
                        # Check if market hours or after hours
                        now = datetime.now()
                        if 9 <= now.hour <= 15:  # Market hours - instant alerts
                            self.send_instant_alert(new_news)
                        else:  # After hours - send summary
                            from news_summarizer import create_and_send_summary
                            create_and_send_summary(new_news)
                        
                        # Update seen news
                        seen_news.update(new_titles)
                        self.save_seen_news(seen_news)
                    else:
                        # Only print during market hours to reduce spam
                        now = datetime.now()
                        if 9 <= now.hour <= 15:  # Market hours
                            print(f"No new news at {datetime.now().strftime('%H:%M:%S')}")
                
                # Different intervals based on market hours
                now = datetime.now()
                if 9 <= now.hour <= 15:  # Market hours - check every 30 seconds
                    time.sleep(self.check_interval)
                else:  # After hours - check every 1 hour
                    time.sleep(3600)
                
            except KeyboardInterrupt:
                print("\nStopping news monitoring...")
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