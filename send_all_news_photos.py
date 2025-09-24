from news_photo_handler import NewsPhotoHandler
from real_news_fetcher import RealNewsFetcher
import time

def send_all_news_with_photos():
    """Send all 14 news items with photos"""
    
    fetcher = RealNewsFetcher()
    handler = NewsPhotoHandler()
    
    news_data = fetcher.get_all_news()
    news_with_images = [item for item in news_data['news'] if item.get('has_images')]
    
    print(f"Sending {len(news_with_images)} news items with photos...")
    
    for i, news_item in enumerate(news_with_images, 1):
        print(f"Sending {i}/{len(news_with_images)}: {news_item['title'][:50]}...")
        
        result = handler.send_news_with_photo(news_item)
        
        if result.get('ok'):
            print(f"[OK] Sent successfully")
        else:
            print(f"[FAIL] Failed: {result.get('error', 'Unknown error')}")
        
        # Small delay to avoid rate limiting
        time.sleep(2)
    
    print(f"\nCompleted sending {len(news_with_images)} news items")

if __name__ == "__main__":
    send_all_news_with_photos()