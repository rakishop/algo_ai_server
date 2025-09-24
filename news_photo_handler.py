import requests
import feedparser
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
import os
from dotenv import load_dotenv

load_dotenv()

class NewsPhotoHandler:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_NEWS_CHANNEL_ID')
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def extract_images_from_rss(self, rss_url):
        """Extract images from RSS feed entries"""
        try:
            feed = feedparser.parse(rss_url)
            news_with_images = []
            
            for entry in feed.entries[:5]:
                images = []
                
                # Check for media content in RSS
                if hasattr(entry, 'media_content'):
                    for media in entry.media_content:
                        if media.get('type', '').startswith('image/'):
                            images.append(media['url'])
                
                # Check for enclosures
                if hasattr(entry, 'enclosures'):
                    for enc in entry.enclosures:
                        if enc.get('type', '').startswith('image/'):
                            images.append(enc.href)
                
                # Parse description/summary for images
                content = entry.get('summary', '') or entry.get('description', '')
                if content:
                    soup = BeautifulSoup(content, 'html.parser')
                    img_tags = soup.find_all('img')
                    for img in img_tags:
                        src = img.get('src')
                        if src:
                            images.append(src)
                
                # If we found images, scrape the article page for more
                if not images and hasattr(entry, 'link'):
                    images = self.scrape_article_images(entry.link)
                
                if images:
                    news_with_images.append({
                        'title': entry.title,
                        'summary': entry.get('summary', '')[:200] + '...',
                        'link': entry.get('link', ''),
                        'images': images[:3],  # Max 3 images per news
                        'source': rss_url
                    })
            
            return news_with_images
        except Exception as e:
            print(f"RSS image extraction failed: {e}")
            return []
    
    def scrape_article_images(self, article_url):
        """Scrape images from article page"""
        try:
            response = requests.get(article_url, headers=self.headers, timeout=10)
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            images = []
            
            # Look for main article images
            selectors = [
                'article img', '.article-image img', '.story-image img',
                '.featured-image img', '.main-image img', 'img[alt*="stock"]',
                'img[alt*="market"]', 'img[src*="stock"]'
            ]
            
            for selector in selectors:
                img_tags = soup.select(selector)
                for img in img_tags:
                    src = img.get('src') or img.get('data-src')
                    if src:
                        # Convert relative URLs to absolute
                        full_url = urljoin(article_url, src)
                        if self.is_valid_image_url(full_url):
                            images.append(full_url)
                
                if images:
                    break
            
            return images[:2]  # Max 2 images from article
        except Exception as e:
            print(f"Article scraping failed: {e}")
            return []
    
    def is_valid_image_url(self, url):
        """Check if URL is a valid image"""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Check file extension
            path = parsed.path.lower()
            valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
            if any(path.endswith(ext) for ext in valid_extensions):
                return True
            
            # Check if URL contains image indicators
            image_indicators = ['image', 'img', 'photo', 'pic']
            return any(indicator in url.lower() for indicator in image_indicators)
        except:
            return False
    
    def send_news_with_photo(self, news_item):
        """Send news with photo to Telegram"""
        try:
            title = news_item['title']
            summary = news_item['summary']
            link = news_item['link']
            images = news_item['images']
            
            # Prepare caption for channel posting
            caption = f"📰 {title}\n\n{summary}\n\n🔗 {link}"
            
            if images:
                # Send first image with caption 
                photo_url = images[0]
                url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
                data = {
                    "chat_id": self.chat_id,
                    "photo": photo_url,
                    "caption": caption[:1024],  # Telegram caption limit
                    "parse_mode": "HTML"
                }
                
                response = requests.post(url, data=data)
                
                # Send additional images if any
                for img_url in images[1:]:
                    img_data = {
                        "chat_id": self.chat_id,
                        "photo": img_url
                    }
                    requests.post(url, data=img_data)
                
                return response.json()
            else:
                # Send as text message if no images
                return self.send_text_message(caption)
                
        except Exception as e:
            print(f"Failed to send news with photo: {e}")
            return {"ok": False, "error": str(e)}
    
    def send_text_message(self, message):
        """Send text message to Telegram"""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        return requests.post(url, data=data).json()
    
    def send_text_message_to_chat(self, message, chat_id):
        """Send text message to specific chat"""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        return requests.post(url, data=data).json()
    
    def process_rss_feeds_with_photos(self):
        """Process multiple RSS feeds and send news with photos"""
        rss_feeds = [
            'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
            'https://www.moneycontrol.com/rss/business.xml',
            'https://www.livemint.com/rss/markets',
            'https://www.cnbctv18.com/rss/market.xml'
        ]
        
        all_results = []
        
        for feed_url in rss_feeds:
            print(f"Processing feed: {feed_url}")
            news_with_images = self.extract_images_from_rss(feed_url)
            
            for news_item in news_with_images[:2]:  # Max 2 news per feed
                result = self.send_news_with_photo(news_item)
                all_results.append(result)
                
                if result.get('ok'):
                    print(f"✅ Sent: {news_item['title'][:50]}...")
                else:
                    print(f"❌ Failed: {news_item['title'][:50]}...")
        
        return all_results

def send_rss_news_with_photos():
    """Main function to send RSS news with photos"""
    handler = NewsPhotoHandler()
    return handler.process_rss_feeds_with_photos()

if __name__ == "__main__":
    results = send_rss_news_with_photos()
    print(f"Processed {len(results)} news items")