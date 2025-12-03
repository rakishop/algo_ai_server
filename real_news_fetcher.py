import atoma
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json
from urllib.parse import urljoin

class RealNewsFetcher:
    def __init__(self):
        self.rss_feeds = {
            'economic_times': 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
            'moneycontrol': 'https://www.moneycontrol.com/rss/business.xml',
            'business_standard': 'https://www.business-standard.com/rss/markets-106.rss',
            'livemint': 'https://www.livemint.com/rss/markets',
            'cnbc_tv18': 'https://www.cnbctv18.com/rss/market.xml',
            'zee_business': 'https://zeenews.india.com/rss/business-news.xml',
            'cnbc_awaaz': 'https://www.cnbctv18.com/rss/cnbc-awaaz.xml',

            'hindu_business': 'https://www.thehindu.com/business/markets/feeder/default.rss',
            'reuters_news': 'https://ir.thomsonreuters.com/rss/news-releases.xml?items=15',
            'reuters_events': 'https://ir.thomsonreuters.com/rss/events.xml?items=15',
            'bloomberg_india': 'https://feeds.bloomberg.com/markets/news.rss',
            'ndtv_business': 'https://feeds.feedburner.com/ndtvprofit-latest'
        }
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def extract_images_from_entry(self, entry):
        """Extract images from RSS entry"""
        images = []
        
        # Parse content for images only
        content = getattr(entry, 'description', '') or ''
        if content:
            try:
                soup = BeautifulSoup(content, 'html.parser')
                img_tags = soup.find_all('img')
                for img in img_tags:
                    src = img.get('src')
                    if src:
                        images.append(src)
            except:
                pass
        
        return images[:2]  # Max 2 images
    
    def fetch_rss_news(self):
        """Fetch news from RSS feeds with images"""
        all_news = []
        
        for source, url in self.rss_feeds.items():
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    try:
                        feed = atoma.parse_rss_bytes(response.content)
                        for entry in feed.items:  # All items from each source
                            images = self.extract_images_from_entry(entry)
                            
                            all_news.append({
                                'title': getattr(entry, 'title', ''),
                                'summary': getattr(entry, 'description', ''),
                                'source': source,
                                'link': getattr(entry, 'link', ''),
                                'published': str(getattr(entry, 'pub_date', '')),
                                'images': images,
                                'has_images': len(images) > 0,
                                'timestamp': datetime.now().isoformat()
                            })
                    except Exception as parse_error:
                        # Try with atom parser if RSS fails
                        try:
                            feed = atoma.parse_atom_bytes(response.content)
                            for entry in feed.entries:
                                all_news.append({
                                    'title': getattr(entry, 'title', {}).get('value', '') if hasattr(getattr(entry, 'title', {}), 'get') else str(getattr(entry, 'title', '')),
                                    'summary': getattr(entry, 'summary', {}).get('value', '') if hasattr(getattr(entry, 'summary', {}), 'get') else str(getattr(entry, 'summary', '')),
                                    'source': source,
                                    'link': getattr(entry, 'id', ''),
                                    'published': str(getattr(entry, 'published', '')),
                                    'images': [],
                                    'has_images': False,
                                    'timestamp': datetime.now().isoformat()
                                })
                        except Exception as atom_error:
                            print(f"RSS/Atom parse failed for {source}: {parse_error}")
            except Exception as e:
                print(f"RSS fetch failed for {source}: {e}")
                continue
        
        return all_news
    
    def scrape_nse_announcements(self):
        """Scrape NSE corporate announcements with proper session"""
        try:
            # Create session and get cookies first
            session = requests.Session()
            session.headers.update(self.headers)
            
            # First visit main page to get cookies
            session.get("https://www.nseindia.com/companies-listing/corporate-filings-announcements", timeout=10)
            
            # Now make API call with cookies
            url = "https://www.nseindia.com/api/corporates-corporateActions?index=equities"
            response = session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                announcements = []
                
                for item in data[:10]:  # Top 10 announcements
                    announcements.append({
                        'title': f"{item.get('symbol', '')} - {item.get('subject', '')}",
                        'source': 'NSE_Announcements',
                        'symbol': item.get('symbol', ''),
                        'subject': item.get('subject', ''),
                        'timestamp': datetime.now().isoformat()
                    })
                
                session.close()
                return announcements
        except Exception as e:
            print(f"NSE scraping failed: {e}")
            return []
    
    def scrape_bse_announcements(self):
        """Scrape BSE announcements"""
        try:
            url = "https://www.bseindia.com/corporates/ann.html"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                announcements = []
                
                # Find announcement table rows
                rows = soup.find_all('tr')[:10]  # Top 10
                
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        announcements.append({
                            'title': f"BSE: {cells[1].get_text(strip=True) if len(cells) > 1 else ''}",
                            'source': 'BSE_Announcements',
                            'company': cells[1].get_text(strip=True) if len(cells) > 1 else '',
                            'subject': cells[2].get_text(strip=True) if len(cells) > 2 else '',
                            'timestamp': datetime.now().isoformat()
                        })
                
                return announcements
        except Exception as e:
            print(f"BSE scraping failed: {e}")
            return []
    
    def scrape_tv_channel_news(self):
        """Scrape major TV channel websites"""
        tv_sources = {
            'cnbc_tv18': 'https://www.cnbctv18.com/market/',
            'cnbc_awaaz': 'https://www.cnbctv18.com/cnbc-awaaz/',
            'et_now': 'https://www.etnow.in/market-news',
            'zee_news': 'https://zeenews.india.com/business',
            'ndtv_profit': 'https://www.ndtv.com/business/news',
            'news18_business': 'https://www.news18.com/business/',
            'india_today_business': 'https://www.indiatoday.in/business'
        }
        
        all_tv_news = []
        
        for source, url in tv_sources.items():
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Generic headline extraction
                    headlines = soup.find_all(['h1', 'h2', 'h3'], limit=5)
                    
                    for headline in headlines:
                        title = headline.get_text(strip=True)
                        if len(title) > 20:  # Filter out short/irrelevant text
                            all_tv_news.append({
                                'title': title,
                                'source': source,
                                'timestamp': datetime.now().isoformat()
                            })
            except Exception as e:
                print(f"TV news scraping failed for {source}: {e}")
                continue
        
        return all_tv_news
    
    def get_all_news(self):
        """Get news from all sources"""
        all_news = []
        
        # RSS feeds
        rss_news = self.fetch_rss_news()
        all_news.extend(rss_news)
        
        # NSE announcements
        nse_news = self.scrape_nse_announcements()
        if nse_news:
            all_news.extend(nse_news)
        
        # BSE announcements
        bse_news = self.scrape_bse_announcements()
        if bse_news:
            all_news.extend(bse_news)
        
        # TV channel news
        tv_news = self.scrape_tv_channel_news()
        if tv_news:
            all_news.extend(tv_news)
        
        return {
            'total_news': len(all_news),
            'rss_count': len(rss_news),
            'nse_count': len(nse_news) if nse_news else 0,
            'bse_count': len(bse_news) if bse_news else 0,
            'tv_count': len(tv_news) if tv_news else 0,
            'news': all_news,
            'last_updated': datetime.now().isoformat()
        }

def fetch_real_news():
    """Function for external use"""
    fetcher = RealNewsFetcher()
    return fetcher.get_all_news()