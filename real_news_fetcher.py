import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json

class RealNewsFetcher:
    def __init__(self):
        self.rss_feeds = {
            'economic_times': 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
            'moneycontrol': 'https://www.moneycontrol.com/rss/business.xml',
            'business_standard': 'https://www.business-standard.com/rss/markets-106.rss',
            'livemint': 'https://www.livemint.com/rss/markets',
            'cnbc_tv18': 'https://www.cnbctv18.com/rss/market.xml',
            'zee_business': 'https://zeenews.india.com/rss/business-news.xml'
        }
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def fetch_rss_news(self):
        """Fetch news from RSS feeds"""
        all_news = []
        
        for source, url in self.rss_feeds.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:5]:  # Top 5 from each source
                    all_news.append({
                        'title': entry.title,
                        'summary': entry.summary if hasattr(entry, 'summary') else entry.description if hasattr(entry, 'description') else '',
                        'source': source,
                        'link': entry.link if hasattr(entry, 'link') else '',
                        'published': entry.published if hasattr(entry, 'published') else '',
                        'timestamp': datetime.now().isoformat()
                    })
            except Exception as e:
                print(f"RSS fetch failed for {source}: {e}")
                continue
        
        return all_news
    
    def scrape_nse_announcements(self):
        """Scrape NSE corporate announcements"""
        try:
            url = "https://www.nseindia.com/api/corporates-corporateActions?index=equities"
            response = requests.get(url, headers=self.headers, timeout=10)
            
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
            'et_now': 'https://www.etnow.in/market-news',
            'zee_news': 'https://zeenews.india.com/business'
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