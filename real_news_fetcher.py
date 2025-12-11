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
        
        # Sort RSS news by timestamp (newest first)
        all_news.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return all_news
    
    def scrape_nse_announcements(self):
        """Scrape NSE announcements from RSS feeds"""
        announcements = []
        
        nse_rss_feeds = {
            'announcements': 'https://nsearchives.nseindia.com/content/RSS/Online_announcements.xml',
            'annual_reports': 'https://nsearchives.nseindia.com/content/RSS/Annual_Reports.xml',
            'board_meetings': 'https://nsearchives.nseindia.com/content/RSS/Board_Meetings.xml',
            'brsr_reports': 'https://nsearchives.nseindia.com/content/RSS/brsr.xml',
            'corporate_actions': 'https://nsearchives.nseindia.com/content/RSS/Corporate_action.xml',
            'corporate_governance': 'https://nsearchives.nseindia.com/content/RSS/Corporate_Governance.xml',
            'daily_buyback': 'https://nsearchives.nseindia.com/content/RSS/Daily_Buyback.xml',
            'financial_results': 'https://nsearchives.nseindia.com/content/RSS/Financial_Results.xml',
            'insider_trading': 'https://nsearchives.nseindia.com/content/RSS/Insider_Trading.xml',
            'investor_complaints': 'https://nsearchives.nseindia.com/content/RSS/Investor_Complaints.xml',
            'offer_documents': 'https://nsearchives.nseindia.com/content/RSS/Offer_Documents.xml',
            'related_party_transactions': 'https://nsearchives.nseindia.com/content/RSS/Related_Party_Trans.xml',
            'sast_regulation29': 'https://nsearchives.nseindia.com/content/RSS/Sast_Regulation29.xml',
            'sast_regulation31': 'https://nsearchives.nseindia.com/content/RSS/Sast_Regulation31.xml',
            'sast_encumbrance': 'https://nsearchives.nseindia.com/content/RSS/Sast_ReasonForEncumbrance.xml',
            'secretarial_compliance': 'https://nsearchives.nseindia.com/content/RSS/Secretarial_Compliance.xml',
            'share_transfers': 'https://nsearchives.nseindia.com/content/RSS/Share_Transfers.xml',
            'shareholding_pattern': 'https://nsearchives.nseindia.com/content/RSS/Shareholding_Pattern.xml',
            'statement_deviation': 'https://nsearchives.nseindia.com/content/RSS/Statement_Of_Deviation.xml',
            'unitholding_patterns': 'https://nsearchives.nseindia.com/content/RSS/Unitholding_Patterns.xml',
            'voting_results': 'https://nsearchives.nseindia.com/content/RSS/Voting_Results.xml',
            'circulars': 'https://nsearchives.nseindia.com/content/RSS/Circulars.xml'
        }
        
        for feed_type, url in nse_rss_feeds.items():
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    try:
                        feed = atoma.parse_rss_bytes(response.content)
                        for entry in feed.items:
                            title = getattr(entry, 'title', '')
                            description = getattr(entry, 'description', '')
                            pub_date = getattr(entry, 'pub_date', None)
                            
                            announcements.append({
                                'title': f"NSE: {title}",
                                'source': 'NSE_Announcements',
                                'symbol': title,
                                'subject': description,
                                'link': getattr(entry, 'link', ''),
                                'published': str(pub_date) if pub_date else '',
                                'timestamp': datetime.now().isoformat(),
                                'feed_type': feed_type
                            })
                    except Exception as parse_error:
                        print(f"NSE {feed_type} RSS parse failed: {parse_error}")
            except Exception as e:
                print(f"NSE {feed_type} fetch failed: {e}")
        
        # Sort NSE announcements by published date (newest first)
        announcements.sort(key=lambda x: x.get('published', ''), reverse=True)
        return announcements
    
    def scrape_bse_announcements(self):
        """Scrape BSE announcements from RSS feeds"""
        announcements = []
        
        bse_rss_feeds = {
            'announcements': 'https://www.bseindia.com/data/xml/announcements.xml',
            'notices': 'https://www.bseindia.com/data/xml/notices.xml',
            'corporate_actions': 'https://www.bseindia.com/data/XML/CorpActionFeed.xml',
            'voting_results': 'https://www.bseindia.com/data/XML/VotingResultFeed.xml'
        }
        
        for feed_type, url in bse_rss_feeds.items():
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    try:
                        feed = atoma.parse_rss_bytes(response.content)
                        for entry in feed.items:
                            # Extract company name and scrip code from title
                            title = getattr(entry, 'title', '')
                            company = title.split('(')[0].strip() if '(' in title else title
                            scrip_code = title.split('(')[1].split(')')[0] if '(' in title and ')' in title else ''
                            pub_date = getattr(entry, 'pub_date', None)
                            
                            announcements.append({
                                'title': f"BSE: {company}",
                                'source': 'BSE_Announcements',
                                'company': company,
                                'symbol': scrip_code,
                                'subject': getattr(entry, 'description', ''),
                                'link': getattr(entry, 'link', ''),
                                'published': str(pub_date) if pub_date else '',
                                'timestamp': datetime.now().isoformat(),
                                'feed_type': feed_type
                            })
                    except Exception as parse_error:
                        print(f"BSE {feed_type} RSS parse failed: {parse_error}")
            except Exception as e:
                print(f"BSE {feed_type} fetch failed: {e}")
        
        # Sort BSE announcements by published date (newest first)
        announcements.sort(key=lambda x: x.get('published', ''), reverse=True)
        return announcements
    
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
        
        # Sort all news by timestamp (newest first)
        all_news.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return {
            'total_news': len(all_news),
            'rss_count': len(rss_news),
            'nse_count': len(nse_news) if nse_news else 0,
            'bse_count': len(bse_news) if bse_news else 0,
            'tv_count': len(tv_news) if tv_news else 0,
            'news': all_news,
            'last_updated': datetime.now().isoformat(),
            'filter_period': 'All available data'
        }

def fetch_real_news():
    """Function for external use"""
    fetcher = RealNewsFetcher()
    return fetcher.get_all_news()