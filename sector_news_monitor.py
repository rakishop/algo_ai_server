import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json
from config import settings

class SectorNewsMonitor:
    def __init__(self):
        self.sector_keywords = {
            'power': ['power', 'electricity', 'renewable', 'solar', 'wind', 'coal', 'thermal', 'NTPC', 'POWERGRID'],
            'banking': ['bank', 'banking', 'RBI', 'credit', 'loan', 'HDFC', 'ICICI', 'SBI'],
            'pharma': ['pharma', 'drug', 'medicine', 'healthcare', 'FDA', 'CIPLA', 'SUNPHARMA'],
            'auto': ['auto', 'automobile', 'car', 'vehicle', 'MARUTI', 'TATA MOTORS', 'BAJAJ'],
            'it': ['IT', 'software', 'technology', 'digital', 'TCS', 'INFOSYS', 'WIPRO'],
            'oil_gas': ['oil', 'gas', 'petroleum', 'crude', 'ONGC', 'RELIANCE', 'IOC'],
            'metals': ['steel', 'iron', 'copper', 'aluminum', 'TATA STEEL', 'JSW', 'HINDALCO']
        }
        
        self.analyst_sources = {
            'clsa': 'https://www.clsa.com/research/',
            'morgan_stanley': 'https://www.morganstanley.com/ideas/',
            'goldman_sachs': 'https://www.goldmansachs.com/insights/',
            'jp_morgan': 'https://www.jpmorgan.com/insights',
            'citi': 'https://www.citigroup.com/citi/research/'
        }
    
    def scrape_analyst_reports(self):
        """Scrape analyst reports and recommendations"""
        reports = []
        
        # Mock analyst reports (replace with real scraping)
        mock_reports = [
            {
                'analyst': 'CLSA',
                'sector': 'Power',
                'title': 'CLSA upgrades Power sector on renewable energy push',
                'summary': 'Positive outlook on renewable energy companies due to government policy support',
                'recommendation': 'BUY',
                'target_stocks': ['NTPC', 'POWERGRID', 'ADANIGREEN']
            },
            {
                'analyst': 'Morgan Stanley',
                'sector': 'Banking',
                'title': 'Morgan Stanley bullish on private banks',
                'summary': 'Credit growth recovery and NIM expansion expected',
                'recommendation': 'OVERWEIGHT',
                'target_stocks': ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK']
            },
            {
                'analyst': 'Goldman Sachs',
                'sector': 'IT',
                'title': 'Goldman Sachs sees AI boom benefiting Indian IT',
                'summary': 'Artificial Intelligence adoption to drive growth in IT services',
                'recommendation': 'BUY',
                'target_stocks': ['TCS', 'INFY', 'HCLTECH']
            }
        ]
        
        return mock_reports
    
    def categorize_news_by_sector(self, news_items):
        """Categorize news by sector based on keywords"""
        sector_news = {sector: [] for sector in self.sector_keywords.keys()}
        
        for news in news_items:
            title_lower = news['title'].lower()
            summary_lower = news.get('summary', '').lower()
            content = f"{title_lower} {summary_lower}"
            
            for sector, keywords in self.sector_keywords.items():
                if any(keyword.lower() in content for keyword in keywords):
                    sector_news[sector].append(news)
                    break
        
        return sector_news
    
    def send_sector_alert(self, sector, news_items, analyst_reports):
        """Send sector-specific alert to Telegram"""
        try:
            if not news_items and not analyst_reports:
                return False
            
            sector_name = sector.replace('_', ' ').title()
            # Get IST time
            utc_now = datetime.utcnow()
            ist_time = utc_now + timedelta(hours=5, minutes=30)
            message = f"📊 {sector_name.upper()} SECTOR UPDATE - {ist_time.strftime('%H:%M')}\n\n"
            
            # Add analyst reports
            sector_reports = [r for r in analyst_reports if r['sector'].lower() == sector_name.lower()]
            if sector_reports:
                message += "🏦 ANALYST REPORTS:\n"
                for report in sector_reports[:2]:
                    message += f"• {report['analyst']}: {report['recommendation']}\n"
                    message += f"  {report['title'][:80]}...\n"
                    if report['target_stocks']:
                        stocks = ', '.join(report['target_stocks'][:3])
                        message += f"  🎯 Stocks: {stocks}\n\n"
            
            # Add sector news
            if news_items:
                message += "📰 SECTOR NEWS:\n"
                for i, news in enumerate(news_items[:3], 1):
                    title = news['title'][:70] + "..." if len(news['title']) > 70 else news['title']
                    source = news['source'].replace('_', ' ').title()
                    message += f"{i}. {title}\n   📡 {source}\n\n"
            
            message += f"🔍 {sector_name} sector analysis"
            
            # Send to news channel
            news_channel = settings.telegram_news_channel_id or "-1003151532010"
            url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
            data = {"chat_id": news_channel, "text": message}
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200 and response.json().get('ok'):
                print(f"Sector alert sent for {sector_name} at {datetime.now().strftime('%H:%M:%S')}")
                return True
            else:
                print(f"Failed to send sector alert: {response.text}")
                return False
                
        except Exception as e:
            print(f"Error sending sector alert: {e}")
            return False
    
    def monitor_sectors(self, news_data):
        """Monitor and send sector-wise alerts"""
        try:
            # Get analyst reports
            analyst_reports = self.scrape_analyst_reports()
            
            # Categorize news by sector
            sector_news = self.categorize_news_by_sector(news_data.get('news', []))
            
            # Send alerts for sectors with significant news
            alerts_sent = 0
            for sector, news_items in sector_news.items():
                if len(news_items) >= 2:  # Send alert if 2+ news items in sector
                    success = self.send_sector_alert(sector, news_items, analyst_reports)
                    if success:
                        alerts_sent += 1
            
            return alerts_sent
            
        except Exception as e:
            print(f"Error in sector monitoring: {e}")
            return 0

def monitor_sector_news():
    """Function for external use"""
    from real_news_fetcher import RealNewsFetcher
    
    fetcher = RealNewsFetcher()
    news_data = fetcher.get_all_news()
    
    monitor = SectorNewsMonitor()
    return monitor.monitor_sectors(news_data)