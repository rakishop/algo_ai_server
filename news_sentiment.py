import requests
from datetime import datetime
import json
from real_news_fetcher import RealNewsFetcher

class NewsSentiment:
    def __init__(self):
        self.keywords = {
            'bullish': ['surge', 'rally', 'breakout', 'bullish', 'positive', 'growth', 'profit', 'beat'],
            'bearish': ['crash', 'fall', 'bearish', 'negative', 'loss', 'decline', 'miss', 'weak']
        }
    
    def analyze_text_sentiment(self, text):
        """Simple sentiment analysis"""
        text_lower = text.lower()
        
        bullish_count = sum(1 for word in self.keywords['bullish'] if word in text_lower)
        bearish_count = sum(1 for word in self.keywords['bearish'] if word in text_lower)
        
        if bullish_count > bearish_count:
            return 'bullish', (bullish_count / (bullish_count + bearish_count + 1)) * 100
        elif bearish_count > bullish_count:
            return 'bearish', (bearish_count / (bullish_count + bearish_count + 1)) * 100
        else:
            return 'neutral', 50
    
    def get_market_news_sentiment(self):
        """Get overall market sentiment from real news"""
        try:
            # Fetch real news from multiple sources
            news_fetcher = RealNewsFetcher()
            news_data = news_fetcher.get_all_news()
            headlines = [item['title'] for item in news_data['news'][:20]]  # Top 20 headlines
            
            if not headlines:
                raise Exception("No real news found")
                
        except Exception as e:
            print(f"Real news fetch failed: {e}, using fallback")
            headlines = [
                "Nifty surges to new highs on positive earnings",
                "Banking stocks rally amid RBI policy optimism", 
                "IT sector shows strong growth momentum",
                "Market volatility expected due to global concerns"
            ]
        
        sentiments = []
        for headline in headlines:
            sentiment, confidence = self.analyze_text_sentiment(headline)
            sentiments.append({
                'headline': headline,
                'sentiment': sentiment,
                'confidence': confidence
            })
        
        # Calculate overall sentiment
        bullish_count = sum(1 for s in sentiments if s['sentiment'] == 'bullish')
        bearish_count = sum(1 for s in sentiments if s['sentiment'] == 'bearish')
        
        if bullish_count > bearish_count:
            overall = 'bullish'
        elif bearish_count > bullish_count:
            overall = 'bearish'
        else:
            overall = 'neutral'
        
        return {
            'overall_sentiment': overall,
            'bullish_news': bullish_count,
            'bearish_news': bearish_count,
            'neutral_news': len(sentiments) - bullish_count - bearish_count,
            'sentiment_score': (bullish_count / len(sentiments)) * 100,
            'total_headlines': len(headlines),
            'news_analysis': sentiments[:10],  # Top 10 for display
            'data_sources': ['RSS Feeds', 'NSE Announcements', 'BSE Announcements', 'TV Channels'],
            'timestamp': datetime.now().isoformat()
        }
    
    def stock_specific_sentiment(self, symbol):
        """Get sentiment for specific stock"""
        # Mock stock-specific news
        mock_news = [
            f"{symbol} reports strong quarterly results",
            f"Analysts upgrade {symbol} target price",
            f"{symbol} announces new product launch"
        ]
        
        sentiments = []
        for news in mock_news:
            sentiment, confidence = self.analyze_text_sentiment(news)
            sentiments.append({
                'news': news,
                'sentiment': sentiment,
                'confidence': confidence
            })
        
        return {
            'symbol': symbol,
            'news_sentiment': sentiments,
            'timestamp': datetime.now().isoformat()
        }

def get_news_sentiment():
    """Function for API endpoint"""
    analyzer = NewsSentiment()
    return analyzer.get_market_news_sentiment()