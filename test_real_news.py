#!/usr/bin/env python3
"""
Test real news fetching
"""

from real_news_fetcher import RealNewsFetcher
from news_sentiment import NewsSentiment

def test_news_sources():
    print("TESTING REAL NEWS SOURCES")
    print("=" * 50)
    
    fetcher = RealNewsFetcher()
    
    # Test RSS feeds
    print("Testing RSS Feeds...")
    rss_news = fetcher.fetch_rss_news()
    print(f"RSS News Count: {len(rss_news)}")
    if rss_news:
        print(f"Sample: {rss_news[0]['title'][:80]}...")
    
    # Test NSE announcements
    print("\nTesting NSE Announcements...")
    nse_news = fetcher.scrape_nse_announcements()
    if nse_news and len(nse_news) > 0:
        print(f"NSE News Count: {len(nse_news)}")
        print(f"Sample: {nse_news[0]['title'][:80]}")
    else:
        print("NSE News Count: 0 (API may be down)")
    
    # Test sentiment analysis
    print("\nTesting Sentiment Analysis...")
    sentiment = NewsSentiment()
    result = sentiment.get_market_news_sentiment()
    print(f"Overall Sentiment: {result.get('overall_sentiment', 'unknown')}")
    print(f"Total Headlines: {result.get('total_headlines', 0)}")
    print(f"Bullish: {result.get('bullish_news', 0)}, Bearish: {result.get('bearish_news', 0)}")
    
    print("\nTest Complete!")

if __name__ == "__main__":
    test_news_sources()