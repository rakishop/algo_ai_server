from real_news_fetcher import RealNewsFetcher
from news_sentiment import NewsSentiment

# Test what data we get
fetcher = RealNewsFetcher()
news_data = fetcher.get_all_news()

print("News data type:", type(news_data))
print("News data:", news_data)

if news_data:
    print("Keys:", news_data.keys() if hasattr(news_data, 'keys') else 'No keys')
    
sentiment = NewsSentiment()
result = sentiment.get_market_news_sentiment()
print("Sentiment result:", result)