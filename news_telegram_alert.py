import requests
from datetime import datetime
from config import settings
from real_news_fetcher import RealNewsFetcher
from news_sentiment import NewsSentiment

def send_news_to_telegram():
    """Send latest news and sentiment to Telegram"""
    try:
        # Get real news
        fetcher = RealNewsFetcher()
        news_data = fetcher.get_all_news()
        
        if not news_data or not news_data.get('news'):
            raise Exception("No news data available")
        
        # Get sentiment analysis
        sentiment = NewsSentiment()
        sentiment_result = sentiment.get_market_news_sentiment()
        
        # Format message
        message = f"📰 MARKET NEWS UPDATE - {datetime.now().strftime('%H:%M')}\n\n"
        
        # Add sentiment
        sentiment_emoji = "🟢" if sentiment_result['overall_sentiment'] == 'bullish' else "🔴" if sentiment_result['overall_sentiment'] == 'bearish' else "🟡"
        message += f"{sentiment_emoji} Sentiment: {sentiment_result['overall_sentiment'].upper()}\n"
        message += f"📊 Bullish: {sentiment_result['bullish_news']} | Bearish: {sentiment_result['bearish_news']}\n\n"
        
        # Add top 5 news headlines
        message += "🔥 TOP NEWS:\n\n"
        for i, news in enumerate(news_data['news'][:5], 1):
            title = news['title'][:80] + "..." if len(news['title']) > 80 else news['title']
            summary = news.get('summary', '')[:120] + "..." if len(news.get('summary', '')) > 120 else news.get('summary', '')
            source = news['source'].replace('_', ' ').title()
            
            message += f"{i}. {title}\n"
            if summary:
                message += f"   📝 {summary}\n"
            message += f"   📡 {source}\n\n"
        
        message += f"📈 Total News: {news_data.get('total_news', 0)} sources"
        
        # Send to News Channel
        news_channel = settings.telegram_news_channel_id or "@MyAlgoFaxNews"
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
        data = {"chat_id": news_channel, "text": message}
        
        response = requests.post(url, data=data, timeout=10)
        
        if response.status_code == 200 and response.json().get('ok'):
            print(f"News alert sent at {datetime.now().strftime('%H:%M:%S')}")
            return True
        else:
            print(f"Failed to send news alert: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error sending news alert: {e}")
        return False

if __name__ == "__main__":
    send_news_to_telegram()