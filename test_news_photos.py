from news_photo_handler import NewsPhotoHandler
from real_news_fetcher import RealNewsFetcher

def test_news_with_photos():
    """Test news fetching with photos"""
    
    # Test RSS news with images
    print("Testing RSS news with images...")
    fetcher = RealNewsFetcher()
    news_data = fetcher.get_all_news()
    
    news_with_images = [item for item in news_data['news'] if item.get('has_images')]
    print(f"Found {len(news_with_images)} news items with images")
    
    # Test photo handler
    print("\nTesting photo handler...")
    handler = NewsPhotoHandler()
    
    if news_with_images:
        # Send first news item with photo
        result = handler.send_news_with_photo(news_with_images[0])
        print(f"Photo send result: {result.get('ok', False)}")
    else:
        print("No news with images found, testing with sample data...")
        sample_news = {
            'title': 'Market Update: Nifty Hits New High',
            'summary': 'Indian stock markets reached new heights today...',
            'link': 'https://example.com/news',
            'images': ['https://via.placeholder.com/400x300.jpg?text=Market+News']
        }
        result = handler.send_news_with_photo(sample_news)
        print(f"Sample photo send result: {result.get('ok', False)}")

if __name__ == "__main__":
    test_news_with_photos()