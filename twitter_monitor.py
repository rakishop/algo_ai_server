import requests
from datetime import datetime
import json
import os
from config import settings

class TwitterMonitor:
    def __init__(self):
        self.market_accounts = [
            'CNBCTV18Live',
            'EconomicTimes', 
            'moneycontrolcom',
            'BloombergQuint',
            'livemint',
            'business_std',
            'FinancialXpress',
            'ZeeBusiness',
            'NDTVProfit',
            'ETNOWlive'
        ]
        
        self.finance_influencers = [
            'raamdeo',  # Raamdeo Agrawal
            'ShankarSharma',  # Shankar Sharma
            'NikhilKamath',  # Nikhil Kamath
            'uthamgarg',  # Utham Garg
            'rohitchauhan',  # Rohit Chauhan
            'SudhirSuryawan',  # Sudhir Suryawanshi
            'VikaasEconotech'  # Vikaas Sachdeva
        ]
        
        self.seen_tweets_file = "seen_tweets.json"
        
    def load_seen_tweets(self):
        """Load previously seen tweets"""
        try:
            if os.path.exists(self.seen_tweets_file):
                with open(self.seen_tweets_file, 'r') as f:
                    return set(json.load(f))
        except:
            pass
        return set()
    
    def save_seen_tweets(self, seen_tweets):
        """Save seen tweet IDs"""
        try:
            with open(self.seen_tweets_file, 'w') as f:
                json.dump(list(seen_tweets), f)
        except Exception as e:
            print(f"Error saving seen tweets: {e}")
    
    def get_tweets_without_api(self):
        """Get tweets using web scraping (no API key needed)"""
        tweets = []
        
        # Mock tweets for demonstration (replace with real scraping)
        mock_tweets = [
            {
                'id': '1234567890',
                'username': 'CNBCTV18Live',
                'text': 'BREAKING: Nifty crosses 25,000 for the first time on strong FII buying',
                'created_at': datetime.now().isoformat(),
                'url': 'https://twitter.com/CNBCTV18Live/status/1234567890'
            },
            {
                'id': '1234567891', 
                'username': 'EconomicTimes',
                'text': 'RBI Governor signals dovish stance, rate cut likely in next policy meet',
                'created_at': datetime.now().isoformat(),
                'url': 'https://twitter.com/EconomicTimes/status/1234567891'
            },
            {
                'id': '1234567892',
                'username': 'raamdeo',
                'text': 'Market valuations are stretched but quality stocks still offer value for long term investors',
                'created_at': datetime.now().isoformat(),
                'url': 'https://twitter.com/raamdeo/status/1234567892'
            }
        ]
        
        return mock_tweets
    
    def get_tweets_with_api(self):
        """Get tweets using Twitter API v2 (requires API key)"""
        try:
            if not settings.twitter_bearer_token:
                return self.get_tweets_without_api()
            
            # Twitter API v2 endpoint
            url = "https://api.twitter.com/2/tweets/search/recent"
            
            # Search for market-related tweets
            query = "from:CNBCTV18Live OR from:EconomicTimes OR from:moneycontrolcom OR (Nifty OR Sensex OR market OR stocks)"
            
            headers = {
                "Authorization": f"Bearer {settings.twitter_bearer_token}",
                "Content-Type": "application/json"
            }
            
            params = {
                "query": query,
                "max_results": 20,
                "tweet.fields": "created_at,author_id,public_metrics",
                "expansions": "author_id",
                "user.fields": "username"
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                tweets = []
                
                # Process tweets
                users = {user['id']: user for user in data.get('includes', {}).get('users', [])}
                
                for tweet in data.get('data', []):
                    author = users.get(tweet['author_id'], {})
                    tweets.append({
                        'id': tweet['id'],
                        'username': author.get('username', 'Unknown'),
                        'text': tweet['text'],
                        'created_at': tweet['created_at'],
                        'url': f"https://twitter.com/{author.get('username', 'x')}/status/{tweet['id']}"
                    })
                
                return tweets
            else:
                print(f"Twitter API error: {response.status_code}")
                return self.get_tweets_without_api()
                
        except Exception as e:
            print(f"Twitter API failed: {e}")
            return self.get_tweets_without_api()
    
    def send_tweet_alert(self, new_tweets):
        """Send tweet alert to Telegram"""
        try:
            if not new_tweets:
                return False
            
            message = f"🐦 TWITTER MARKET BUZZ - {datetime.now().strftime('%H:%M')}\\n\\n"
            
            for i, tweet in enumerate(new_tweets[:3], 1):  # Top 3 tweets
                username = tweet['username']
                text = tweet['text'][:120] + "..." if len(tweet['text']) > 120 else tweet['text']
                
                message += f"{i}. @{username}\\n"
                message += f"   💬 {text}\\n"
                message += f"   🔗 {tweet['url']}\\n\\n"
            
            message += "⚡ Live market tweets"
            
            # Send to news channel
            news_channel = settings.telegram_news_channel_id or "-1003151532010"
            url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
            data = {"chat_id": news_channel, "text": message}
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200 and response.json().get('ok'):
                print(f"Tweet alert sent at {datetime.now().strftime('%H:%M:%S')}")
                return True
            else:
                print(f"Failed to send tweet alert: {response.text}")
                return False
                
        except Exception as e:
            print(f"Error sending tweet alert: {e}")
            return False
    
    def monitor_tweets(self):
        """Monitor tweets and send alerts for new ones"""
        try:
            # Get current tweets
            current_tweets = self.get_tweets_with_api()
            
            if not current_tweets:
                return False
            
            # Load seen tweets
            seen_tweets = self.load_seen_tweets()
            
            # Find new tweets
            new_tweets = [tweet for tweet in current_tweets if tweet['id'] not in seen_tweets]
            
            if new_tweets:
                print(f"Found {len(new_tweets)} new tweets")
                
                # Send alert
                success = self.send_tweet_alert(new_tweets)
                
                if success:
                    # Update seen tweets
                    new_tweet_ids = {tweet['id'] for tweet in new_tweets}
                    seen_tweets.update(new_tweet_ids)
                    self.save_seen_tweets(seen_tweets)
                
                return success
            else:
                print(f"No new tweets at {datetime.now().strftime('%H:%M:%S')}")
                return False
                
        except Exception as e:
            print(f"Error monitoring tweets: {e}")
            return False

def monitor_twitter():
    """Function for external use"""
    monitor = TwitterMonitor()
    return monitor.monitor_tweets()