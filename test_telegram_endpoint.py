import requests

# Test the telegram alert endpoint manually
url = "http://localhost:8000/api/v1/ai/telegram-alert"

try:
    response = requests.get(url)
    result = response.json()
    print("Response:", result)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Status: {result.get('status')}")
        print(f"Top gainers: {result.get('top_gainers')}")
        print(f"Top losers: {result.get('top_losers')}")
        print(f"Telegram response: {result.get('telegram_response')}")
        
except Exception as e:
    print(f"Request failed: {e}")