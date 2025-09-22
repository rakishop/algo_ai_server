import requests

# Test the telegram alert endpoint to send to channel
url = "http://localhost:8000/api/v1/ai/telegram-alert"

try:
    response = requests.get(url)
    result = response.json()
    
    print("Response:", result)
    
    if result.get("status") == "success":
        print("✓ AI alert sent to channel successfully!")
        print(f"Gainers: {result.get('ai_gainers')}")
        print(f"Losers: {result.get('ai_losers')}")
        print(f"Total analyzed: {result.get('total_analyzed')}")
    else:
        print(f"✗ Failed: {result.get('message', 'Unknown error')}")
        
except Exception as e:
    print(f"Request failed: {e}")
    print("Make sure your server is running with: py run.py")