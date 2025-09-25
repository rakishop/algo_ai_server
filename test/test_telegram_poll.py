import requests
import time

print("🤖 Testing Telegram polling...")
print("Send a stock symbol to your bot, then this will check for messages")

while True:
    try:
        response = requests.get("http://localhost:8000/api/v1/telegram/poll")
        result = response.json()
        
        if result.get("processed_messages", 0) > 0:
            print(f"✓ Processed {result['processed_messages']} messages")
        
        time.sleep(3)  # Check every 3 seconds
        
    except KeyboardInterrupt:
        print("\n🛑 Stopped")
        break
    except Exception as e:
        print(f"❌ Error: {e}")
        time.sleep(5)