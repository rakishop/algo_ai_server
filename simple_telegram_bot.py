import requests
import time
from config import settings
from telegram_handler import TelegramHandler

handler = TelegramHandler()
last_update_id = 0

print("🤖 Simple bot started. Send stock symbols...")

while True:
    try:
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/getUpdates"
        params = {"offset": last_update_id + 1}
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get("ok") and data.get("result"):
            for update in data["result"]:
                last_update_id = update["update_id"]
                
                if 'message' in update and update['message'].get('text'):
                    print(f"📨 Got: {update['message']['text']}")
                    handler.handle_message(update)
        
        time.sleep(2)
        
    except KeyboardInterrupt:
        print("\n🛑 Stopped")
        break
    except Exception as e:
        print(f"❌ Error: {e}")
        time.sleep(5)