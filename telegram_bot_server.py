import requests
import time
from config import settings
from telegram_handler import TelegramHandler

class TelegramBotServer:
    def __init__(self):
        self.handler = TelegramHandler()
        self.last_update_id = 0
        
    def get_updates(self):
        """Get new messages from Telegram"""
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/getUpdates"
        params = {"offset": self.last_update_id + 1, "timeout": 10}
        
        try:
            response = requests.get(url, params=params)
            return response.json()
        except:
            return {"ok": False}
    
    def run(self):
        """Run the bot server"""
        print("🤖 Telegram bot started. Send stock symbols for AI analysis...")
        
        while True:
            try:
                updates = self.get_updates()
                
                if updates.get("ok") and updates.get("result"):
                    for update in updates["result"]:
                        self.last_update_id = update["update_id"]
                        
                        # Handle user messages
                        if 'message' in update and update['message'].get('text'):
                            print(f"📨 Message: {update['message']['text']}")
                            self.handler.handle_message(update)
                
                time.sleep(1)
                
            except KeyboardInterrupt:
                print("\n🛑 Bot stopped")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(5)

if __name__ == "__main__":
    bot = TelegramBotServer()
    bot.run()