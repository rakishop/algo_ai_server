import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_chat_id():
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    
    print("1. Go to t.me/MyAlgoFaxBot")
    print("2. Click 'START' button")
    print("3. Send any message like 'hello'")
    print("4. Press Enter after sending the message")
    input()
    
    response = requests.get(url)
    data = response.json()
    
    print(f"API Response: {data}")
    
    if data.get("ok") and data.get("result"):
        for update in data["result"]:
            if "message" in update:
                chat_id = update["message"]["chat"]["id"]
                username = update["message"]["chat"].get("username", "N/A")
                print(f"Your Chat ID: {chat_id}")
                print(f"Username: @{username}")
                return chat_id
    
    print("No messages found. Make sure you:")
    print("- Clicked START in the bot")
    print("- Sent a message after clicking START")
    return None

if __name__ == "__main__":
    get_chat_id()