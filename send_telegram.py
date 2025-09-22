import requests
import os
from dotenv import load_dotenv

load_dotenv()

def send_telegram_message(message):
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    
    response = requests.post(url, data=data)
    return response.json()

if __name__ == "__main__":
    message = input("Enter message: ")
    result = send_telegram_message(message)
    print("Sent!" if result.get('ok') else f"Error: {result}")