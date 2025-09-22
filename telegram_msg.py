import requests
from config import settings

def send_message(text):
    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
    data = {"chat_id": settings.telegram_chat_id, "text": text}
    return requests.post(url, data=data).json()

if __name__ == "__main__":
    msg = input("Message: ")
    result = send_message(msg)
    print("✓" if result.get('ok') else f"Error: {result}")