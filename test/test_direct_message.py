import requests
from telegram_handler import TelegramHandler

# Test direct message handling
handler = TelegramHandler()

# Simulate a message from Telegram
test_update = {
    "message": {
        "chat": {"id": 6007713678},  # Your chat ID
        "text": "RELIANCE"
    }
}

print("Testing direct message handling...")
handler.handle_message(test_update)
print("Done!")