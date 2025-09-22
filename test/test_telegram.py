import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

def test_telegram_alert():
    try:
        api_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        response = requests.get(f"{api_url}/api/v1/ai/telegram-alert")
        result = response.json()
        print("Test Result:")
        print(json.dumps(result, indent=2))
        return result
    except Exception as e:
        print(f"Test failed: {e}")
        return None

if __name__ == "__main__":
    test_telegram_alert()