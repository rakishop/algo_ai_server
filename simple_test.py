#!/usr/bin/env python3
"""
Simple test script for Telegram scheduler
"""
import os
import sys
from datetime import datetime
from config import settings

def test_config():
    """Test configuration"""
    print("Testing Telegram Configuration...")
    
    if not settings.telegram_bot_token:
        print("ERROR: TELEGRAM_BOT_TOKEN not set")
        return False
    
    if not settings.telegram_chat_id:
        print("ERROR: TELEGRAM_CHAT_ID not set")
        return False
    
    print(f"Bot Token: {settings.telegram_bot_token[:10]}...")
    print(f"Chat ID: {settings.telegram_chat_id}")
    return True

def test_telegram():
    """Test Telegram send"""
    print("\nTesting Telegram Send...")
    
    try:
        import requests
        
        test_message = f"Test message - {datetime.now().strftime('%H:%M:%S')}"
        
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
        data = {
            "chat_id": settings.telegram_chat_id,
            "text": test_message
        }
        
        response = requests.post(url, data=data, timeout=10)
        
        if response.status_code == 200 and response.json().get('ok'):
            print("SUCCESS: Test message sent!")
            return True
        else:
            print(f"FAILED: {response.text}")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_nse():
    """Test NSE data"""
    print("\nTesting NSE Data...")
    
    try:
        from nse_client import NSEClient
        nse = NSEClient()
        
        active_data = nse.get_most_active_securities()
        
        if active_data and active_data.get('data'):
            print(f"SUCCESS: {len(active_data['data'])} stocks available")
            return True
        else:
            print("FAILED: No NSE data")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_alert():
    """Test stock alert"""
    print("\nTesting Stock Alert...")
    
    try:
        from auto_stock_alerts import send_stock_alert
        result = send_stock_alert()
        
        if result:
            print("SUCCESS: Stock alert sent!")
            return True
        else:
            print("FAILED: Stock alert failed")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """Run tests"""
    print("Telegram Scheduler Test")
    print("=" * 30)
    
    tests = [
        ("Config", test_config),
        ("NSE Data", test_nse),
        ("Telegram", test_telegram),
        ("Alert", test_alert)
    ]
    
    results = []
    
    for name, func in tests:
        try:
            result = func()
            results.append((name, result))
        except Exception as e:
            print(f"CRASH in {name}: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 30)
    print("Results:")
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {name}")

if __name__ == "__main__":
    main()