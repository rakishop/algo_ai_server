#!/usr/bin/env python3
"""
Test script to verify Telegram scheduler functionality
"""
import os
import sys
from datetime import datetime
from config import settings

def test_telegram_config():
    """Test Telegram configuration"""
    print("🔧 Testing Telegram Configuration...")
    
    if not settings.telegram_bot_token:
        print("❌ TELEGRAM_BOT_TOKEN not set in .env file")
        return False
    
    if not settings.telegram_chat_id:
        print("❌ TELEGRAM_CHAT_ID not set in .env file")
        return False
    
    print(f"✅ Bot Token: {settings.telegram_bot_token[:10]}...")
    print(f"✅ Chat ID: {settings.telegram_chat_id}")
    return True

def test_telegram_send():
    """Test sending a message to Telegram"""
    print("\n📱 Testing Telegram Message Send...")
    
    try:
        import requests
        
        test_message = f"🧪 Test message from scheduler - {datetime.now().strftime('%H:%M:%S')}"
        
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
        data = {
            "chat_id": settings.telegram_chat_id,
            "text": test_message
        }
        
        response = requests.post(url, data=data, timeout=10)
        
        if response.status_code == 200 and response.json().get('ok'):
            print("✅ Test message sent successfully!")
            return True
        else:
            print(f"❌ Failed to send test message: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error sending test message: {e}")
        return False

def test_stock_alert():
    """Test the stock alert function"""
    print("\n📊 Testing Stock Alert Function...")
    
    try:
        from auto_stock_alerts import send_stock_alert
        result = send_stock_alert()
        
        if result:
            print("✅ Stock alert function working!")
            return True
        else:
            print("❌ Stock alert function failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing stock alert: {e}")
        return False

def test_nse_data():
    """Test NSE data availability"""
    print("\n📈 Testing NSE Data Availability...")
    
    try:
        from nse_client import NSEClient
        nse = NSEClient()
        
        active_data = nse.get_most_active_securities()
        
        if active_data and active_data.get('data'):
            print(f"✅ NSE data available: {len(active_data['data'])} stocks")
            return True
        else:
            print("❌ No NSE data available")
            return False
            
    except Exception as e:
        print(f"❌ Error getting NSE data: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Telegram Scheduler Test Suite")
    print("=" * 40)
    
    tests = [
        ("Configuration", test_telegram_config),
        ("NSE Data", test_nse_data),
        ("Telegram Send", test_telegram_send),
        ("Stock Alert", test_stock_alert)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 40)
    print("📋 Test Results Summary:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Scheduler should work correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the issues above.")
    
    return all_passed

if __name__ == "__main__":
    main()