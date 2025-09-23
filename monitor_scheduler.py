#!/usr/bin/env python3
"""
Monitor the Telegram scheduler status
"""
import requests
import time
from datetime import datetime, timedelta
from config import settings

def check_telegram_bot():
    """Check if Telegram bot is responsive"""
    try:
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/getMe"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200 and response.json().get('ok'):
            bot_info = response.json()['result']
            print(f"✅ Bot '{bot_info['first_name']}' is active")
            return True
        else:
            print(f"❌ Bot check failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Bot check error: {e}")
        return False

def send_test_alert():
    """Send a test alert manually"""
    try:
        from auto_stock_alerts import send_stock_alert
        print("🧪 Sending test alert...")
        result = send_stock_alert()
        return result
    except Exception as e:
        print(f"❌ Test alert error: {e}")
        return False

def check_server_status():
    """Check if the main server is running"""
    try:
        url = f"{settings.base_url}/test"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print("✅ Server is running")
            return True
        else:
            print(f"❌ Server check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server not reachable: {e}")
        return False

def monitor_loop():
    """Main monitoring loop"""
    print("🔍 Starting Telegram Scheduler Monitor")
    print("=" * 50)
    
    last_check = datetime.now()
    
    while True:
        try:
            current_time = datetime.now()
            print(f"\n⏰ Monitor check at {current_time.strftime('%H:%M:%S')}")
            
            # Check bot status
            bot_ok = check_telegram_bot()
            
            # Check server status
            server_ok = check_server_status()
            
            # Check if it's time for a scheduled alert (every 30 minutes)
            minutes_since_start = (current_time - last_check).total_seconds() / 60
            
            if minutes_since_start >= 30:
                print("⏰ 30 minutes elapsed - testing alert...")
                alert_ok = send_test_alert()
                last_check = current_time
            else:
                next_alert = last_check + timedelta(minutes=30)
                print(f"⏳ Next alert scheduled for {next_alert.strftime('%H:%M:%S')}")
            
            # Overall status
            if bot_ok and server_ok:
                print("✅ All systems operational")
            else:
                print("⚠️ Some issues detected")
            
            # Wait 5 minutes before next check
            print("💤 Sleeping for 5 minutes...")
            time.sleep(300)
            
        except KeyboardInterrupt:
            print("\n🛑 Monitor stopped by user")
            break
        except Exception as e:
            print(f"❌ Monitor error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    monitor_loop()