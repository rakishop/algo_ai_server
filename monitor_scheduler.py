#!/usr/bin/env python3
"""
Intelligent Derivative Monitor
Monitors derivative markets and sends intelligent recommendations
"""
import requests
import time
from datetime import datetime
from config import settings
from intelligent_derivative_analyzer import IntelligentDerivativeAnalyzer

def check_telegram_bot():
    """Check if Telegram bot is responsive"""
    try:
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/getMe"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200 and response.json().get('ok'):
            bot_info = response.json()['result']
            print(f"🤖 Bot '{bot_info['first_name']}' is active")
            return True
        else:
            print(f"❌ Bot check failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Bot check error: {e}")
        return False

def run_intelligent_derivative_analysis():
    """Run intelligent derivative analysis"""
    try:
        analyzer = IntelligentDerivativeAnalyzer()
        print("🧠 Running intelligent derivative analysis...")
        result = analyzer.run_intelligent_analysis()
        if result:
            print("✅ Intelligent analysis completed and notification sent")
        else:
            print("📊 Analysis completed - no new opportunities found")
        return result
    except Exception as e:
        print(f"❌ Intelligent analysis error: {e}")
        return False

def send_test_alert():
    """Send a test alert manually"""
    try:
        from auto_stock_alerts import send_stock_alert
        print("📱 Sending test stock alert...")
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
            print("🟢 Server is running")
            return True
        else:
            print(f"🔴 Server check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"🔴 Server not reachable: {e}")
        return False

def monitor_loop():
    """Main monitoring loop with intelligent derivative analysis"""
    print("🚀 Starting Intelligent Derivative Monitor")
    print("=" * 60)
    print("📊 Derivative Analysis: Every 15 minutes (market hours only)")
    print("📱 Stock Alerts: Every 30 minutes")
    print("🔍 System Check: Every 5 minutes")
    print("=" * 60)
    
    analyzer = IntelligentDerivativeAnalyzer()
    last_derivative_check = 0
    last_stock_alert = 0
    
    while True:
        try:
            current_time = datetime.now()
            current_timestamp = time.time()
            print(f"\n⏰ Monitor check at {current_time.strftime('%H:%M:%S')}")
            
            # Check bot status
            bot_ok = check_telegram_bot()
            
            # Check server status
            server_ok = check_server_status()
            
            # Intelligent Derivative Analysis (every 15 minutes during market hours)
            if current_timestamp - last_derivative_check >= 900:  # 15 minutes
                if analyzer.is_market_open():
                    print("\n🧠 Running intelligent derivative analysis...")
                    analysis_result = run_intelligent_derivative_analysis()
                    last_derivative_check = current_timestamp
                    
                    if analysis_result:
                        print("✅ High-confidence opportunities found and sent!")
                    else:
                        print("📊 No new high-confidence opportunities")
                else:
                    print("🕐 Market closed - skipping derivative analysis")
                    last_derivative_check = current_timestamp
            else:
                next_derivative = int((900 - (current_timestamp - last_derivative_check)) / 60)
                print(f"📊 Next derivative analysis in {next_derivative} minutes")
            
            # Stock Alerts (every 30 minutes)
            if current_timestamp - last_stock_alert >= 1800:  # 30 minutes
                print("\n📱 Running stock alert check...")
                alert_ok = send_test_alert()
                last_stock_alert = current_timestamp
                
                if alert_ok:
                    print("✅ Stock alert sent successfully")
                else:
                    print("📭 No stock alerts sent")
            else:
                next_stock = int((1800 - (current_timestamp - last_stock_alert)) / 60)
                print(f"📱 Next stock alert in {next_stock} minutes")
            
            # Overall status
            if bot_ok and server_ok:
                print("🟢 All systems operational")
            else:
                print("🟡 Some issues detected")
            
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