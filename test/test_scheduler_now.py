#!/usr/bin/env python3
"""
Test scheduler functionality right now
"""
import schedule
import time
import threading
from datetime import datetime

def test_alert():
    """Test alert function that always runs"""
    try:
        from auto_stock_alerts import send_stock_alert
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Running test alert...")
        
        # Temporarily override market check for testing
        import auto_stock_alerts
        original_is_market_open = auto_stock_alerts.is_market_open
        auto_stock_alerts.is_market_open = lambda: True  # Force market open
        
        result = send_stock_alert()
        
        # Restore original function
        auto_stock_alerts.is_market_open = original_is_market_open
        
        if result:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Alert sent successfully")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ No new stocks or alert failed")
            
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {e}")

def run_test_scheduler():
    """Run scheduler every 2 minutes for testing"""
    print("Starting test scheduler (every 2 minutes)...")
    
    # Schedule every 2 minutes for testing
    schedule.every(2).minutes.do(test_alert)
    
    # Run immediate test
    test_alert()
    
    # Keep running
    while True:
        try:
            schedule.run_pending()
            time.sleep(30)
        except KeyboardInterrupt:
            print("\nStopped by user")
            break
        except Exception as e:
            print(f"Scheduler error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    print("Testing Scheduler - Every 2 minutes")
    print("Press Ctrl+C to stop")
    print("=" * 40)
    run_test_scheduler()