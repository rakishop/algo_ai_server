#!/usr/bin/env python3
"""
Manual test of the scheduler functionality
"""
import schedule
import time
import threading
from datetime import datetime
from auto_stock_alerts import send_stock_alert

def run_telegram_alerts():
    """Run telegram alerts"""
    try:
        print(f"Running scheduled alert at {datetime.now().strftime('%H:%M:%S')}")
        result = send_stock_alert()
        if result:
            print(f"Alert sent successfully at {datetime.now().strftime('%H:%M:%S')}")
        else:
            print(f"Alert failed at {datetime.now().strftime('%H:%M:%S')}")
    except Exception as e:
        print(f"Scheduler error: {e}")

def run_scheduler():
    """Run the scheduler"""
    print("Starting Telegram scheduler (every 2 minutes for testing)...")
    
    # Schedule every 2 minutes for testing (change to 30 for production)
    schedule.every(2).minutes.do(run_telegram_alerts)
    
    # Run an immediate test
    print("Running initial test alert...")
    run_telegram_alerts()
    
    # Keep the scheduler running
    while True:
        try:
            schedule.run_pending()
            time.sleep(30)  # Check every 30 seconds
        except KeyboardInterrupt:
            print("\nScheduler stopped by user")
            break
        except Exception as e:
            print(f"Scheduler loop error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    print("Manual Scheduler Test")
    print("=" * 30)
    print("This will send alerts every 2 minutes")
    print("Press Ctrl+C to stop")
    print("=" * 30)
    
    run_scheduler()