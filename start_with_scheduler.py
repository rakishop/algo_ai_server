#!/usr/bin/env python3
"""
Start the FastAPI server with Telegram scheduler
"""
import uvicorn
import threading
import schedule
import time
from datetime import datetime
from config import settings

def run_telegram_alerts():
    """Run telegram alerts"""
    try:
        from auto_stock_alerts import send_stock_alert
        print(f"🔄 Running scheduled alert at {datetime.now().strftime('%H:%M:%S')}")
        result = send_stock_alert()
        if result:
            print(f"✅ Alert sent successfully at {datetime.now().strftime('%H:%M:%S')}")
        else:
            print(f"❌ Alert failed at {datetime.now().strftime('%H:%M:%S')}")
    except Exception as e:
        print(f"❌ Scheduler error: {e}")

def run_scheduler():
    """Run the scheduler in a separate thread"""
    print("🕐 Starting Telegram scheduler (every 30 minutes)...")
    
    # Schedule the alerts
    schedule.every(30).minutes.do(run_telegram_alerts)
    
    # Run an immediate test
    print("🧪 Running initial test alert...")
    run_telegram_alerts()
    
    # Keep the scheduler running
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except Exception as e:
            print(f"Scheduler loop error: {e}")
            time.sleep(60)

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting FastAPI server...")
    
    # Start scheduler in background thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    # Start the server
    port = int(os.environ.get("PORT", settings.port))
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=port,
        reload=False  # Disable reload to prevent scheduler conflicts
    )

if __name__ == "__main__":
    import os
    start_server()