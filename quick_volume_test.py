#!/usr/bin/env python3
"""
Quick test for volume alert system
"""

from volume_alert_system import VolumeAlertSystem
from datetime import datetime

def test_volume_alert():
    print("🚨 TESTING VOLUME ALERT SYSTEM")
    print("=" * 40)
    
    alert_system = VolumeAlertSystem()
    
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Market Open: {alert_system.is_market_open()}")
    
    try:
        result = alert_system.run_volume_check()
        if result:
            print("✅ Alert sent successfully!")
        else:
            print("ℹ️ No alerts sent")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_volume_alert()