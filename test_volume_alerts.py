#!/usr/bin/env python3
"""
Test script for volume alert system
Run this to test the volume fluctuation alerts
"""

import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from volume_alert_system import VolumeAlertSystem

def test_volume_alerts():
    """Test the volume alert system"""
    print("=" * 50)
    print("TESTING VOLUME ALERT SYSTEM")
    print("=" * 50)
    
    # Initialize the alert system
    alert_system = VolumeAlertSystem()
    
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Market open check: {alert_system.is_market_open()}")
    print()
    
    # Test single volume check
    print("Running single volume check...")
    try:
        result = alert_system.run_volume_check()
        if result:
            print("✅ Volume alert sent successfully!")
        else:
            print("ℹ️ No volume spikes detected or market closed")
    except Exception as e:
        print(f"❌ Error during volume check: {e}")
    
    print()
    print("=" * 50)
    print("Test completed!")
    print("=" * 50)

def continuous_test():
    """Run continuous volume monitoring for testing"""
    print("Starting continuous volume monitoring test...")
    print("This will run every 3 minutes. Press Ctrl+C to stop.")
    
    alert_system = VolumeAlertSystem()
    alert_system.start_monitoring()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Volume Alert System")
    parser.add_argument("--continuous", "-c", action="store_true", 
                       help="Run continuous monitoring (every 3 minutes)")
    
    args = parser.parse_args()
    
    if args.continuous:
        continuous_test()
    else:
        test_volume_alerts()