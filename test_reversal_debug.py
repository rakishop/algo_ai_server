#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from market_scanner import MarketScanner
from nse_client import NSEClient

def test_reversal_scanner():
    print("Testing Reversal Scanner...")
    
    # Initialize scanner
    scanner = MarketScanner()
    
    # Test with very lenient parameters
    result = scanner.scan_reversal_candidates(oversold_threshold=-2.0, volume_spike=1.0)
    
    print(f"Result: {result}")
    
    # Also test the NSE client directly
    nse_client = NSEClient()
    
    print("\nTesting NSE Client directly...")
    losers_data = nse_client.get_losers_data()
    print(f"Losers data keys: {losers_data.keys() if isinstance(losers_data, dict) else 'Not a dict'}")
    
    if "losers" in losers_data:
        print(f"Losers structure: {list(losers_data['losers'].keys())}")
        if "NIFTY" in losers_data["losers"] and "data" in losers_data["losers"]["NIFTY"]:
            nifty_losers = losers_data["losers"]["NIFTY"]["data"]
            print(f"NIFTY losers count: {len(nifty_losers)}")
            if nifty_losers:
                print(f"First loser: {nifty_losers[0]}")
    
    volume_data = nse_client.get_volume_gainers()
    print(f"Volume gainers keys: {volume_data.keys() if isinstance(volume_data, dict) else 'Not a dict'}")

if __name__ == "__main__":
    test_reversal_scanner()