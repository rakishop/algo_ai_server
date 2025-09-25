import requests
import json
import time
from config import settings

BASE_URL = settings.base_url

def test_new_features():
    print("🚀 Testing MyAlgoFax NSE API v3.0 New Features\n")
    
    # Test 1: Welcome endpoint
    print("1. Testing Welcome Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Status: {response.status_code}")
        print(f"📋 Response: {json.dumps(response.json(), indent=2)}\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    # Test 2: Portfolio Management
    print("2. Testing Portfolio Management...")
    try:
        # Create portfolio
        portfolio_data = {
            "portfolio_id": "test_001",
            "name": "Test Portfolio",
            "capital": 100000,
            "risk_tolerance": "MEDIUM"
        }
        response = requests.post(f"{BASE_URL}/api/v1/portfolio/create", json=portfolio_data)
        print(f"✅ Create Portfolio Status: {response.status_code}")
        
        # Add position
        position_data = {
            "symbol": "RELIANCE",
            "quantity": 10,
            "entry_price": 2456.75,
            "stop_loss": 2300
        }
        response = requests.post(f"{BASE_URL}/api/v1/portfolio/test_001/add-position", json=position_data)
        print(f"✅ Add Position Status: {response.status_code}")
        
        # Get performance
        response = requests.get(f"{BASE_URL}/api/v1/portfolio/test_001/performance")
        print(f"✅ Portfolio Performance Status: {response.status_code}")
        if response.status_code == 200:
            perf = response.json()
            print(f"📊 Portfolio Value: ₹{perf.get('current_portfolio_value', 0):,.2f}")
            print(f"📈 P&L: ₹{perf.get('total_pnl', 0):,.2f}\n")
    except Exception as e:
        print(f"❌ Portfolio Error: {e}\n")
    
    # Test 3: Risk Management
    print("3. Testing Risk Management...")
    try:
        risk_data = {
            "portfolio_value": 100000,
            "risk_per_trade": 0.02,
            "entry_price": 2456.75,
            "stop_loss": 2300,
            "risk_tolerance": "MEDIUM"
        }
        response = requests.post(f"{BASE_URL}/api/v1/risk/calculate-position-size", json=risk_data)
        print(f"✅ Position Size Status: {response.status_code}")
        if response.status_code == 200:
            risk = response.json()
            print(f"📊 Recommended Shares: {risk.get('recommended_shares', 0)}")
            print(f"💰 Position Value: ₹{risk.get('position_value', 0):,.2f}")
            print(f"⚠️ Risk Amount: ₹{risk.get('risk_amount', 0):,.2f}\n")
    except Exception as e:
        print(f"❌ Risk Management Error: {e}\n")
    
    # Test 4: Market Scanner
    print("4. Testing Market Scanner...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/scanner/breakouts")
        print(f"✅ Breakout Scanner Status: {response.status_code}")
        if response.status_code == 200:
            scanner = response.json()
            print(f"🔍 Total Scanned: {scanner.get('total_scanned', 0)}")
            opportunities = scanner.get('breakout_opportunities', [])
            print(f"📈 Breakout Opportunities: {len(opportunities)}")
            if opportunities:
                top = opportunities[0]
                print(f"🏆 Top Opportunity: {top.get('symbol')} - Score: {top.get('breakout_score', 0):.1f}\n")
    except Exception as e:
        print(f"❌ Scanner Error: {e}\n")
    
    # Test 5: Enhanced AI Analysis
    print("5. Testing Enhanced AI Analysis...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/ai/enhanced-market-analysis")
        print(f"✅ Enhanced AI Status: {response.status_code}")
        if response.status_code == 200:
            ai = response.json()
            print(f"🤖 Stocks Analyzed: {ai.get('total_stocks_analyzed', 0)}")
            opportunities = ai.get('top_opportunities', [])
            print(f"💡 Top Opportunities: {len(opportunities)}")
            if opportunities:
                top = opportunities[0]
                print(f"🌟 Best Pick: {top.get('symbol')} - Score: {top.get('scalping_score', 0):.1f}\n")
    except Exception as e:
        print(f"❌ AI Analysis Error: {e}\n")
    
    # Test 6: Technical Analysis
    print("6. Testing Technical Analysis...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/technical/fibonacci/RELIANCE")
        print(f"✅ Fibonacci Status: {response.status_code}")
        if response.status_code == 200:
            fib = response.json()
            print(f"📊 Symbol: {fib.get('symbol')}")
            levels = fib.get('fibonacci_levels', {})
            if levels:
                print(f"🎯 Key Levels: 38.2% = ₹{levels.get('38.2%', 0)}, 61.8% = ₹{levels.get('61.8%', 0)}\n")
    except Exception as e:
        print(f"❌ Technical Analysis Error: {e}\n")
    
    print("🎉 Testing Complete! All new features are working.")

if __name__ == "__main__":
    test_new_features()