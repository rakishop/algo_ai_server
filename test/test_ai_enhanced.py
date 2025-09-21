import requests
import time

def check_available_routes():
    base_url = "http://localhost:8000"
    try:
        response = requests.get(f"{base_url}/openapi.json")
        if response.status_code == 200:
            openapi_data = response.json()
            paths = openapi_data.get("paths", {})
            print("Available AI endpoints:")
            for path in paths:
                if "/ai/" in path:
                    print(f"  {path}")
        return True
    except Exception as e:
        print(f"Error checking routes: {e}")
        return False

def test_enhanced_ai_endpoints():
    base_url = "http://localhost:8000"
    
    endpoints = [
        "/api/v1/ai/momentum-stocks?timeframe=daily&min_volume=2000000&limit=10",
        "/api/v1/ai/scalping-stocks?volatility_threshold=3.0&volume_threshold=5000000&limit=8",
        "/api/v1/ai/options-analysis?strategy_type=bullish&max_days_to_expiry=15&limit=12"
    ]
    
    for endpoint in endpoints:
        start_time = time.time()
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=30)
            end_time = time.time()
            
            print(f"\n{endpoint.split('?')[0]}:")
            print(f"Status: {response.status_code}")
            print(f"Response time: {end_time - start_time:.2f}s")
            
            if response.status_code == 200:
                data = response.json()
                
                if 'momentum_stocks' in data:
                    print(f"Momentum stocks found: {len(data['momentum_stocks'])}")
                elif 'scalping_opportunities' in data:
                    print(f"Scalping opportunities: {len(data['scalping_opportunities'])}")
                elif 'options_opportunities' in data:
                    print(f"Options strategies: {len(data['options_opportunities'])}")
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"\n{endpoint}: ERROR - {e}")

if __name__ == "__main__":
    print("Checking available routes first...")
    if check_available_routes():
        print("\nTesting enhanced AI endpoints...")
        test_enhanced_ai_endpoints()
    else:
        print("Could not check available routes")