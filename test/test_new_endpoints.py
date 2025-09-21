import requests

def test_new_endpoints():
    base_url = "http://localhost:8000"
    
    endpoints = [
        "/api/v1/ai/momentum-analysis?timeframe=daily&min_volume=2000000&limit=10",
        "/api/v1/ai/scalping-analysis?volatility_threshold=3.0&volume_threshold=5000000&limit=8", 
        "/api/v1/ai/options-strategies?strategy_type=bullish&limit=12"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=30)
            print(f"\n{endpoint.split('?')[0]}:")
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'momentum_stocks' in data:
                    print(f"Momentum stocks: {len(data['momentum_stocks'])}")
                elif 'scalping_opportunities' in data:
                    print(f"Scalping opportunities: {len(data['scalping_opportunities'])}")
                elif 'options_opportunities' in data:
                    print(f"Options strategies: {len(data['options_opportunities'])}")
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"\n{endpoint}: ERROR - {e}")

def check_routes_first():
    base_url = "http://localhost:8000"
    try:
        response = requests.get(f"{base_url}/openapi.json")
        if response.status_code == 200:
            openapi_data = response.json()
            paths = openapi_data.get("paths", {})
            ai_endpoints = [path for path in paths if "/ai/" in path]
            print(f"Available AI endpoints ({len(ai_endpoints)}):")
            for endpoint in sorted(ai_endpoints):
                print(f"  {endpoint}")
            return ai_endpoints
    except Exception as e:
        print(f"Error checking routes: {e}")
    return []

if __name__ == "__main__":
    print("Checking available routes first...")
    available_routes = check_routes_first()
    
    print("\nTesting new AI endpoints...")
    test_new_endpoints()