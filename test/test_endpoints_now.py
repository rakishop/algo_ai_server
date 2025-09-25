import requests
import time

def test_endpoints():
    base_url = "http://localhost:8000"
    
    # Check available routes first
    try:
        response = requests.get(f"{base_url}/openapi.json")
        if response.status_code == 200:
            openapi_data = response.json()
            paths = openapi_data.get("paths", {})
            ai_endpoints = [path for path in paths if "/ai/" in path]
            print(f"Available AI endpoints ({len(ai_endpoints)}):")
            for endpoint in sorted(ai_endpoints):
                print(f"  {endpoint}")
    except Exception as e:
        print(f"Error checking routes: {e}")
    
    print("\nTesting new AI endpoints...")
    
    endpoints = [
        "/api/v1/ai/momentum-analysis?timeframe=daily&min_volume=1000000&limit=5",
        "/api/v1/ai/scalping-analysis?volatility_threshold=2.0&volume_threshold=3000000&limit=5", 
        "/api/v1/ai/options-strategies?strategy_type=all&limit=5"
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
                    print(f"Momentum stocks: {len(data['momentum_stocks'])}")
                elif 'scalping_opportunities' in data:
                    print(f"Scalping opportunities: {len(data['scalping_opportunities'])}")
                elif 'options_opportunities' in data:
                    print(f"Options strategies: {len(data['options_opportunities'])}")
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"ERROR - {e}")

if __name__ == "__main__":
    test_endpoints()