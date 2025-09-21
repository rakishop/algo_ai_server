import requests

def check_all_ai_routes():
    base_url = "http://localhost:8000"
    
    # Get all available routes
    try:
        response = requests.get(f"{base_url}/openapi.json")
        if response.status_code == 200:
            openapi_data = response.json()
            paths = openapi_data.get("paths", {})
            
            print("=== ALL AVAILABLE ROUTES ===")
            all_routes = list(paths.keys())
            for route in sorted(all_routes):
                print(f"  {route}")
            
            print(f"\n=== AI ROUTES ONLY ({len([r for r in all_routes if '/ai/' in r])}) ===")
            ai_routes = [r for r in all_routes if '/ai/' in r]
            for route in sorted(ai_routes):
                print(f"  {route}")
                
            print(f"\n=== TESTING NEW ROUTES ===")
            new_routes = [
                "/api/v1/ai/momentum-analysis",
                "/api/v1/ai/scalping-analysis", 
                "/api/v1/ai/options-strategies"
            ]
            
            for route in new_routes:
                if route in all_routes:
                    print(f"✓ {route} - FOUND")
                else:
                    print(f"✗ {route} - NOT FOUND")
                    
        else:
            print(f"Error getting routes: {response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_all_ai_routes()