import requests

def test_available_routes():
    base_url = "http://localhost:8000"
    
    try:
        response = requests.get(f"{base_url}/docs")
        print(f"Docs Status: {response.status_code}")
        
        response = requests.get(f"{base_url}/openapi.json")
        if response.status_code == 200:
            openapi_data = response.json()
            paths = openapi_data.get("paths", {})
            
            print("\nAvailable AI endpoints:")
            for path in paths:
                if "/ai/" in path:
                    print(f"  {path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_available_routes()