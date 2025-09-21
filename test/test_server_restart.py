import requests
import time
import subprocess
import os

def restart_and_test():
    print("Testing new AI endpoints after restart...")
    base_url = "http://localhost:8000"
    
    # Test basic connectivity first
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"Server status: {response.status_code}")
    except:
        print("Server not responding")
        return
    
    # Check available routes
    try:
        response = requests.get(f"{base_url}/openapi.json")
        if response.status_code == 200:
            openapi_data = response.json()
            paths = openapi_data.get("paths", {})
            ai_endpoints = [path for path in paths if "/ai/" in path]
            print(f"Total AI endpoints: {len(ai_endpoints)}")
            for endpoint in ai_endpoints:
                print(f"  {endpoint}")
    except Exception as e:
        print(f"Error checking routes: {e}")

if __name__ == "__main__":
    restart_and_test()