import requests
import json

# Test AI model endpoints
BASE_URL = "http://localhost:8000"

def test_ai_model():
    """Test basic AI model functionality"""
    try:
        # Test welcome endpoint first
        response = requests.get(f"{BASE_URL}/")
        print(f"Welcome endpoint: {response.status_code}")
        
        # Test AI endpoints if available
        ai_endpoints = [
            "/api/v1/ai/analyze",
            "/api/v1/ai/predict", 
            "/api/v1/ai/recommendations"
        ]
        
        for endpoint in ai_endpoints:
            try:
                response = requests.get(f"{BASE_URL}{endpoint}")
                print(f"{endpoint}: {response.status_code} - {response.json()}")
            except Exception as e:
                print(f"{endpoint}: Error - {str(e)}")
                
    except requests.exceptions.ConnectionError:
        print("Server not running. Start with: uvicorn main:app --reload")

if __name__ == "__main__":
    test_ai_model()