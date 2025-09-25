import requests
import time
import json

def test_ai_endpoints():
    base_url = "http://localhost:8000"
    
    endpoints = [
        "/api/v1/ai/smart-picks",
        "/api/v1/ai/anomaly-detection", 
        "/api/v1/ai/similar-stocks/RELIANCE"
    ]
    
    for endpoint in endpoints:
        start_time = time.time()
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=30)
            end_time = time.time()
            
            print(f"\n{endpoint}:")
            print(f"Status: {response.status_code}")
            print(f"Response time: {end_time - start_time:.2f}s")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Response size: {len(json.dumps(data))} chars")
                if 'recommendations' in data:
                    print(f"Recommendations: {len(data['recommendations'])}")
                elif 'anomalies' in data:
                    print(f"Anomalies found: {len(data['anomalies'])}")
                elif 'similar_stocks' in data:
                    print(f"Similar stocks: {len(data['similar_stocks'])}")
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"\n{endpoint}: ERROR - {e}")

if __name__ == "__main__":
    test_ai_endpoints()