import requests
import json
from datetime import datetime
from ml_analyzer import MLStockAnalyzer
from data_processor import DataProcessor

class APITester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.ml_analyzer = MLStockAnalyzer()
        self.processor = DataProcessor()
        self.endpoints = [
            "/api/v1/market/52-week-extremes",
            "/api/v1/market/daily-movers", 
            "/api/v1/market/activity-summary",
            "/api/v1/market/price-band-hits",
            "/api/v1/market/volume-leaders",
            "/api/v1/market/breadth-indicators",
            "/api/v1/market/trading-statistics",
            "/api/v1/derivatives/market-snapshot",
            "/api/v1/derivatives/active-underlyings",
            "/api/v1/derivatives/open-interest-spurts",
            "/api/v1/market/block-deals",
            "/api/v1/indices/live-data"
        ]
    
    def test_all_endpoints(self):
        """Test all endpoints and collect responses"""
        results = {}
        
        for endpoint in self.endpoints:
            try:
                print(f"Testing {endpoint}...")
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    results[endpoint] = data
                    print(f"SUCCESS {endpoint}")
                else:
                    results[endpoint] = {"error": f"HTTP {response.status_code}"}
                    print(f"ERROR {endpoint} - HTTP {response.status_code}")
                    
            except Exception as e:
                results[endpoint] = {"error": str(e)}
                print(f"ERROR {endpoint} - Exception: {str(e)}")
        
        return results
    
    def save_responses(self, responses, filename="live_api_responses.json"):
        """Save API responses to separate files"""
        timestamp = datetime.now().isoformat()
        
        # Save all responses in one file
        data_to_save = {
            "timestamp": timestamp,
            "responses": responses
        }
        
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=2, default=str)
        
        # Save each endpoint response separately
        for endpoint, response in responses.items():
            endpoint_name = endpoint.replace("/api/v1/", "").replace("/", "_")
            separate_filename = f"response_{endpoint_name}.json"
            
            endpoint_data = {
                "timestamp": timestamp,
                "endpoint": endpoint,
                "response": response
            }
            
            with open(separate_filename, 'w') as f:
                json.dump(endpoint_data, f, indent=2, default=str)
        
        print(f"All responses saved to {filename}")
        print(f"Separate files created for {len(responses)} endpoints")
    
    def analyze_responses(self, responses):
        """Analyze responses and extract stock data"""
        all_stocks = []
        
        for endpoint, response in responses.items():
            if isinstance(response, dict) and "error" not in response:
                stocks = self.processor.extract_stock_data(response)
                for stock in stocks:
                    stock['source_endpoint'] = endpoint
                    features = self.ml_analyzer.extract_features(stock)
                    all_stocks.append({**stock, **features})
        
        print(f"Extracted {len(all_stocks)} stock records")
        return all_stocks
    
    def train_model_with_live_data(self, stocks):
        """Train ML model with live stock data"""
        if not stocks:
            print("No stock data available for training")
            return
        
        import pandas as pd
        df = pd.DataFrame(stocks)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['symbol'], keep='first')
        
        print(f"Training model with {len(df)} unique stocks")
        self.ml_analyzer.train_models(df)
        
        if self.ml_analyzer.is_trained:
            print("SUCCESS: ML Model trained successfully!")
            
            # Test predictions
            stocks_with_predictions = self.ml_analyzer.predict_clusters(stocks)
            
            # Show sample results
            print("\nSample predictions:")
            for i, stock in enumerate(stocks_with_predictions[:5]):
                print(f"{stock.get('symbol', 'N/A')}: Cluster {stock.get('cluster', 'N/A')}, Anomaly: {stock.get('is_anomaly', False)}")
        else:
            print("ERROR: Model training failed")
    
    def generate_insights(self, stocks):
        """Generate market insights from stock data"""
        insights = self.ml_analyzer.get_market_insights(stocks)
        
        print("\n=== MARKET INSIGHTS ===")
        print(f"Total Stocks Analyzed: {insights.get('total_stocks', 0)}")
        print(f"Average Change: {insights.get('avg_change', 0):.2f}%")
        
        vol_stats = insights.get('volatility_stats', {})
        print(f"High Volatility Stocks: {vol_stats.get('high_volatility_count', 0)}")
        print(f"Average Volatility: {vol_stats.get('avg_volatility', 0):.2f}%")
        
        if 'cluster_distribution' in insights:
            print("Cluster Distribution:", insights['cluster_distribution'])
        
        if 'anomaly_count' in insights:
            print(f"Anomalies Detected: {insights['anomaly_count']}")
    
    def run_full_test(self):
        """Run complete test and analysis"""
        print("=== NSE API TESTING & ML TRAINING ===\n")
        
        # Test all endpoints
        responses = self.test_all_endpoints()
        
        # Save responses
        self.save_responses(responses)
        
        # Analyze and extract stock data
        stocks = self.analyze_responses(responses)
        
        if stocks:
            # Train ML model
            self.train_model_with_live_data(stocks)
            
            # Generate insights
            self.generate_insights(stocks)
            
            # Save processed stock data
            processed_data = {
                "timestamp": datetime.now().isoformat(),
                "total_stocks": len(stocks),
                "model_trained": self.ml_analyzer.is_trained,
                "stocks": stocks
            }
            
            with open("processed_stocks.json", 'w') as f:
                json.dump(processed_data, f, indent=2, default=str)
            print("\nProcessed stock data saved to processed_stocks.json")
        
        print("\n=== TEST COMPLETE ===")
        return responses, stocks

if __name__ == "__main__":
    tester = APITester()
    responses, stocks = tester.run_full_test()