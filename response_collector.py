import asyncio
import json
from datetime import datetime
from nse_client import NSEClient
from data_processor import DataProcessor

class ResponseCollector:
    def __init__(self):
        self.nse_client = NSEClient()
        self.processor = DataProcessor()
        self.endpoints = {
            "gainers": self.nse_client.get_gainers_data,
            "losers": self.nse_client.get_losers_data,
            "52week_high": self.nse_client.get_52week_high_stocks_data,
            "52week_low": self.nse_client.get_52week_low_stocks_data,
            "volume_gainers": self.nse_client.get_volume_gainers,
            "most_active": self.nse_client.get_most_active_securities,
            "price_band": self.nse_client.get_price_band_hitter,
            "advance_decline": self.nse_client.get_advance_decline,
            "all_indices": self.nse_client.get_all_indices
        }
    
    def collect_all_responses(self):
        """Collect responses from all endpoints"""
        collected_data = {}
        
        for endpoint_name, endpoint_func in self.endpoints.items():
            try:
                print(f"Collecting data from {endpoint_name}...")
                response = endpoint_func()
                collected_data[endpoint_name] = response
                
                # Store in processor for analysis
                self.processor.store_api_response(endpoint_name, response)
                
                # Analyze structure
                structure = self.processor.analyze_response_structure(endpoint_name)
                print(f"Structure for {endpoint_name}: {len(structure)} fields found")
                
            except Exception as e:
                print(f"Error collecting {endpoint_name}: {str(e)}")
                collected_data[endpoint_name] = {"error": str(e)}
        
        return collected_data
    
    def save_sample_responses(self, filename: str = "sample_responses.json"):
        """Save sample responses for training"""
        data = self.collect_all_responses()
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Sample responses saved to {filename}")
        return data
    
    def analyze_all_structures(self):
        """Analyze structure of all API responses"""
        analysis = {}
        
        for endpoint_name in self.endpoints.keys():
            if endpoint_name in self.processor.data_storage:
                structure = self.processor.analyze_response_structure(endpoint_name)
                analysis[endpoint_name] = structure
        
        return analysis

# Usage example
if __name__ == "__main__":
    collector = ResponseCollector()
    
    # Collect and save sample responses
    sample_data = collector.save_sample_responses()
    
    # Analyze structures
    structures = collector.analyze_all_structures()
    
    with open("api_structures.json", 'w') as f:
        json.dump(structures, f, indent=2)
    
    print("Data collection and analysis complete!")
    print("Files created:")
    print("- sample_responses.json (API responses)")
    print("- api_structures.json (Data structures)")