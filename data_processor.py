import json
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
import os

class DataProcessor:
    def __init__(self):
        self.data_storage = {}
        self.processed_data = {}
        
    def store_api_response(self, endpoint: str, response_data: Dict[str, Any]):
        """Store API response data for analysis"""
        timestamp = datetime.now().isoformat()
        
        if endpoint not in self.data_storage:
            self.data_storage[endpoint] = []
            
        self.data_storage[endpoint].append({
            "timestamp": timestamp,
            "data": response_data
        })
        
    def analyze_response_structure(self, endpoint: str) -> Dict[str, Any]:
        """Analyze the structure of API responses"""
        if endpoint not in self.data_storage or not self.data_storage[endpoint]:
            return {"error": "No data available for analysis"}
            
        latest_response = self.data_storage[endpoint][-1]["data"]
        
        def get_structure(obj, path=""):
            structure = {}
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    structure[current_path] = {
                        "type": type(value).__name__,
                        "sample_value": str(value)[:100] if not isinstance(value, (dict, list)) else None
                    }
                    if isinstance(value, (dict, list)):
                        structure.update(get_structure(value, current_path))
            elif isinstance(obj, list) and obj:
                structure[f"{path}[0]"] = {
                    "type": "list_item",
                    "sample_value": None
                }
                structure.update(get_structure(obj[0], f"{path}[0]"))
            return structure
            
        return get_structure(latest_response)
    
    def extract_stock_data(self, response_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract stock information from response"""
        stocks = []
        
        def extract_from_obj(obj, parent_key=""):
            if isinstance(obj, dict):
                if "symbol" in obj or "name" in obj:
                    stock_info = {}
                    for key, value in obj.items():
                        if key in ["symbol", "name", "price", "change", "pChange", "volume", "value", "high", "low", "open", "close"]:
                            stock_info[key] = value
                    if stock_info:
                        stocks.append(stock_info)
                
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        extract_from_obj(value, key)
            elif isinstance(obj, list):
                for item in obj:
                    extract_from_obj(item, parent_key)
        
        extract_from_obj(response_data)
        return stocks
    
    def filter_top_performers(self, stocks: List[Dict[str, Any]], metric: str = "pChange", limit: int = 10) -> List[Dict[str, Any]]:
        """Filter top performing stocks based on a metric"""
        if not stocks:
            return []
            
        valid_stocks = [stock for stock in stocks if metric in stock and stock[metric] is not None]
        
        try:
            sorted_stocks = sorted(valid_stocks, key=lambda x: float(x[metric]), reverse=True)
            return sorted_stocks[:limit]
        except (ValueError, TypeError):
            return valid_stocks[:limit]
    
    def filter_by_volume(self, stocks: List[Dict[str, Any]], min_volume: int = 1000000) -> List[Dict[str, Any]]:
        """Filter stocks by minimum volume"""
        return [stock for stock in stocks if stock.get("volume", 0) and int(stock["volume"]) >= min_volume]
    
    def filter_by_price_range(self, stocks: List[Dict[str, Any]], min_price: float = 0, max_price: float = float('inf')) -> List[Dict[str, Any]]:
        """Filter stocks by price range"""
        filtered = []
        for stock in stocks:
            price = stock.get("price") or stock.get("lastPrice") or stock.get("close")
            if price and min_price <= float(price) <= max_price:
                filtered.append(stock)
        return filtered
    
    def save_processed_data(self, filename: str = "processed_data.json"):
        """Save processed data to file"""
        with open(filename, 'w') as f:
            json.dump(self.processed_data, f, indent=2)
    
    def load_processed_data(self, filename: str = "processed_data.json"):
        """Load processed data from file"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.processed_data = json.load(f)
    
    def extract_derivatives_data(self, response_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract derivatives/options information from response"""
        derivatives = []
        
        if isinstance(response_data, dict):
            # Handle derivatives snapshot format
            if "volume" in response_data and "data" in response_data["volume"]:
                derivatives.extend(response_data["volume"]["data"])
            
            # Handle OI spurts contracts format
            elif "data" in response_data:
                data = response_data["data"]
                if isinstance(data, list):
                    for item in data:
                        if "Slide-in-OI-Slide" in item:
                            derivatives.extend(item["Slide-in-OI-Slide"])
                        else:
                            derivatives.append(item)
                elif isinstance(data, dict):
                    derivatives.append(data)
        
        return derivatives
    
    def extract_option_chain_data(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract option chain data from NSE API response"""
        if not isinstance(response_data, dict) or "records" not in response_data:
            return {"calls": [], "puts": [], "underlying_value": 0}
        
        records = response_data["records"]
        underlying_value = records.get("underlyingValue", 0)
        data = records.get("data", [])
        
        calls = []
        puts = []
        
        for item in data:
            strike = item.get("strikePrice", 0)
            
            # Extract Call data
            if "CE" in item and item["CE"]:
                ce_data = item["CE"]
                ce_data["optionType"] = "Call"
                ce_data["strikePrice"] = strike
                calls.append(ce_data)
            
            # Extract Put data
            if "PE" in item and item["PE"]:
                pe_data = item["PE"]
                pe_data["optionType"] = "Put"
                pe_data["strikePrice"] = strike
                puts.append(pe_data)
        
        return {
            "calls": calls,
            "puts": puts,
            "underlying_value": underlying_value,
            "timestamp": records.get("timestamp", ""),
            "total_strikes": len(data)
        }