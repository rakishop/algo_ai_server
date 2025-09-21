import json
import os
import glob
from ml_analyzer import MLStockAnalyzer
from data_processor import DataProcessor
import pandas as pd

def train_from_json_files():
    """Train ML models from JSON response files"""
    print("🤖 Starting ML model training from JSON files...")
    
    # Initialize components
    ml_analyzer = MLStockAnalyzer()
    processor = DataProcessor()
    
    # Find all JSON files
    json_files = glob.glob("response_*.json") + glob.glob("*.json")
    json_files = [f for f in json_files if f.startswith(('response_', 'live_', 'processed_'))]
    
    if not json_files:
        print("❌ No JSON files found for training")
        return False
    
    print(f"📁 Found {len(json_files)} JSON files")
    
    # Load and process all JSON data
    all_responses = {}
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                endpoint_name = json_file.replace('.json', '').replace('response_', '').replace('live_', '')
                all_responses[endpoint_name] = data
                print(f"✅ Loaded: {json_file}")
        except Exception as e:
            print(f"❌ Error loading {json_file}: {e}")
    
    if not all_responses:
        print("❌ No valid JSON data loaded")
        return False
    
    # Prepare dataset
    print("📊 Preparing dataset...")
    df = ml_analyzer.prepare_dataset(all_responses)
    
    if df.empty:
        print("❌ No stock data extracted from JSON files")
        return False
    
    print(f"📈 Extracted {len(df)} stock records")
    
    # Train models
    print("🎯 Training ML models...")
    ml_analyzer.train_models(df)
    
    print(f"✅ ML models trained successfully from {len(json_files)} JSON files!")
    return True

def setup_training_data():
    """Setup training data from live API responses"""
    print("🔄 Setting up training data from live responses...")
    
    from nse_client import NSEClient
    nse_client = NSEClient()
    
    # Collect fresh data
    endpoints = {
        'active_securities': nse_client.get_most_active_securities,
        'gainers': nse_client.get_gainers_data,
        'losers': nse_client.get_losers_data,
        'volume_gainers': nse_client.get_volume_gainers,
        'derivatives': nse_client.get_derivatives_snapshot
    }
    
    for name, func in endpoints.items():
        try:
            data = func()
            filename = f"live_{name}.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ Saved: {filename}")
        except Exception as e:
            print(f"❌ Error fetching {name}: {e}")
    
    return train_from_json_files()

if __name__ == "__main__":
    # Try training from existing files first
    if not train_from_json_files():
        # If no files, collect fresh data
        setup_training_data()