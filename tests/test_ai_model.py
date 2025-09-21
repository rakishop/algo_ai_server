import json
from ml_analyzer import MLStockAnalyzer
from data_processor import DataProcessor

# Load the processed stock data
with open("processed_stocks.json", 'r') as f:
    stocks = json.load(f)

print(f"Loaded {len(stocks)} stock records")

# Test feature extraction
processor = DataProcessor()
ml_analyzer = MLStockAnalyzer()

# Show sample stock data
print("\nSample stock data:")
for i, stock in enumerate(stocks[:3]):
    print(f"{i+1}. {stock.get('symbol', 'N/A')}: LTP={stock.get('ltp', 0)}, Change={stock.get('pChange', 0)}%")

# Test ML model training
import pandas as pd
df = pd.DataFrame(stocks)
df = df.drop_duplicates(subset=['symbol'], keep='first')

print(f"\nTraining model with {len(df)} unique stocks...")
ml_analyzer.train_models(df)

if ml_analyzer.is_trained:
    print("Model trained successfully!")
    
    # Test predictions
    test_stocks = stocks[:10]
    predictions = ml_analyzer.predict_clusters(test_stocks)
    
    print("\nPredictions for first 10 stocks:")
    for stock in predictions:
        print(f"{stock.get('symbol', 'N/A')}: Cluster {stock.get('cluster', 'N/A')}, Anomaly: {stock.get('is_anomaly', False)}")
    
    # Test filtering by risk
    medium_risk_stocks = []
    for stock in stocks:
        if stock.get('ltp', 0) > 0 and abs(stock.get('pChange', 0)) <= 10:
            medium_risk_stocks.append(stock)
    
    print(f"\nMedium risk stocks found: {len(medium_risk_stocks)}")
    
    # Show top 5 medium risk stocks
    print("Top 5 medium risk stocks:")
    for i, stock in enumerate(medium_risk_stocks[:5]):
        print(f"{i+1}. {stock.get('symbol', 'N/A')}: Price={stock.get('ltp', 0)}, Change={stock.get('pChange', 0)}%")

else:
    print("Model training failed!")