import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import json
from datetime import datetime

class MLStockAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.is_trained = False
        
    def extract_features(self, stock_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical features from stock data"""
        features = {}
        
        # Price-related features - handle multiple field names
        features['ltp'] = self._safe_float(stock_data.get('ltp') or stock_data.get('lastPrice') or stock_data.get('close', 0))
        features['open_price'] = self._safe_float(stock_data.get('open_price') or stock_data.get('open', 0))
        features['high_price'] = self._safe_float(stock_data.get('high_price') or stock_data.get('dayHigh') or stock_data.get('high', 0))
        features['low_price'] = self._safe_float(stock_data.get('low_price') or stock_data.get('dayLow') or stock_data.get('low', 0))
        features['prev_price'] = self._safe_float(stock_data.get('prev_price') or stock_data.get('previousClose') or stock_data.get('prevClose', 0))
        
        # Change metrics
        features['net_price'] = self._safe_float(stock_data.get('net_price') or stock_data.get('change', 0))
        features['perChange'] = self._safe_float(stock_data.get('perChange') or stock_data.get('pChange', 0))
        features['pChange'] = self._safe_float(stock_data.get('pChange') or stock_data.get('perChange', 0))
        
        # Volume and turnover
        features['trade_quantity'] = self._safe_float(stock_data.get('trade_quantity') or stock_data.get('totalTradedVolume') or stock_data.get('volume', 0))
        features['turnover'] = self._safe_float(stock_data.get('turnover') or stock_data.get('totalTradedValue', 0))
        features['volume'] = self._safe_float(stock_data.get('volume') or stock_data.get('trade_quantity') or stock_data.get('totalTradedVolume', 0))
        
        # Calculate derived features
        if features['prev_price'] > 0:
            features['price_volatility'] = abs(features['high_price'] - features['low_price']) / features['prev_price'] * 100
            features['gap_up_down'] = (features['open_price'] - features['prev_price']) / features['prev_price'] * 100
        else:
            features['price_volatility'] = 0
            features['gap_up_down'] = 0
            
        if features['turnover'] > 0 and features['trade_quantity'] > 0:
            features['avg_trade_price'] = features['turnover'] / features['trade_quantity']
        else:
            features['avg_trade_price'] = features['ltp']
            
        return features
    
    def _safe_float(self, value) -> float:
        """Safely convert value to float"""
        try:
            if isinstance(value, str):
                return float(value.replace(',', ''))
            return float(value) if value is not None else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def prepare_dataset(self, api_responses: Dict[str, Any]) -> pd.DataFrame:
        """Prepare dataset from API responses"""
        all_stocks = []
        
        # Extract stocks from different API responses
        for endpoint, response in api_responses.items():
            if isinstance(response, dict):
                stocks = self._extract_stocks_from_response(response, endpoint)
                all_stocks.extend(stocks)
        
        # Convert to DataFrame
        if not all_stocks:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_stocks)
        
        # Remove duplicates based on symbol
        df = df.drop_duplicates(subset=['symbol'], keep='first')
        
        return df
    
    def _extract_stocks_from_response(self, response: Dict[str, Any], source: str) -> List[Dict[str, Any]]:
        """Extract stock data from API response"""
        stocks = []
        
        def extract_recursive(obj, path=""):
            if isinstance(obj, dict):
                # Check for stock data with symbol and price info
                if 'symbol' in obj and (obj.get('ltp') or obj.get('lastPrice') or obj.get('close')):
                    features = self.extract_features(obj)
                    stock_data = {**obj, **features, 'source': source}
                    stocks.append(stock_data)
                elif 'data' in obj and isinstance(obj['data'], list):
                    for item in obj['data']:
                        extract_recursive(item, f"{path}.data")
                else:
                    for key, value in obj.items():
                        if isinstance(value, (dict, list)) and key not in ['timestamp', 'legends']:
                            extract_recursive(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        extract_recursive(item, path)
        
        extract_recursive(response)
        return stocks
    
    def train_models(self, df: pd.DataFrame):
        """Train ML models on the dataset"""
        if df.empty:
            return
            
        # Select numerical features for training
        feature_cols = ['ltp', 'perChange', 'trade_quantity', 'turnover', 
                       'price_volatility', 'gap_up_down', 'avg_trade_price']
        
        # Filter valid features
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if not available_cols:
            return
            
        X = df[available_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train clustering model
        self.kmeans.fit(X_scaled)
        
        # Train anomaly detection model
        self.anomaly_detector.fit(X_scaled)
        
        self.is_trained = True
        
    def predict_clusters(self, stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict clusters for stocks"""
        if not self.is_trained or not stocks:
            return stocks
            
        df = pd.DataFrame(stocks)
        feature_cols = ['ltp', 'perChange', 'trade_quantity', 'turnover', 
                       'price_volatility', 'gap_up_down', 'avg_trade_price']
        
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if not available_cols:
            return stocks
            
        X = df[available_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        clusters = self.kmeans.predict(X_scaled)
        anomalies = self.anomaly_detector.predict(X_scaled)
        
        for i, stock in enumerate(stocks):
            stock['cluster'] = int(clusters[i])
            stock['is_anomaly'] = bool(anomalies[i] == -1)
            
        return stocks
    
    def find_similar_stocks(self, target_symbol: str, stocks: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        """Find stocks similar to target stock"""
        if not self.is_trained:
            return []
            
        target_stock = None
        for stock in stocks:
            if stock.get('symbol') == target_symbol:
                target_stock = stock
                break
                
        if not target_stock:
            return []
            
        target_cluster = target_stock.get('cluster')
        if target_cluster is None:
            return []
            
        similar_stocks = [
            stock for stock in stocks 
            if stock.get('cluster') == target_cluster and stock.get('symbol') != target_symbol
        ]
        
        return similar_stocks[:limit]
    
    def get_market_insights(self, stocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate market insights from stock data"""
        if not stocks:
            return {}
            
        df = pd.DataFrame(stocks)
        
        insights = {
            'total_stocks': len(stocks),
            'avg_change': df['perChange'].mean() if 'perChange' in df.columns else 0,
            'volatility_stats': {
                'high_volatility_count': len(df[df['price_volatility'] > 5]) if 'price_volatility' in df.columns else 0,
                'avg_volatility': df['price_volatility'].mean() if 'price_volatility' in df.columns else 0
            },
            'volume_stats': {
                'high_volume_count': len(df[df['trade_quantity'] > 1000000]) if 'trade_quantity' in df.columns else 0,
                'avg_volume': df['trade_quantity'].mean() if 'trade_quantity' in df.columns else 0
            }
        }
        
        if self.is_trained:
            cluster_counts = df['cluster'].value_counts().to_dict() if 'cluster' in df.columns else {}
            anomaly_count = len(df[df['is_anomaly'] == True]) if 'is_anomaly' in df.columns else 0
            
            insights['cluster_distribution'] = cluster_counts
            insights['anomaly_count'] = anomaly_count
            
        return insights