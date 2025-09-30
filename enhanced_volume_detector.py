import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import json
import os

class EnhancedVolumeDetector:
    def __init__(self):
        self.history_file = "volume_ml_history.json"
        self.scaler = StandardScaler()
        
    def load_historical_data(self):
        """Load historical volume data for ML analysis"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                return pd.DataFrame(data)
        except:
            pass
        return pd.DataFrame()
    
    def save_data_point(self, symbol, volume, price, timestamp):
        """Save single data point"""
        try:
            df = self.load_historical_data()
            new_row = {
                'symbol': symbol,
                'volume': volume,
                'price': price,
                'timestamp': timestamp,
                'hour': datetime.fromisoformat(timestamp).hour,
                'minute': datetime.fromisoformat(timestamp).minute
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Keep only last 1000 records per symbol
            df = df.groupby('symbol').tail(1000).reset_index(drop=True)
            
            with open(self.history_file, 'w') as f:
                json.dump(df.to_dict('records'), f)
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def detect_anomalies(self, current_volumes):
        """Use pandas/numpy for anomaly detection"""
        anomalies = []
        df = self.load_historical_data()
        
        if df.empty:
            return []
        
        for symbol, current_data in current_volumes.items():
            symbol_df = df[df['symbol'] == symbol].copy()
            
            if len(symbol_df) < 5:  # Need minimum history
                continue
            
            # Calculate rolling statistics
            symbol_df['volume_ma'] = symbol_df['volume'].rolling(window=5).mean()
            symbol_df['volume_std'] = symbol_df['volume'].rolling(window=5).std()
            
            # Get latest stats
            latest_ma = symbol_df['volume_ma'].iloc[-1]
            latest_std = symbol_df['volume_std'].iloc[-1]
            current_vol = current_data['volume']
            
            if pd.notna(latest_ma) and pd.notna(latest_std) and latest_std > 0:
                # Z-score calculation
                z_score = (current_vol - latest_ma) / latest_std
                
                # Anomaly if z-score > 2 (2 standard deviations)
                if z_score > 2:
                    # Calculate additional features
                    volume_ratio = current_vol / latest_ma if latest_ma > 0 else 1
                    
                    anomalies.append({
                        'symbol': symbol,
                        'current_volume': current_vol,
                        'avg_volume': latest_ma,
                        'z_score': z_score,
                        'volume_ratio': volume_ratio,
                        'anomaly_strength': min(z_score * 20, 100),  # Scale to 0-100
                        'price': current_data['price'],
                        'price_change': current_data['change']
                    })
        
        return sorted(anomalies, key=lambda x: x['z_score'], reverse=True)
    
    def time_based_analysis(self, current_volumes):
        """Analyze volume patterns by time of day"""
        df = self.load_historical_data()
        time_anomalies = []
        
        if df.empty:
            return []
        
        current_hour = datetime.now().hour
        
        for symbol, current_data in current_volumes.items():
            symbol_df = df[df['symbol'] == symbol].copy()
            
            if len(symbol_df) < 10:
                continue
            
            # Get historical volumes for same hour
            hour_data = symbol_df[symbol_df['hour'] == current_hour]
            
            if len(hour_data) >= 3:
                hour_avg = hour_data['volume'].mean()
                hour_std = hour_data['volume'].std()
                current_vol = current_data['volume']
                
                if hour_std > 0:
                    time_z_score = (current_vol - hour_avg) / hour_std
                    
                    if time_z_score > 1.5:  # Lower threshold for time-based
                        time_anomalies.append({
                            'symbol': symbol,
                            'current_volume': current_vol,
                            'hour_avg_volume': hour_avg,
                            'time_z_score': time_z_score,
                            'time_anomaly_score': min(time_z_score * 25, 100),
                            'price': current_data['price'],
                            'price_change': current_data['change']
                        })
        
        return sorted(time_anomalies, key=lambda x: x['time_z_score'], reverse=True)

def enhanced_volume_detection(current_volumes):
    """Enhanced detection using pandas and statistical analysis"""
    detector = EnhancedVolumeDetector()
    
    # Save current data points
    for symbol, data in current_volumes.items():
        detector.save_data_point(
            symbol, data['volume'], data['price'], data['timestamp']
        )
    
    # Detect anomalies using multiple methods
    statistical_anomalies = detector.detect_anomalies(current_volumes)
    time_anomalies = detector.time_based_analysis(current_volumes)
    
    return {
        'statistical_anomalies': statistical_anomalies[:5],
        'time_based_anomalies': time_anomalies[:5],
        'total_analyzed': len(current_volumes)
    }