import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

class PriceTargets:
    def __init__(self):
        self.model = LinearRegression()
    
    def calculate_support_resistance(self, price_data):
        """Calculate support and resistance levels"""
        if len(price_data) < 10:
            return None
            
        prices = np.array(price_data)
        
        # Find local maxima and minima
        resistance_levels = []
        support_levels = []
        
        for i in range(2, len(prices) - 2):
            # Resistance (local maxima)
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                if prices[i] > prices[i-2] and prices[i] > prices[i+2]:
                    resistance_levels.append(prices[i])
            
            # Support (local minima)
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                if prices[i] < prices[i-2] and prices[i] < prices[i+2]:
                    support_levels.append(prices[i])
        
        return {
            'resistance': np.mean(resistance_levels) if resistance_levels else None,
            'support': np.mean(support_levels) if support_levels else None
        }
    
    def predict_price_target(self, stock_data):
        """Predict next price target using ML"""
        try:
            current_price = stock_data.get('lastPrice', 0)
            volume = stock_data.get('quantityTraded', 0)
            change = stock_data.get('pChange', 0)
            
            # Simple feature engineering
            features = np.array([[
                current_price,
                volume / 1000000,  # Volume in millions
                change,
                abs(change),  # Volatility
                1 if change > 0 else 0  # Direction
            ]])
            
            # Mock prediction (replace with trained model)
            price_momentum = change * 0.3
            volume_factor = min(volume / 10000000, 1) * 0.1
            
            target_change = price_momentum + volume_factor
            target_price = current_price * (1 + target_change / 100)
            
            confidence = min(abs(change) * 2 + volume / 20000000, 100)
            
            return {
                'current_price': current_price,
                'target_price': round(target_price, 2),
                'target_change': round(target_change, 2),
                'confidence': min(confidence, 95),
                'timeframe': '1-3 days'
            }
            
        except Exception as e:
            return {'error': str(e)}

def analyze_stock_targets(symbol):
    """Analyze price targets for a stock"""
    from nse_client import NSEClient
    
    nse = NSEClient()
    active_data = nse.get_most_active_securities()
    
    stock_data = None
    if active_data.get('data'):
        for stock in active_data['data']:
            if stock.get('symbol') == symbol.upper():
                stock_data = stock
                break
    
    if not stock_data:
        return {'error': f'Stock {symbol} not found'}
    
    analyzer = PriceTargets()
    targets = analyzer.predict_price_target(stock_data)
    
    return {
        'symbol': symbol.upper(),
        'analysis': targets,
        'timestamp': datetime.now().isoformat()
    }