import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
from datetime import datetime
from utils.ai_risk_calculator import AIRiskCalculator

class EquityMLModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data_dir = "equity_training_data"
        self.model_file = "equity_trained_model.pkl"
        self.last_training_time = None
        self.training_interval = 300  # Retrain every 5 minutes
        self.max_training_files = 100  # Use only latest 100 files
        self.risk_calculator = AIRiskCalculator()
        os.makedirs(self.training_data_dir, exist_ok=True)
        
        # Try to load existing model
        if os.path.exists(self.model_file):
            load_result = self.load_model(self.model_file)
            if load_result.get("status") == "loaded":
                print(f"Loaded existing equity model: {load_result}")
                self.last_training_time = datetime.now().timestamp()
        
    def prepare_features(self, data):
        """Extract features from equity data"""
        features = []
        for stock in data:
            volume = stock.get('totalTradedVolume', stock.get('trade_quantity', stock.get('volume', 0)))
            price_change = stock.get('pChange', stock.get('perChange', 0))
            last_price = stock.get('lastPrice', stock.get('ltp', 0))
            high_52w = stock.get('yearHigh', stock.get('high_price', last_price * 1.1))
            low_52w = stock.get('yearLow', stock.get('low_price', last_price * 0.9))
            total_value = stock.get('totalTradedValue', stock.get('turnover', volume * last_price))
            
            # Calculate derived features
            volume_ratio = volume / 1000000 if volume > 0 else 0
            price_momentum = abs(price_change) / 10
            year_high_ratio = last_price / high_52w if high_52w > 0 else 0
            year_low_ratio = last_price / low_52w if low_52w > 0 else 0
            value_ratio = total_value / 1000000 if total_value > 0 else 0
            volatility = (high_52w - low_52w) / low_52w if low_52w > 0 else 0
            
            features.append([
                volume_ratio,
                price_momentum,
                year_high_ratio,
                year_low_ratio,
                value_ratio,
                volatility,
                1 if price_change > 0 else 0
            ])
        
        return np.array(features)
    
    def create_labels(self, data):
        """Create training labels for equity data"""
        labels = []
        for stock in data:
            volume = stock.get('trade_quantity', stock.get('totalTradedVolume', 0))
            price_change = stock.get('perChange', stock.get('pChange', 0))
            total_value = stock.get('turnover', stock.get('totalTradedValue', 0))
            
            # Label as 1 (good trade) if meets criteria
            score = 0
            if volume > 500000: score += 1
            if abs(price_change) > 3: score += 1
            if total_value > 10000000: score += 1
            
            labels.append(1 if score >= 2 else 0)
        
        return np.array(labels)
    
    def should_retrain(self):
        """Check if model should be retrained"""
        if not self.is_trained:
            return True
        
        if self.last_training_time is None:
            return True
            
        time_since_training = datetime.now().timestamp() - self.last_training_time
        return time_since_training > self.training_interval
    
    def save_training_data(self, data):
        """Save current equity data for training"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"equity_data_{timestamp}.json"
        filepath = os.path.join(self.training_data_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        return {"status": "saved", "file": filename}
    
    def load_recent_training_data(self):
        """Load only recent training data"""
        if not os.path.exists(self.training_data_dir):
            return []
        
        files = []
        for filename in os.listdir(self.training_data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.training_data_dir, filename)
                mtime = os.path.getmtime(filepath)
                files.append((mtime, filepath))
        
        files.sort(reverse=True)
        recent_files = files[:self.max_training_files]
        
        recent_data = []
        for _, filepath in recent_files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    recent_data.append(data)
            except Exception as e:
                continue
        
        return recent_data
    
    def train(self, historical_data):
        """Train model on equity data"""
        all_features = []
        all_labels = []
        
        print(f"Processing {len(historical_data)} equity datasets for training")
        
        for i, day_data in enumerate(historical_data):
            stocks = []
            
            # Handle NSE equity data structure
            if isinstance(day_data, dict):
                # Check for nested data categories (gainers, losers, etc.)
                for key, value in day_data.items():
                    if isinstance(value, dict) and 'data' in value:
                        stocks.extend(value['data'])
                    elif key == 'data' and isinstance(value, list):
                        stocks.extend(value)
                
                # Fallback for direct data array
                if not stocks and 'data' in day_data:
                    stocks = day_data['data']
            elif isinstance(day_data, list):
                stocks = day_data
            
            if stocks:
                features = self.prepare_features(stocks)
                labels = self.create_labels(stocks)
                
                print(f"Equity Dataset {i+1}: {len(stocks)} stocks, {len(features)} features")
                
                all_features.extend(features)
                all_labels.extend(labels)
        
        print(f"Total equity features: {len(all_features)}, Total labels: {len(all_labels)}")
        
        if len(all_features) > 0:
            X = np.array(all_features)
            y = np.array(all_labels)
            
            print(f"Training equity model with X shape: {X.shape}, y shape: {y.shape}")
            
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            print(f"Equity model trained successfully with {len(X)} samples")
            return {"status": "trained", "samples": len(X)}
        
        print("Equity training failed: No features extracted")
        return {"status": "failed", "error": "No training data"}
    
    def train_with_historical_data(self, current_data=None):
        """Smart training for equity data"""
        if current_data:
            save_result = self.save_training_data(current_data)
            print(f"Saved current equity data: {save_result}")
        
        if not self.should_retrain():
            print("Using existing equity model (no retraining needed)")
            return {"status": "using_existing", "last_trained": self.last_training_time}
        
        historical_data = self.load_recent_training_data()
        if current_data:
            historical_data.append(current_data)
        
        print(f"Training equity model with {len(historical_data)} recent datasets")
        
        if len(historical_data) > 0:
            training_result = self.train(historical_data)
            self.last_training_time = datetime.now().timestamp()
            self.save_model(self.model_file)
            return training_result
        
        return {"status": "failed", "error": "No training data available"}
    
    def predict_recommendations(self, current_data):
        """AI-powered equity recommendations using pandas and ML"""
        stocks = []
        
        # Handle NSE equity data structure
        if isinstance(current_data, dict):
            # Check for nested data categories (gainers, losers, allSec, etc.)
            for key, value in current_data.items():
                if isinstance(value, dict) and 'data' in value:
                    stocks.extend(value['data'])
                elif key == 'data' and isinstance(value, list):
                    stocks.extend(value)
            
            # Fallback for direct data array
            if not stocks and 'data' in current_data:
                stocks = current_data['data']
        elif isinstance(current_data, list):
            stocks = current_data
        
        if not stocks:
            return {"error": "Invalid data format - no stock data found"}
        df = pd.DataFrame(stocks)
        
        # Normalize field names for NSE equity data
        df['volume'] = df['trade_quantity'].fillna(0) if 'trade_quantity' in df.columns else df.get('volume', 0)
        df['price_change'] = df['perChange'].fillna(0) if 'perChange' in df.columns else df.get('pChange', 0)
        df['last_price'] = df['ltp'].fillna(0) if 'ltp' in df.columns else df.get('lastPrice', 0)
        df['year_high'] = df['high_price'].fillna(df['last_price'] * 1.1) if 'high_price' in df.columns else df['last_price'] * 1.1
        df['year_low'] = df['low_price'].fillna(df['last_price'] * 0.9) if 'low_price' in df.columns else df['last_price'] * 0.9
        df['total_value'] = df['turnover'].fillna(df['volume'] * df['last_price']) if 'turnover' in df.columns else df['volume'] * df['last_price']
        
        # Pandas-based feature engineering
        df['volume_rank'] = df['volume'].rank(pct=True)
        df['price_momentum'] = abs(df['price_change'])
        df['value_rank'] = df['total_value'].rank(pct=True)
        df['year_high_proximity'] = df['last_price'] / df['year_high']
        df['year_low_distance'] = df['last_price'] / df['year_low']
        df['volatility'] = (df['year_high'] - df['year_low']) / df['year_low']
        
        # AI Scoring
        df['ai_score'] = (
            df['volume_rank'] * 30 +
            (df['price_momentum'] / df['price_momentum'].max()) * 25 +
            df['value_rank'] * 25 +
            df['year_high_proximity'] * 10 +
            (1 - df['volatility'] / df['volatility'].max()) * 10
        )
        
        # ML predictions if model is trained
        if self.is_trained:
            features = self.prepare_features(stocks)
            if len(features) > 0:
                try:
                    X_scaled = self.scaler.transform(features)
                    ml_predictions = self.model.predict_proba(X_scaled)
                    # Handle both binary and single class predictions
                    if ml_predictions.shape[1] > 1:
                        df['ml_probability'] = [prob[1] for prob in ml_predictions]
                    else:
                        df['ml_probability'] = [prob[0] for prob in ml_predictions]
                    df['ai_score'] = df['ai_score'] * 0.6 + df['ml_probability'] * 100 * 0.4
                except Exception as e:
                    print(f"ML prediction error: {e}")
                    # Continue without ML enhancement
        
        # Generate recommendations
        def get_recommendation(row):
            change = row['price_change']
            if change > 5:
                return 'STRONG BUY'
            elif change > 2:
                return 'BUY'
            elif change < -5:
                return 'STRONG SELL'
            elif change < -2:
                return 'SELL'
            else:
                return 'HOLD'
        
        def get_trend(row):
            change = row['price_change']
            if change > 2:
                return 'BULLISH'
            elif change < -2:
                return 'BEARISH'
            else:
                return 'NEUTRAL'
        
        df['recommendation'] = df.apply(get_recommendation, axis=1)
        df['trend'] = df.apply(get_trend, axis=1)
        df['risk_level'] = pd.cut(df['ai_score'], bins=[0, 33, 66, 100], labels=['HIGH', 'MEDIUM', 'LOW']).astype(str)
        
        # Top 3 unique recommendations
        top_recs = df.drop_duplicates(subset=['symbol']).nlargest(3, 'ai_score')
        
        # Batch process historical data for top symbols to avoid repeated API calls
        top_symbols = top_recs['symbol'].unique()[:3]  # Only top 3 unique symbols
        
        recommendations = []
        for _, row in top_recs.iterrows():
            # AI-based risk calculation using cached historical data
            current_price = float(row['last_price'])
            action = str(row['recommendation'])
            symbol = str(row.get('symbol', 'N/A'))
            
            # Get AI-calculated levels using real data only
            ai_levels = self.risk_calculator.calculate_ai_levels(symbol, current_price, action)
            
            if ai_levels:  # Only add if real data is available
                recommendations.append({
                    'symbol': str(row.get('symbol', 'N/A')),
                    'company_name': str(row.get('companyName', '')),
                    'last_price': current_price,
                    'stop_loss': ai_levels['stop_loss'],
                    'target': ai_levels['target'],
                    'sl_percentage': ai_levels['sl_percentage'],
                    'target_percentage': ai_levels['target_percentage'],
                    'risk_reward_ratio': ai_levels['risk_reward_ratio'],
                    'atr': ai_levels.get('atr', 0),
                    'volatility': ai_levels.get('volatility', 0),
                    'support': ai_levels.get('support', 0),
                    'resistance': ai_levels.get('resistance', 0),
                    'price_change': float(row['price_change']),
                    'ai_score': float(round(row['ai_score'], 2)),
                    'recommendation': action,
                    'trend': str(row['trend']),
                    'risk_level': str(row['risk_level']) if pd.notna(row['risk_level']) else 'MEDIUM',
                    'volume_percentile': float(round(row['volume_rank'] * 100, 1)),
                    'year_high_proximity': float(round(row['year_high_proximity'], 3)),
                    'total_traded_volume': int(row['volume']),
                    'trading_plan': {
                        "entry": f"{action} at ₹{current_price}",
                        "stop_loss": f"₹{ai_levels['stop_loss']} ({ai_levels['sl_percentage']}%)",
                        "target": f"₹{ai_levels['target']} ({ai_levels['target_percentage']}%)",
                        "risk_reward": f"1:{ai_levels['risk_reward_ratio']}",
                        "technical_levels": f"Support: ₹{ai_levels.get('support', 0)}, Resistance: ₹{ai_levels.get('resistance', 0)}"
                    } if action != 'HOLD' else None
                })
            else:  # No real data available - add with null values
                recommendations.append({
                    'symbol': str(row.get('symbol', 'N/A')),
                    'company_name': str(row.get('companyName', '')),
                    'last_price': current_price,
                    'stop_loss': None,
                    'target': None,
                    'sl_percentage': None,
                    'target_percentage': None,
                    'risk_reward_ratio': None,
                    'atr': None,
                    'volatility': None,
                    'support': None,
                    'resistance': None,
                    'price_change': float(row['price_change']),
                    'ai_score': float(round(row['ai_score'], 2)),
                    'recommendation': action,
                    'trend': str(row['trend']),
                    'risk_level': str(row['risk_level']) if pd.notna(row['risk_level']) else 'MEDIUM',
                    'volume_percentile': float(round(row['volume_rank'] * 100, 1)),
                    'year_high_proximity': float(round(row['year_high_proximity'], 3)),
                    'total_traded_volume': int(row['volume']),
                    'trading_plan': None
                })
        
        # Best trades with safety checks
        buy_candidates = df[df['recommendation'].isin(['BUY', 'STRONG BUY'])]
        sell_candidates = df[df['recommendation'].isin(['SELL', 'STRONG SELL'])]
        
        best_buy = buy_candidates.nlargest(1, 'ai_score') if not buy_candidates.empty else pd.DataFrame()
        best_sell = sell_candidates.nlargest(1, 'ai_score') if not sell_candidates.empty else pd.DataFrame()
        
        # Optimize trade plan creation to avoid duplicate calculations
        best_trades = {
            'primary_recommendation': recommendations[0] if recommendations else None,
            'best_buy_trade': self._create_trade_plan_optimized(best_buy.iloc[0]) if not best_buy.empty else None,
            'best_sell_trade': self._create_trade_plan_optimized(best_sell.iloc[0]) if not best_sell.empty else None
        }
        
        # Market analysis
        market_analysis = {
            'total_stocks': int(len(df)),
            'avg_ai_score': float(df['ai_score'].mean()),
            'high_confidence_trades': int(len(df[df['ai_score'] > 70])),
            'bullish_signals': int(len(df[df['trend'] == 'BULLISH'])),
            'bearish_signals': int(len(df[df['trend'] == 'BEARISH'])),
            'neutral_signals': int(len(df[df['trend'] == 'NEUTRAL'])),
            'volume_leaders': [str(x) for x in df.nlargest(3, 'volume')['symbol'].tolist()],
            'momentum_leaders': [str(x) for x in df.nlargest(3, 'price_momentum')['symbol'].tolist()]
        }
        
        market_sentiment = 'BULLISH' if market_analysis['bullish_signals'] > market_analysis['bearish_signals'] else 'BEARISH' if market_analysis['bearish_signals'] > market_analysis['bullish_signals'] else 'NEUTRAL'
        
        # Ensure unique recommendations
        unique_recommendations = self._ensure_unique_recommendations(recommendations)
        
        return {
            'best_trades': best_trades,
            'top_3_recommendations': unique_recommendations,
            'market_sentiment': market_sentiment,
            'confidence_level': 'HIGH' if market_analysis['avg_ai_score'] > 75 else 'MEDIUM' if market_analysis['avg_ai_score'] > 50 else 'LOW',
            'trading_strategy': f"Focus on {'buying strong performers' if market_sentiment == 'BULLISH' else 'selling weak stocks' if market_sentiment == 'BEARISH' else 'selective trading'}",
            'model_status': 'ML_ENHANCED' if self.is_trained else 'PANDAS_ONLY'
        }
    
    def save_model(self, filepath):
        """Save trained model"""
        if self.is_trained:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'trained_at': datetime.now().isoformat()
            }, filepath)
            return {"status": "saved", "path": filepath}
        return {"error": "No trained model to save"}
    
    def _create_trade_plan_optimized(self, row):
        """Create trade plan with real data only - return None if no real data"""
        current_price = float(row['last_price'])
        action = str(row['recommendation'])
        symbol = str(row['symbol'])
        
        # Get AI-calculated levels using real data only
        ai_levels = self.risk_calculator.calculate_ai_levels(symbol, current_price, action)
        
        if not ai_levels:
            # No real data available - return None
            return None
        
        return {
            'symbol': symbol,
            'action': action,
            'current_price': current_price,
            'stop_loss': ai_levels['stop_loss'],
            'target': ai_levels['target'],
            'risk_reward_ratio': ai_levels['risk_reward_ratio'],
            'ai_score': float(round(row['ai_score'], 2)),
            'atr': ai_levels.get('atr', 0),
            'volatility': ai_levels.get('volatility', 0),
            'trading_plan': {
                "entry": f"{action} at ₹{current_price}",
                "stop_loss": f"₹{ai_levels['stop_loss']} ({ai_levels['sl_percentage']}%)",
                "target": f"₹{ai_levels['target']} ({ai_levels['target_percentage']}%)",
                "risk_reward": f"1:{ai_levels['risk_reward_ratio']}",
                "technical_analysis": f"ATR: {ai_levels.get('atr', 0)}, Vol: {ai_levels.get('volatility', 0)}%"
            }
        }
    
    def _create_trade_plan(self, row):
        """Legacy method - redirects to optimized version"""
        return self._create_trade_plan_optimized(row)
    
    def _ensure_unique_recommendations(self, recommendations):
        """Ensure unique stock symbols in recommendations list"""
        seen_symbols = set()
        unique_recs = []
        
        for rec in recommendations:
            symbol = rec.get('symbol')
            if symbol and symbol not in seen_symbols:
                seen_symbols.add(symbol)
                unique_recs.append(rec)
        
        return unique_recs
    
    def load_model(self, filepath):
        """Load trained model"""
        try:
            data = joblib.load(filepath)
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_trained = True
            return {"status": "loaded", "trained_at": data.get('trained_at')}
        except Exception as e:
            return {"error": str(e)}