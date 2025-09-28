import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
from datetime import datetime

class DerivativesMLModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data_dir = "training_data"
        self.model_file = "trained_model.pkl"
        self.last_training_time = None
        self.training_interval = 300  # Retrain every 5 minutes
        self.max_training_files = 100  # Use only latest 100 files
        os.makedirs(self.training_data_dir, exist_ok=True)
        
        # Try to load existing model
        if os.path.exists(self.model_file):
            load_result = self.load_model(self.model_file)
            if load_result.get("status") == "loaded":
                print(f"Loaded existing model: {load_result}")
                self.last_training_time = datetime.now().timestamp()
        
    def prepare_features(self, data):
        """Extract features from derivatives data"""
        features = []
        for contract in data:
            # Handle both API formats
            volume = contract.get('numberOfContractsTraded', contract.get('noOfTrades', contract.get('volume', 0)))
            price_change = contract.get('pChange', 0)
            oi = contract.get('openInterest', 0)
            last_price = contract.get('lastPrice', 0)
            strike = contract.get('strikePrice', 0)
            underlying_value = contract.get('underlyingValue', 0)
            premium_turnover = contract.get('premiumTurnover', contract.get('premiumTurnOver', 0))
            
            # Calculate derived features
            volume_ratio = volume / 1000000 if volume > 0 else 0
            oi_ratio = oi / 50000 if oi > 0 else 0
            moneyness = abs(underlying_value - strike) / underlying_value if underlying_value > 0 else 0
            price_momentum = abs(price_change) / 100
            liquidity_score = (volume * last_price) / 1000000 if last_price > 0 else 0
            
            features.append([
                volume_ratio,
                price_momentum,
                oi_ratio,
                moneyness,
                liquidity_score,
                1 if contract.get('optionType') == 'Call' else 0,
                premium_turnover / 1000000
            ])
        
        return np.array(features)
    
    def create_labels(self, data):
        """Create training labels based on trading success criteria"""
        labels = []
        for contract in data:
            volume = contract.get('numberOfContractsTraded', 0)
            price_change = contract.get('pChange', 0)
            oi = contract.get('openInterest', 0)
            
            # Label as 1 (good trade) if meets criteria
            score = 0
            if volume > 2000000: score += 1
            if abs(price_change) > 50: score += 1
            if oi > 50000: score += 1
            
            labels.append(1 if score >= 2 else 0)
        
        return np.array(labels)
    
    def train(self, historical_data):
        """Train model on historical derivatives data"""
        all_features = []
        all_labels = []
        
        print(f"Processing {len(historical_data)} datasets for training")
        
        for i, day_data in enumerate(historical_data):
            contracts = []
            if 'volume' in day_data and 'data' in day_data['volume']:
                contracts = day_data['volume']['data']
            elif 'data' in day_data:
                contracts = day_data['data']
            elif isinstance(day_data, list):
                contracts = day_data
            
            if contracts:
                features = self.prepare_features(contracts)
                labels = self.create_labels(contracts)
                
                print(f"Dataset {i+1}: {len(contracts)} contracts, {len(features)} features")
                
                all_features.extend(features)
                all_labels.extend(labels)
        
        print(f"Total features: {len(all_features)}, Total labels: {len(all_labels)}")
        
        if len(all_features) > 0:
            X = np.array(all_features)
            y = np.array(all_labels)
            
            print(f"Training with X shape: {X.shape}, y shape: {y.shape}")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            print(f"Model trained successfully with {len(X)} samples")
            return {"status": "trained", "samples": len(X)}
        
        print("Training failed: No features extracted")
        return {"status": "failed", "error": "No training data"}
    
    def predict_recommendations(self, current_data):
        """AI-powered recommendations using pandas analysis and ML"""
        if 'volume' not in current_data or 'data' not in current_data['volume']:
            return {"error": "Invalid data format"}
        
        contracts = current_data['volume']['data']
        df = pd.DataFrame(contracts)
        
        # Pandas-based feature engineering
        df['volume_rank'] = df['numberOfContractsTraded'].rank(pct=True)
        df['price_momentum'] = abs(df['pChange'])
        df['oi_rank'] = df['openInterest'].rank(pct=True)
        df['moneyness'] = abs(df['underlyingValue'] - df['strikePrice']) / df['underlyingValue']
        df['liquidity_score'] = df['numberOfContractsTraded'] * df['lastPrice']
        df['premium_efficiency'] = df['premiumTurnover'] / df['numberOfContractsTraded']
        
        # AI Scoring using pandas operations
        df['ai_score'] = (
            df['volume_rank'] * 40 +
            (df['price_momentum'] / df['price_momentum'].max()) * 30 +
            df['oi_rank'] * 20 +
            (1 - df['moneyness']) * 10
        )
        
        # ML predictions if model is trained
        if self.is_trained:
            features = self.prepare_features(contracts)
            if len(features) > 0:
                X_scaled = self.scaler.transform(features)
                ml_predictions = self.model.predict_proba(X_scaled)
                df['ml_probability'] = [prob[1] for prob in ml_predictions]
                df['ai_score'] = df['ai_score'] * 0.6 + df['ml_probability'] * 100 * 0.4
        
        # Generate recommendations using pandas
        def get_recommendation(row):
            if row['optionType'] == 'Call':
                return 'BUY CALL' if row['pChange'] > 0 else 'SELL CALL'
            else:
                return 'BUY PUT' if row['pChange'] > 0 else 'SELL PUT'
        
        def get_trend(row):
            if row['optionType'] == 'Call':
                return 'BULLISH' if row['pChange'] > 0 else 'BEARISH'
            else:
                return 'BEARISH' if row['pChange'] > 0 else 'BULLISH'
        
        df['recommendation'] = df.apply(get_recommendation, axis=1)
        df['trend'] = df.apply(get_trend, axis=1)
        df['risk_level'] = pd.cut(df['ai_score'], bins=[0, 33, 66, 100], labels=['HIGH', 'MEDIUM', 'LOW'])
        
        # Top 3 best recommendations only
        top_recs = df.nlargest(3, 'ai_score')
        
        recommendations = []
        for _, row in top_recs.iterrows():
            rec = {
                'identifier': str(row['identifier']),
                'underlying': str(row['underlying']),
                'option_type': str(row['optionType']),
                'strike_price': float(row['strikePrice']),
                'ai_score': float(round(row['ai_score'], 2)),
                'recommendation': str(row['recommendation']),
                'trend': str(row['trend']),
                'risk_level': str(row['risk_level']),
                'volume_percentile': float(round(row['volume_rank'] * 100, 1)),
                'price_change': float(row['pChange']),
                'liquidity_score': float(row['liquidity_score']),
                'last_price': float(row['lastPrice']),
                'expiry_date': str(row.get('expiryDate', ''))
            }
            
            # Add historical analysis if available
            if 'historical_analysis' in row and pd.notna(row['historical_analysis']):
                rec['historical_analysis'] = row['historical_analysis']
            
            recommendations.append(rec)
        
        # Market analysis using pandas
        market_analysis = {
            'total_contracts': int(len(df)),
            'avg_ai_score': float(df['ai_score'].mean()),
            'high_confidence_trades': int(len(df[df['ai_score'] > 70])),
            'call_put_ratio': float(len(df[df['optionType'] == 'Call']) / len(df[df['optionType'] == 'Put'])),
            'bullish_signals': int(len(df[df['trend'] == 'BULLISH'])),
            'bearish_signals': int(len(df[df['trend'] == 'BEARISH'])),
            'volume_leaders': [str(x) for x in df.nlargest(3, 'numberOfContractsTraded')['identifier'].tolist()],
            'momentum_leaders': [str(x) for x in df.nlargest(3, 'price_momentum')['identifier'].tolist()]
        }
        
        # Best strike selection
        best_call = df[df['optionType'] == 'Call'].nlargest(1, 'ai_score')
        best_put = df[df['optionType'] == 'Put'].nlargest(1, 'ai_score')
        
        # Add historical analysis to primary recommendation
        primary_rec = recommendations[0] if recommendations else None
        if primary_rec:
            # Find matching contract in original data for historical analysis
            for contract in contracts:
                if contract.get('identifier') == primary_rec['identifier']:
                    if 'historical_analysis' in contract:
                        primary_rec['historical_analysis'] = contract['historical_analysis']
                    break
        
        best_trades = {
            'primary_recommendation': primary_rec,
            'best_call_trade': {
                'strike': float(best_call.iloc[0]['strikePrice']) if not best_call.empty else None,
                'action': str(best_call.iloc[0]['recommendation']) if not best_call.empty else None,
                'ai_score': float(round(best_call.iloc[0]['ai_score'], 2)) if not best_call.empty else None
            } if not best_call.empty else None,
            'best_put_trade': {
                'strike': float(best_put.iloc[0]['strikePrice']) if not best_put.empty else None,
                'action': str(best_put.iloc[0]['recommendation']) if not best_put.empty else None,
                'ai_score': float(round(best_put.iloc[0]['ai_score'], 2)) if not best_put.empty else None
            } if not best_put.empty else None
        }
        
        return {
            'best_trades': best_trades,
            'top_3_recommendations': recommendations,
            'market_sentiment': 'BEARISH' if market_analysis['bearish_signals'] > market_analysis['bullish_signals'] else 'BULLISH',
            'confidence_level': 'HIGH' if market_analysis['avg_ai_score'] > 75 else 'MEDIUM' if market_analysis['avg_ai_score'] > 50 else 'LOW',
            'trading_strategy': f"Focus on {'PUT buying' if market_analysis['bearish_signals'] > market_analysis['bullish_signals'] else 'CALL buying'}",
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
    
    def save_training_data(self, data):
        """Save current data for training"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"derivatives_data_{timestamp}.json"
        filepath = os.path.join(self.training_data_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        return {"status": "saved", "file": filename}
    
    def load_all_training_data(self):
        """Load all historical training data"""
        all_data = []
        
        if not os.path.exists(self.training_data_dir):
            return all_data
        
        for filename in os.listdir(self.training_data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.training_data_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        all_data.append(data)
                except Exception as e:
                    continue
        
        return all_data
    
    def should_retrain(self):
        """Check if model should be retrained"""
        if not self.is_trained:
            return True
        
        if self.last_training_time is None:
            return True
            
        # Retrain every 5 minutes
        time_since_training = datetime.now().timestamp() - self.last_training_time
        return time_since_training > self.training_interval
    
    def load_recent_training_data(self):
        """Load only recent training data to avoid performance issues"""
        if not os.path.exists(self.training_data_dir):
            return []
        
        # Get all files sorted by modification time (newest first)
        files = []
        for filename in os.listdir(self.training_data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.training_data_dir, filename)
                mtime = os.path.getmtime(filepath)
                files.append((mtime, filepath))
        
        # Sort by modification time (newest first) and take only recent files
        files.sort(reverse=True)
        recent_files = files[:self.max_training_files]
        
        # Load recent data
        recent_data = []
        for _, filepath in recent_files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    recent_data.append(data)
            except Exception as e:
                continue
        
        return recent_data
    
    def train_with_historical_data(self, current_data=None):
        """Smart training with performance optimization"""
        # Save current data
        if current_data:
            save_result = self.save_training_data(current_data)
            print(f"Saved current data: {save_result}")
        
        # Check if retraining is needed
        if not self.should_retrain():
            print("Using existing trained model (no retraining needed)")
            return {"status": "using_existing", "last_trained": self.last_training_time}
        
        # Load only recent data for training
        historical_data = self.load_recent_training_data()
        if current_data:
            historical_data.append(current_data)
        
        print(f"Training with {len(historical_data)} recent datasets (max: {self.max_training_files})")
        
        # Train with recent data only
        if len(historical_data) > 0:
            training_result = self.train(historical_data)
            self.last_training_time = datetime.now().timestamp()
            
            # Save trained model
            self.save_model(self.model_file)
            
            print(f"Training result: {training_result}")
            return training_result
        
        return {"status": "failed", "error": "No training data available"}
    
    def analyze_data_with_pandas(self, data):
        """Perform pandas-based analysis on derivatives data"""
        if 'volume' not in data or 'data' not in data['volume']:
            return {"error": "Invalid data format"}
        
        contracts = data['volume']['data']
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(contracts)
        
        # Basic statistics
        analysis = {
            "total_contracts": len(df),
            "volume_stats": {
                "mean_volume": df['numberOfContractsTraded'].mean(),
                "median_volume": df['numberOfContractsTraded'].median(),
                "max_volume": df['numberOfContractsTraded'].max(),
                "min_volume": df['numberOfContractsTraded'].min()
            },
            "price_change_stats": {
                "mean_change": df['pChange'].mean(),
                "median_change": df['pChange'].median(),
                "max_gain": df['pChange'].max(),
                "max_loss": df['pChange'].min(),
                "positive_changes": (df['pChange'] > 0).sum(),
                "negative_changes": (df['pChange'] < 0).sum()
            },
            "open_interest_stats": {
                "mean_oi": df['openInterest'].mean(),
                "median_oi": df['openInterest'].median(),
                "max_oi": df['openInterest'].max()
            },
            "option_type_distribution": df['optionType'].value_counts().to_dict(),
            "underlying_distribution": df['underlying'].value_counts().to_dict(),
            "high_volume_contracts": len(df[df['numberOfContractsTraded'] > df['numberOfContractsTraded'].quantile(0.75)]),
            "high_movement_contracts": len(df[abs(df['pChange']) > abs(df['pChange']).quantile(0.75)]),
            "correlation_analysis": {
                "volume_price_corr": df['numberOfContractsTraded'].corr(df['pChange']),
                "oi_volume_corr": df['openInterest'].corr(df['numberOfContractsTraded'])
            }
        }
        
        # Top performers
        analysis["top_gainers"] = df.nlargest(5, 'pChange')[['identifier', 'pChange', 'numberOfContractsTraded']].to_dict('records')
        analysis["top_losers"] = df.nsmallest(5, 'pChange')[['identifier', 'pChange', 'numberOfContractsTraded']].to_dict('records')
        analysis["highest_volume"] = df.nlargest(5, 'numberOfContractsTraded')[['identifier', 'numberOfContractsTraded', 'pChange']].to_dict('records')
        
        return analysis