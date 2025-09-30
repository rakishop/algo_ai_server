import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from ml_analyzer import MLStockAnalyzer
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    entry_date: str
    position_type: str  # 'LONG' or 'SHORT'
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    current_price: Optional[float] = None

@dataclass
class Portfolio:
    portfolio_id: str
    name: str
    total_capital: float
    available_cash: float
    positions: List[Position]
    created_date: str
    risk_tolerance: str  # 'LOW', 'MEDIUM', 'HIGH'

class PortfolioManager:
    def __init__(self):
        self.portfolios: Dict[str, Portfolio] = {}
        self.ml_analyzer = MLStockAnalyzer()
        self.load_portfolios()
    
    def create_portfolio(self, portfolio_id: str, name: str, capital: float, risk_tolerance: str = 'MEDIUM') -> Dict:
        if portfolio_id in self.portfolios:
            return {"error": "Portfolio already exists"}
        
        portfolio = Portfolio(
            portfolio_id=portfolio_id,
            name=name,
            total_capital=capital,
            available_cash=capital,
            positions=[],
            created_date=datetime.now().isoformat(),
            risk_tolerance=risk_tolerance
        )
        
        self.portfolios[portfolio_id] = portfolio
        self.save_portfolios()
        return {"status": "success", "portfolio": asdict(portfolio)}
    
    def add_position(self, portfolio_id: str, symbol: str, quantity: int, entry_price: float, 
                    position_type: str = 'LONG', stop_loss: float = None, target_price: float = None) -> Dict:
        if portfolio_id not in self.portfolios:
            return {"error": "Portfolio not found"}
        
        portfolio = self.portfolios[portfolio_id]
        investment_amount = quantity * entry_price
        
        if investment_amount > portfolio.available_cash:
            return {"error": "Insufficient funds"}
        
        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            entry_date=datetime.now().isoformat(),
            position_type=position_type,
            stop_loss=stop_loss,
            target_price=target_price
        )
        
        portfolio.positions.append(position)
        portfolio.available_cash -= investment_amount
        self.save_portfolios()
        
        return {"status": "success", "position": asdict(position)}
    
    def get_portfolio_performance(self, portfolio_id: str, current_prices: Dict[str, float]) -> Dict:
        if portfolio_id not in self.portfolios:
            return {"error": "Portfolio not found"}
        
        portfolio = self.portfolios[portfolio_id]
        total_invested = 0
        current_value = 0
        positions_data = []
        
        for position in portfolio.positions:
            invested = position.quantity * position.entry_price
            current_price = current_prices.get(position.symbol, position.entry_price)
            position_value = position.quantity * current_price
            
            pnl = position_value - invested
            pnl_pct = (pnl / invested) * 100 if invested > 0 else 0
            
            total_invested += invested
            current_value += position_value
            
            positions_data.append({
                "symbol": position.symbol,
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "current_price": current_price,
                "invested_amount": invested,
                "current_value": position_value,
                "pnl": pnl,
                "pnl_percentage": pnl_pct,
                "position_type": position.position_type,
                "days_held": (datetime.now() - datetime.fromisoformat(position.entry_date)).days
            })
        
        total_portfolio_value = current_value + portfolio.available_cash
        total_pnl = current_value - total_invested
        total_pnl_pct = (total_pnl / total_invested) * 100 if total_invested > 0 else 0
        
        return {
            "portfolio_id": portfolio_id,
            "portfolio_name": portfolio.name,
            "total_capital": portfolio.total_capital,
            "available_cash": portfolio.available_cash,
            "total_invested": total_invested,
            "current_portfolio_value": total_portfolio_value,
            "total_pnl": total_pnl,
            "total_pnl_percentage": total_pnl_pct,
            "positions": positions_data,
            "risk_tolerance": portfolio.risk_tolerance
        }
    
    def get_portfolio_recommendations(self, portfolio_id: str, market_data: List[Dict]) -> Dict:
        if portfolio_id not in self.portfolios:
            return {"error": "Portfolio not found"}
        
        portfolio = self.portfolios[portfolio_id]
        
        # Risk-based position sizing
        risk_multiplier = {"LOW": 0.02, "MEDIUM": 0.05, "HIGH": 0.10}[portfolio.risk_tolerance]
        max_position_size = portfolio.total_capital * risk_multiplier
        
        # Analyze market data for recommendations
        recommendations = []
        for stock in market_data[:10]:  # Top 10 stocks
            features = self.ml_analyzer.extract_features(stock)
            
            # Check if already in portfolio
            existing_position = next((p for p in portfolio.positions if p.symbol == stock.get('symbol')), None)
            
            if not existing_position and features.get('scalping_score', 0) > 60:
                recommended_quantity = int(max_position_size / stock.get('ltp', 1))
                recommendations.append({
                    "symbol": stock.get('symbol'),
                    "action": "BUY",
                    "recommended_quantity": recommended_quantity,
                    "current_price": stock.get('ltp'),
                    "confidence": features.get('scalping_score'),
                    "reason": f"High ML score: {features.get('scalping_score'):.1f}",
                    "risk_level": portfolio.risk_tolerance
                })
        
        return {
            "portfolio_id": portfolio_id,
            "recommendations": recommendations[:5],
            "available_cash": portfolio.available_cash,
            "max_position_size": max_position_size
        }
    
    def save_portfolios(self):
        try:
            data = {}
            for pid, portfolio in self.portfolios.items():
                data[pid] = asdict(portfolio)
            
            data_path = os.getenv("TRAINING_DATA_PATH", "./data/")
            os.makedirs(data_path, exist_ok=True)
            
            with open(f'{data_path}portfolios.json', 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving portfolios: {e}")
    
    def load_portfolios(self):
        try:
            data_path = os.getenv("TRAINING_DATA_PATH", "./data/")
            with open(f'{data_path}portfolios.json', 'r') as f:
                data = json.load(f)
            
            for pid, portfolio_data in data.items():
                positions = [Position(**pos) for pos in portfolio_data['positions']]
                portfolio_data['positions'] = positions
                self.portfolios[pid] = Portfolio(**portfolio_data)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error loading portfolios: {e}")