import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class RiskMetrics:
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    expected_shortfall: float
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    beta: float

class RiskManager:
    def __init__(self):
        self.risk_free_rate = float(os.getenv("RISK_FREE_RATE", "0.06"))  # Annual risk-free rate from env
        
    def calculate_portfolio_risk(self, returns: List[float], market_returns: List[float] = None) -> RiskMetrics:
        """Calculate comprehensive risk metrics for a portfolio"""
        if not returns or len(returns) < 10:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0)
        
        returns_array = np.array(returns)
        
        # Value at Risk calculations
        var_95 = np.percentile(returns_array, 5)
        var_99 = np.percentile(returns_array, 1)
        
        # Expected Shortfall (Conditional VaR)
        tail_losses = returns_array[returns_array <= var_95]
        expected_shortfall = np.mean(tail_losses) if len(tail_losses) > 0 else 0
        
        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Volatility (annualized)
        volatility = np.std(returns_array) * np.sqrt(252)
        
        # Sharpe Ratio
        excess_returns = np.mean(returns_array) - (self.risk_free_rate / 252)
        sharpe_ratio = excess_returns / np.std(returns_array) if np.std(returns_array) > 0 else 0
        sharpe_ratio *= np.sqrt(252)  # Annualized
        
        # Beta calculation
        beta = 1.0
        if market_returns and len(market_returns) == len(returns):
            market_array = np.array(market_returns)
            covariance = np.cov(returns_array, market_array)[0, 1]
            market_variance = np.var(market_array)
            beta = covariance / market_variance if market_variance > 0 else 1.0
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            volatility=volatility,
            beta=beta
        )
    
    def calculate_position_size(self, portfolio_value: float, risk_per_trade: float, 
                              entry_price: float, stop_loss: float, risk_tolerance: str) -> Dict:
        """Calculate optimal position size based on risk management"""
        
        # Risk tolerance multipliers
        risk_multipliers = {"LOW": 0.5, "MEDIUM": 1.0, "HIGH": 1.5}
        adjusted_risk = risk_per_trade * risk_multipliers.get(risk_tolerance, 1.0)
        
        # Maximum risk amount
        max_risk_amount = portfolio_value * adjusted_risk
        
        # Risk per share
        if stop_loss and entry_price > stop_loss:
            risk_per_share = entry_price - stop_loss
            max_shares = int(max_risk_amount / risk_per_share)
        else:
            # Default 2% stop loss if not provided
            risk_per_share = entry_price * 0.02
            max_shares = int(max_risk_amount / risk_per_share)
        
        # Position value limits (max 20% of portfolio)
        max_position_value = portfolio_value * 0.20
        max_shares_by_value = int(max_position_value / entry_price)
        
        recommended_shares = min(max_shares, max_shares_by_value)
        position_value = recommended_shares * entry_price
        
        return {
            "recommended_shares": recommended_shares,
            "position_value": position_value,
            "risk_amount": recommended_shares * risk_per_share,
            "risk_percentage": (recommended_shares * risk_per_share) / portfolio_value * 100,
            "position_percentage": position_value / portfolio_value * 100,
            "stop_loss_price": stop_loss or (entry_price * 0.98)
        }
    
    def assess_market_risk(self, market_data: List[Dict]) -> Dict:
        """Assess overall market risk conditions"""
        if not market_data:
            return {"error": "No market data provided"}
        
        # Extract price changes
        price_changes = []
        volumes = []
        
        for stock in market_data:
            change = stock.get('perChange', 0) or stock.get('pChange', 0)
            volume = stock.get('trade_quantity', 0) or stock.get('totalTradedVolume', 0)
            
            if change is not None:
                price_changes.append(float(change))
            if volume:
                volumes.append(float(volume))
        
        if not price_changes:
            return {"error": "No valid price change data"}
        
        # Market volatility
        market_volatility = np.std(price_changes)
        
        # Market sentiment
        positive_moves = len([x for x in price_changes if x > 0])
        negative_moves = len([x for x in price_changes if x < 0])
        sentiment_ratio = positive_moves / len(price_changes) if price_changes else 0.5
        
        # Volume analysis
        avg_volume = np.mean(volumes) if volumes else 0
        high_volume_stocks = len([v for v in volumes if v > avg_volume * 1.5])
        
        # Risk level assessment
        if market_volatility > 3.0:
            risk_level = "HIGH"
        elif market_volatility > 1.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            "market_volatility": market_volatility,
            "risk_level": risk_level,
            "sentiment_ratio": sentiment_ratio,
            "market_sentiment": "BULLISH" if sentiment_ratio > 0.6 else "BEARISH" if sentiment_ratio < 0.4 else "NEUTRAL",
            "avg_volume": avg_volume,
            "high_volume_stocks": high_volume_stocks,
            "total_stocks_analyzed": len(market_data),
            "recommendation": self._get_risk_recommendation(risk_level, sentiment_ratio)
        }
    
    def _get_risk_recommendation(self, risk_level: str, sentiment_ratio: float) -> str:
        """Get trading recommendation based on risk assessment"""
        if risk_level == "HIGH":
            return "High volatility detected. Reduce position sizes and use tight stop losses."
        elif risk_level == "MEDIUM" and sentiment_ratio > 0.6:
            return "Moderate volatility with bullish sentiment. Normal position sizing recommended."
        elif risk_level == "MEDIUM" and sentiment_ratio < 0.4:
            return "Moderate volatility with bearish sentiment. Consider defensive positions."
        elif risk_level == "LOW" and sentiment_ratio > 0.6:
            return "Low volatility with bullish sentiment. Good conditions for larger positions."
        else:
            return "Mixed market conditions. Use standard risk management practices."
    
    def calculate_correlation_risk(self, portfolio_positions: List[Dict], market_data: Dict) -> Dict:
        """Calculate correlation risk between portfolio positions"""
        if len(portfolio_positions) < 2:
            return {"correlation_risk": "LOW", "diversification_score": 100}
        
        # Extract symbols and their price changes
        symbols = [pos['symbol'] for pos in portfolio_positions]
        price_changes = {}
        
        for symbol in symbols:
            # Find price change data for each symbol
            for stock in market_data.get('data', []):
                if stock.get('symbol') == symbol:
                    price_changes[symbol] = stock.get('perChange', 0)
                    break
        
        if len(price_changes) < 2:
            return {"correlation_risk": "UNKNOWN", "diversification_score": 50}
        
        # Calculate average correlation (simplified)
        changes = list(price_changes.values())
        avg_correlation = np.corrcoef(changes)[0, 1] if len(changes) >= 2 else 0
        
        # Diversification score
        diversification_score = max(0, 100 - abs(avg_correlation) * 100)
        
        if abs(avg_correlation) > 0.8:
            correlation_risk = "HIGH"
        elif abs(avg_correlation) > 0.5:
            correlation_risk = "MEDIUM"
        else:
            correlation_risk = "LOW"
        
        return {
            "correlation_risk": correlation_risk,
            "average_correlation": avg_correlation,
            "diversification_score": diversification_score,
            "recommendation": "Consider diversifying across different sectors" if correlation_risk == "HIGH" else "Good diversification"
        }