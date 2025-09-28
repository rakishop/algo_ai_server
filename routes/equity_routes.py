from fastapi import APIRouter, Query
from typing import Optional
from .ml_equity_model import EquityMLModel
import json

def create_equity_routes(nse_client):
    router = APIRouter(prefix="/api/v1/equity", tags=["equity"])
    ml_model = EquityMLModel()
    
    @router.get("/auto-ml-recommendations")
    def get_auto_equity_ml_recommendations(limit: Optional[int] = Query(50)):
        """Auto-train ML model with current equity data and get recommendations"""
        try:
            # Get current equity data (using gainers as example)
            current_data = nse_client.get_gainers_data()
            if "error" in current_data:
                return current_data
            
            # Auto-train model with current data
            training_result = ml_model.train_with_historical_data(current_data)
            
            # Get ML predictions
            predictions = ml_model.predict_recommendations(current_data)
            
            # Combine training info with predictions
            predictions["training_info"] = training_result
            predictions["auto_trained"] = True
            
            return predictions
            
        except Exception as e:
            return {"error": str(e)}
    
    @router.get("/gainers-analysis")
    def get_gainers_ml_analysis():
        """Get ML analysis of top gainers"""
        try:
            current_data = nse_client.get_gainers_data()
            if "error" in current_data:
                return current_data
            
            training_result = ml_model.train_with_historical_data(current_data)
            predictions = ml_model.predict_recommendations(current_data)
            predictions["data_source"] = "gainers"
            predictions["training_info"] = training_result
            
            return predictions
        except Exception as e:
            return {"error": str(e)}
    
    @router.get("/losers-analysis")
    def get_losers_ml_analysis():
        """Get ML analysis of top losers"""
        try:
            current_data = nse_client.get_losers_data()
            if "error" in current_data:
                return current_data
            
            training_result = ml_model.train_with_historical_data(current_data)
            predictions = ml_model.predict_recommendations(current_data)
            predictions["data_source"] = "losers"
            predictions["training_info"] = training_result
            
            return predictions
        except Exception as e:
            return {"error": str(e)}
    
    @router.get("/52week-high-analysis")
    def get_52week_high_analysis():
        """Get ML analysis of 52-week high stocks"""
        try:
            current_data = nse_client.get_52week_high_stocks_data()
            if "error" in current_data:
                return current_data
            
            training_result = ml_model.train_with_historical_data(current_data)
            predictions = ml_model.predict_recommendations(current_data)
            predictions["data_source"] = "52week_high"
            predictions["training_info"] = training_result
            
            return predictions
        except Exception as e:
            return {"error": str(e)}
    
    @router.get("/volume-gainers-analysis")
    def get_volume_gainers_analysis():
        """Get ML analysis of volume gainers"""
        try:
            current_data = nse_client.get_volume_gainers()
            if "error" in current_data:
                return current_data
            
            training_result = ml_model.train_with_historical_data(current_data)
            predictions = ml_model.predict_recommendations(current_data)
            predictions["data_source"] = "volume_gainers"
            predictions["training_info"] = training_result
            
            return predictions
        except Exception as e:
            return {"error": str(e)}
    
    return router