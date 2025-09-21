from fastapi import FastAPI, Query
from typing import Optional, List
from data_processor import DataProcessor
from nse_client import NSEClient

class FilteredEndpoints:
    def __init__(self, app: FastAPI, nse_client: NSEClient):
        self.app = app
        self.nse_client = nse_client
        self.processor = DataProcessor()
        self.setup_filtered_endpoints()
    
    def setup_filtered_endpoints(self):
        """Setup filtered API endpoints"""
        
        @self.app.get("/api/v1/filtered/top-gainers")
        def get_top_gainers(limit: int = Query(10, ge=1, le=50)):
            """Get top gaining stocks"""
            gainers_data = self.nse_client.get_gainers_data()
            stocks = self.processor.extract_stock_data(gainers_data)
            top_gainers = self.processor.filter_top_performers(stocks, "pChange", limit)
            return {"top_gainers": top_gainers, "count": len(top_gainers)}
        
        @self.app.get("/api/v1/filtered/top-losers")
        def get_top_losers(limit: int = Query(10, ge=1, le=50)):
            """Get top losing stocks"""
            losers_data = self.nse_client.get_losers_data()
            stocks = self.processor.extract_stock_data(losers_data)
            top_losers = self.processor.filter_top_performers(stocks, "pChange", limit)
            top_losers.reverse()  # Reverse to show biggest losers first
            return {"top_losers": top_losers, "count": len(top_losers)}
        
        @self.app.get("/api/v1/filtered/high-volume-stocks")
        def get_high_volume_stocks(
            min_volume: int = Query(1000000, ge=0),
            limit: int = Query(20, ge=1, le=100)
        ):
            """Get stocks with high trading volume"""
            volume_data = self.nse_client.get_volume_gainers()
            stocks = self.processor.extract_stock_data(volume_data)
            high_volume = self.processor.filter_by_volume(stocks, min_volume)[:limit]
            return {"high_volume_stocks": high_volume, "count": len(high_volume)}
        
        @self.app.get("/api/v1/filtered/price-range-stocks")
        def get_stocks_by_price_range(
            min_price: float = Query(0, ge=0),
            max_price: float = Query(10000, ge=0),
            limit: int = Query(50, ge=1, le=200)
        ):
            """Get stocks within specified price range"""
            # Get data from multiple sources
            gainers_data = self.nse_client.get_gainers_data()
            losers_data = self.nse_client.get_losers_data()
            
            all_stocks = []
            all_stocks.extend(self.processor.extract_stock_data(gainers_data))
            all_stocks.extend(self.processor.extract_stock_data(losers_data))
            
            # Remove duplicates based on symbol
            unique_stocks = {}
            for stock in all_stocks:
                symbol = stock.get("symbol")
                if symbol and symbol not in unique_stocks:
                    unique_stocks[symbol] = stock
            
            filtered_stocks = self.processor.filter_by_price_range(
                list(unique_stocks.values()), min_price, max_price
            )[:limit]
            
            return {"stocks_in_range": filtered_stocks, "count": len(filtered_stocks)}
        
        @self.app.get("/api/v1/filtered/52week-high-performers")
        def get_52week_high_performers(limit: int = Query(15, ge=1, le=50)):
            """Get top performers from 52-week high list"""
            high_data = self.nse_client.get_52week_high_stocks_data()
            stocks = self.processor.extract_stock_data(high_data)
            top_performers = self.processor.filter_top_performers(stocks, "pChange", limit)
            return {"high_performers": top_performers, "count": len(top_performers)}
        
        @self.app.get("/api/v1/filtered/active-large-cap")
        def get_active_large_cap(
            min_price: float = Query(500, ge=0),
            min_volume: int = Query(500000, ge=0),
            limit: int = Query(20, ge=1, le=50)
        ):
            """Get active large-cap stocks"""
            active_data = self.nse_client.get_most_active_securities()
            stocks = self.processor.extract_stock_data(active_data)
            
            # Filter by price and volume
            large_cap = self.processor.filter_by_price_range(stocks, min_price, float('inf'))
            active_large_cap = self.processor.filter_by_volume(large_cap, min_volume)[:limit]
            
            return {"active_large_cap": active_large_cap, "count": len(active_large_cap)}
        
        @self.app.get("/api/v1/filtered/momentum-stocks")
        def get_momentum_stocks(
            min_change: float = Query(2.0, ge=0),
            min_volume: int = Query(100000, ge=0),
            limit: int = Query(25, ge=1, le=100)
        ):
            """Get momentum stocks with significant price change and volume"""
            gainers_data = self.nse_client.get_gainers_data()
            volume_data = self.nse_client.get_volume_gainers()
            
            all_stocks = []
            all_stocks.extend(self.processor.extract_stock_data(gainers_data))
            all_stocks.extend(self.processor.extract_stock_data(volume_data))
            
            # Filter momentum stocks
            momentum_stocks = []
            for stock in all_stocks:
                pchange = stock.get("pChange", 0)
                volume = stock.get("volume", 0)
                
                try:
                    if float(pchange) >= min_change and int(volume) >= min_volume:
                        momentum_stocks.append(stock)
                except (ValueError, TypeError):
                    continue
            
            # Remove duplicates and limit
            unique_momentum = {}
            for stock in momentum_stocks:
                symbol = stock.get("symbol")
                if symbol and symbol not in unique_momentum:
                    unique_momentum[symbol] = stock
            
            result = list(unique_momentum.values())[:limit]
            return {"momentum_stocks": result, "count": len(result)}
        
        @self.app.get("/api/v1/filtered/market-summary")
        def get_market_summary():
            """Get comprehensive market summary with key metrics"""
            try:
                # Get data from multiple endpoints
                gainers_data = self.nse_client.get_gainers_data()
                losers_data = self.nse_client.get_losers_data()
                advance_data = self.nse_client.get_advance_decline()
                volume_data = self.nse_client.get_volume_gainers()
                
                # Extract and process data
                gainers = self.processor.extract_stock_data(gainers_data)
                losers = self.processor.extract_stock_data(losers_data)
                volume_leaders = self.processor.extract_stock_data(volume_data)
                
                summary = {
                    "market_breadth": advance_data,
                    "top_gainers": self.processor.filter_top_performers(gainers, "pChange", 5),
                    "top_losers": self.processor.filter_top_performers(losers, "pChange", 5)[::-1],
                    "volume_leaders": volume_leaders[:5],
                    "total_gainers": len(gainers),
                    "total_losers": len(losers),
                    "timestamp": self.processor.data_storage.get("summary", [{"timestamp": "N/A"}])[-1]["timestamp"] if self.processor.data_storage.get("summary") else "N/A"
                }
                
                return summary
            except Exception as e:
                return {"error": str(e), "summary": None}