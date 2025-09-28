from fastapi import APIRouter

def create_indices_routes(nse_client):
    router = APIRouter(prefix="/api/v1/indices", tags=["indices"])
    
    @router.get("/live-data")
    def get_all_indices():
        return nse_client.get_all_indices()
    
    @router.get("/constituents/{index_name}")
    def get_index_constituents_with_52w_data(index_name: str):
        """Get index constituents with their 52-week high/low data"""
        try:
            # Get index constituents using the exact index name from /available endpoint
            constituents_data = nse_client.get_index_constituents(index_name)
            
            if "error" in constituents_data:
                return constituents_data
            
            # Extract constituent symbols and their data
            if "data" not in constituents_data:
                return {"error": "No data found for index", "index_name": index_name}
            
            constituents = constituents_data["data"]
            
            # Process each constituent to include 52-week data
            processed_constituents = []
            stocks_at_52w_high = []
            stocks_at_52w_low = []
            
            for stock in constituents:
                stock_info = {
                    "symbol": stock.get("symbol"),
                    "ltp": stock.get("lastPrice"),
                    "perChange": stock.get("pChange"),
                    "high_52w": stock.get("yearHigh"),
                    "low_52w": stock.get("yearLow"),
                    "volume": stock.get("totalTradedVolume", 0),
                    "value": stock.get("totalTradedValue", 0)
                }
                
                processed_constituents.append(stock_info)
                
                # Check if at 52-week high/low
                if stock.get("lastPrice") == stock.get("yearHigh"):
                    stocks_at_52w_high.append(stock_info)
                
                if stock.get("lastPrice") == stock.get("yearLow"):
                    stocks_at_52w_low.append(stock_info)
            
            return {
                "index_name": index_name,
                "total_constituents": len(processed_constituents),
                "constituents_data": sorted(processed_constituents, key=lambda x: x.get("perChange", 0), reverse=True),
                "stocks_at_52w_high": stocks_at_52w_high,
                "stocks_at_52w_low": stocks_at_52w_low,
                "high_count": len(stocks_at_52w_high),
                "low_count": len(stocks_at_52w_low),
                "analysis_time": "real-time"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    @router.get("/available")
    def get_available_indices_list():
        """Get list of available indices from NSE"""
        try:
            indices_data = nse_client.get_all_indices()
            if "error" in indices_data:
                return indices_data
            
            available_indices = []
            if "data" in indices_data:
                for index in indices_data["data"]:
                    available_indices.append({
                        "index_name": index.get("index"),
                        "index_symbol": index.get("indexSymbol"),
                        "category": index.get("key")
                    })
            
            return {
                "available_indices": available_indices,
                "description": "Use index_symbol with /constituents/{index_name} endpoint"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    return router