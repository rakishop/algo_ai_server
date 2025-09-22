from nse_client import NSEClient
from ai_stock_selector import AIStockSelector

try:
    nse = NSEClient()
    print("NSE Client initialized")
    
    gainers_data = nse.get_gainers_data()
    print(f"Gainers data: {len(gainers_data.get('data', []))} stocks")
    
    losers_data = nse.get_losers_data()
    print(f"Losers data: {len(losers_data.get('data', []))} stocks")
    
    ai_selector = AIStockSelector()
    top_gainers, top_losers = ai_selector.select_top_stocks(gainers_data, losers_data)
    
    print(f"Top gainers: {len(top_gainers)}")
    print(f"Top losers: {len(top_losers)}")
    
    if top_gainers:
        print(f"First gainer: {top_gainers[0]}")
    if top_losers:
        print(f"First loser: {top_losers[0]}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()