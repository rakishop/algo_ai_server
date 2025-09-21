from ai_endpoints import AIEndpoints
from nse_client import NSEClient
from fastapi import FastAPI

def test_ai_simple():
    app = FastAPI()
    nse_client = NSEClient()
    ai_endpoints = AIEndpoints(app, nse_client)
    
    print("Testing AI Model...")
    
    try:
        gainers = nse_client.get_gainers_data()
        processor = ai_endpoints.processor
        stocks = processor.extract_stock_data(gainers)
        
        if stocks:
            low_risk = ai_endpoints._filter_by_risk_level(stocks, "low")
            medium_risk = ai_endpoints._filter_by_risk_level(stocks, "medium") 
            high_risk = ai_endpoints._filter_by_risk_level(stocks, "high")
            
            print(f"Risk filtering results:")
            print(f"  Low risk: {len(low_risk)} stocks")
            print(f"  Medium risk: {len(medium_risk)} stocks") 
            print(f"  High risk: {len(high_risk)} stocks")
            
            if medium_risk:
                positions = ai_endpoints._calculate_position_sizes(medium_risk[:5], 100000, 5)
                total_allocation = sum(p['investment_amount'] for p in positions)
                print(f"Position sizing: Rs.{total_allocation:,.0f} allocated")
                
                for pos in positions[:3]:
                    expected = pos['quantity'] * pos['price']
                    actual = pos['investment_amount']
                    status = "OK" if abs(expected - actual) < 1 else "ERROR"
                    print(f"{pos['symbol']}: {status}")
        
        print("AI test completed successfully!")
        
    except Exception as e:
        print(f"AI test failed: {e}")

if __name__ == "__main__":
    test_ai_simple()