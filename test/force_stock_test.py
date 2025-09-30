#!/usr/bin/env python3
"""
Force Stock test - bypass all checks
"""
from analyzers.intelligent_stock_analyzer import IntelligentStockAnalyzer
import requests

def force_stock_test():
    print("Force Stock Test - Bypassing all checks...")
    
    analyzer = IntelligentStockAnalyzer()
    
    # Get data and opportunities
    data = analyzer.fetch_all_stock_data()
    opportunities = analyzer.extract_opportunities_from_data(data)
    
    print(f"Found {len(opportunities)} total stock opportunities")
    
    if opportunities:
        # Take top 3 opportunities regardless of previous analysis
        try:
            top_opportunities = sorted(opportunities, key=lambda x: x.ai_score, reverse=True)[:3]
            
            print("Top 3 stock opportunities:")
            for i, opp in enumerate(top_opportunities, 1):
                print(f"  {i}. {opp.symbol} ({opp.sector}) - AI Score: {opp.ai_score:.1f}")
        except Exception as e:
            print(f"Error processing stock opportunities: {e}")
            print(f"First opportunity type: {type(opportunities[0])}")
            return
        
        # Force send to Telegram
        print("\nForcing Stock Telegram send...")
        success = analyzer.send_telegram_notification(top_opportunities)
        
        if success:
            print("SUCCESS: Stock Telegram notification sent!")
            
            # Also send WebSocket broadcast via HTTP API
            try:
                ws_data = {
                    "type": "stock_alert",
                    "alert_count": len(top_opportunities),
                    "opportunities": [
                        {
                            "symbol": opp.symbol,
                            "current_price": opp.current_price,
                            "price_change": opp.price_change,
                            "volume": opp.volume,
                            "recommendation": opp.recommendation,
                            "ai_score": opp.ai_score,
                            "confidence": opp.confidence,
                            "target": opp.target,
                            "stop_loss": opp.stop_loss,
                            "sector": opp.sector
                        }
                        for opp in top_opportunities
                    ],
                    "message": "Stock opportunities from force test"
                }
                
                response = requests.post("http://localhost:8000/api/v1/websocket/broadcast", json=ws_data, timeout=5)
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Stock WebSocket broadcast sent to {result.get('active_connections', 0)} connections")
                else:
                    print(f"❌ Stock WebSocket broadcast failed: {response.status_code}")
            except Exception as e:
                print(f"❌ Stock WebSocket broadcast error: {e}")
        else:
            print("FAILED: Stock Telegram notification failed")
            
            # Check Telegram config
            from config import settings
            print(f"Bot token configured: {'Yes' if settings.telegram_bot_token else 'No'}")
            print(f"Chat ID: {settings.telegram_chat_id}")
    else:
        print("No stock opportunities found to send")

if __name__ == "__main__":
    force_stock_test()