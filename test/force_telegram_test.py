#!/usr/bin/env python3
"""
Force Telegram test - bypass all checks
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzers.intelligent_derivative_analyzer import IntelligentDerivativeAnalyzer
import requests

def force_telegram_test():
    print("Force Telegram Test - Bypassing all checks...")
    
    analyzer = IntelligentDerivativeAnalyzer()
    
    # Get data and opportunities
    data = analyzer.fetch_all_derivative_data()
    opportunities = analyzer.extract_opportunities_from_data(data)
    
    print(f"Found {len(opportunities)} total opportunities")
    
    if opportunities:
        # Take top 3 opportunities regardless of previous analysis
        try:
            top_opportunities = sorted(opportunities, key=lambda x: x.ai_score, reverse=True)[:3]
            
            print("Top 3 opportunities:")
            for i, opp in enumerate(top_opportunities, 1):
                print(f"  {i}. {opp.symbol} {opp.option_type} - AI Score: {opp.ai_score:.1f}")
        except Exception as e:
            print(f"Error processing opportunities: {e}")
            print(f"First opportunity type: {type(opportunities[0])}")
            return
        
        # Force send to Telegram
        print("\nForcing Telegram send...")
        
        # Import WebSocket manager for testing
        try:
            from ws.websocket_streaming import manager
            print(f"WebSocket manager imported with {len(manager.active_connections)} connections")
            success = analyzer.send_telegram_notification(top_opportunities, websocket_manager=manager)
        except Exception as e:
            print(f"WebSocket manager import failed: {e}")
            success = analyzer.send_telegram_notification(top_opportunities)
        
        # Test WebSocket broadcast directly
        try:
            import asyncio
            test_data = {"type": "test_from_analyzer", "message": "Direct test broadcast"}
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            ws_success = loop.run_until_complete(manager.broadcast_json(test_data))
            loop.close()
            print(f"Direct WebSocket test: {'SUCCESS' if ws_success else 'NO_CONNECTIONS'}")
        except Exception as e:
            print(f"Direct WebSocket test failed: {e}")
        
        if success:
            print("SUCCESS: Telegram notification sent!")
            
            print("WebSocket broadcast handled by analyzer.send_telegram_notification()")
        else:
            print("FAILED: Telegram notification failed")
            from config import settings
            print(f"Bot token configured: {'Yes' if settings.telegram_bot_token else 'No'}")
            print(f"Chat ID: {settings.telegram_chat_id}")
    else:
        print("No opportunities found to send")

if __name__ == "__main__":
    force_telegram_test()