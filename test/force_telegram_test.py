#!/usr/bin/env python3
"""
Force Telegram test - bypass all checks
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzers.intelligent_derivative_analyzer import IntelligentDerivativeAnalyzer
import requests
import websocket
import threading
import time
import json

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
        
        # Connect WebSocket client first
        print("\nConnecting WebSocket client...")
        received_messages = []
        
        def on_message(ws, message):
            data = json.loads(message)
            received_messages.append(data)
            print(f"WebSocket received: {data.get('type', 'unknown')}")
        
        def on_open(ws):
            print("WebSocket connected successfully")
        
        ws = websocket.WebSocketApp("ws://localhost:8000/ws", on_message=on_message, on_open=on_open)
        ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
        ws_thread.start()
        time.sleep(2)  # Wait for connection
        
        # Force send to Telegram
        print("\nForcing Telegram send...")
        
        # Import WebSocket manager for testing
        try:
            from ws.websocket_streaming import manager

            print(f"WebSocket manager imported with {len(manager.active_connections)} connections")
            
            # Try to get the actual server manager instance
            try:
                import sys
                if 'main' in sys.modules:
                    from main import manager as server_manager
                    print(f"Server manager found with {len(server_manager.active_connections)} connections")
                    manager = server_manager  # Use the server's manager
                else:
                    print("Server not running or main module not loaded")
            except Exception as e:
                print(f"Could not access server manager: {e}")
            success = analyzer.send_telegram_notification(top_opportunities, websocket_manager=manager)
            
            # If telegram was successful, broadcast the same data via server
            if success:
                analysis_data = analyzer._prepare_analysis_data(top_opportunities)
                ws_payload = {
                    "type": "derivative_alert",
                    "timestamp": time.time(),
                    "alert_count": len(top_opportunities),
                    "market_sentiment": analysis_data["market_sentiment"],
                    "recommended_strategy": analysis_data["best_strategy"],
                    "opportunities": [{
                        "symbol": opp.symbol,
                        "option_type": opp.option_type,
                        "strike_price": opp.strike_price,
                        "current_price": opp.current_price,
                        "recommendation": opp.recommendation,
                        "ai_score": opp.ai_score,
                        "confidence": opp.confidence,
                        "target": opp.target,
                        "stop_loss": opp.stop_loss,
                        "reasons": opp.reasons[:2]
                    } for opp in top_opportunities]
                }
                try:
                    ws_response = requests.post("http://localhost:8000/test-broadcast", json=ws_payload)
                    print(f"Full derivative alert broadcast: {ws_response.status_code}")
                except Exception as e:
                    print(f"Full derivative alert broadcast failed: {e}")
        except Exception as e:
            print(f"WebSocket manager import failed: {e}")
            success = analyzer.send_telegram_notification(top_opportunities)
        
        # Broadcast actual telegram data via server
        try:
            telegram_payload = {
                "type": "derivative_alert",
                "message": f"BULL CALL SPREAD: BUY 24700 CALL + SELL 24800 CALL | Market: STRONG BULLISH",
                "opportunities": [{
                    "symbol": opp.symbol,
                    "option_type": opp.option_type,
                    "ai_score": opp.ai_score,
                    "recommendation": opp.recommendation
                } for opp in top_opportunities],
                "timestamp": time.time()
            }
            broadcast_response = requests.post("http://localhost:8000/test-broadcast", json=telegram_payload)
            print(f"Telegram data broadcast: {broadcast_response.status_code}")
        except Exception as e:
            print(f"Telegram data broadcast failed: {e}")
        
        # Trigger server analysis
        try:
            response = requests.get("http://localhost:8000/api/v1/ai/intelligent-derivative-analysis")
            if response.status_code == 200:
                result = response.json()
                print(f"Server analysis triggered: {result.get('status', 'unknown')}")
        except Exception as e:
            print(f"Server endpoint test failed: {e}")
        
        if success:
            print("SUCCESS: Telegram notification sent!")
            print("WebSocket broadcast handled by analyzer.send_telegram_notification()")
        else:
            print("FAILED: Telegram notification failed")
            from config import settings
            print(f"Bot token configured: {'Yes' if settings.telegram_bot_token else 'No'}")
            print(f"Chat ID: {settings.telegram_chat_id}")
        
        # Check WebSocket messages
        time.sleep(3)  # Wait for any pending messages
        print(f"\nWebSocket messages received: {len(received_messages)}")
        for i, msg in enumerate(received_messages, 1):
            print(f"  {i}. {msg.get('type', 'unknown')}: {msg.get('message', 'no message')[:50]}...")
        
        ws.close()
    else:
        print("No opportunities found to send")

if __name__ == "__main__":
    force_telegram_test()