#!/usr/bin/env python3
import websocket
import json
import threading
import time

def on_message(ws, message):
    try:
        data = json.loads(message)
        print(f"\n=== WebSocket Message Received ===")
        print(f"Type: {data.get('type', 'unknown')}")
        print(f"Timestamp: {data.get('timestamp', 'N/A')}")
        if data.get('type') == 'derivative_alert':
            print(f"Market Sentiment: {data.get('market_sentiment', 'N/A')}")
            print(f"Strategy: {data.get('recommended_strategy', 'N/A')}")
            print(f"Opportunities: {len(data.get('opportunities', []))}")
        print("=" * 35)
    except Exception as e:
        print(f"Error parsing message: {e}")
        print(f"Raw message: {message}")

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    print(f"WebSocket closed: {close_status_code} - {close_msg}")

def on_open(ws):
    print("WebSocket connected successfully!")
    print("Waiting for messages... (Press Ctrl+C to quit)")

if __name__ == "__main__":
    print("Connecting to WebSocket...")
    ws = websocket.WebSocketApp("ws://localhost:8000/ws",
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close,
                              on_open=on_open)
    
    try:
        ws.run_forever()
    except KeyboardInterrupt:
        print("\nDisconnecting...")
        ws.close()