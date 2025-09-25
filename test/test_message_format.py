#!/usr/bin/env python3
"""
Test message format for volume alerts
"""

from datetime import datetime

def test_message_format():
    # Sample data
    volume_spikes = [
        {
            'symbol': 'BAJAJELEC',
            'volume_increase': 9666.9,
            'price': 635.4,
            'price_change': 10.1,
            'current_volume': 4223222,
            'detection_method': 'Basic',
            'ai_score': 9667
        },
        {
            'symbol': 'MINDACORP',
            'volume_increase': 1319.7,
            'price': 580.5,
            'price_change': 8.6,
            'current_volume': 7628570,
            'detection_method': 'AI-Statistical',
            'ai_score': 1320
        }
    ]
    
    # Format message
    message = f"🚨 VOLUME SPIKE ALERT - {datetime.now().strftime('%H:%M')}\n\n"
    message += f"📈 HIGH VOLUME GAINERS ({len(volume_spikes)})\n\n"
    
    for i, spike in enumerate(volume_spikes, 1):
        symbol = spike['symbol']
        vol_inc = spike['volume_increase']
        price = spike['price']
        price_change = spike['price_change']
        curr_vol = spike['current_volume']
        
        detection_method = spike.get('detection_method', 'Basic')
        ai_score = spike.get('ai_score', vol_inc)
        method_emoji = "🤖" if "AI" in detection_method else "📊"
        
        message += f"{i}. {symbol} {method_emoji}\n"
        message += f"   💰 ₹{price:.1f} ({price_change:+.1f}%)\n"
        message += f"   📊 Volume: +{vol_inc:.1f}%\n"
        message += f"   🎯 AI Score: {ai_score:.0f}\n"
        message += f"   📈 Qty: {curr_vol:,}\n\n"
    
    message += "⚡ Volume spikes detected every 3min"
    
    print("MESSAGE FORMAT TEST:")
    print("=" * 50)
    print(message)
    print("=" * 50)
    
    return message

if __name__ == "__main__":
    test_message_format()