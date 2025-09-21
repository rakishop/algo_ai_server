import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any

class SpikeTracker:
    def __init__(self):
        self.data_file = "spike_history.json"
        self.history = self.load_history()
    
    def load_history(self) -> Dict:
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_history(self):
        with open(self.data_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def store_snapshot(self, symbol: str, options_data: List[Dict]):
        timestamp = datetime.now().isoformat()
        key = f"{symbol}_{timestamp[:16]}"  # Minute precision
        
        self.history[key] = {
            "timestamp": timestamp,
            "symbol": symbol,
            "options": options_data
        }
        
        # Keep only last 24 hours
        cutoff = datetime.now() - timedelta(hours=24)
        self.history = {k: v for k, v in self.history.items() 
                       if datetime.fromisoformat(v["timestamp"]) > cutoff}
        
        self.save_history()
    
    def detect_period_spikes(self, symbol: str, minutes: int = 5) -> List[Dict]:
        now = datetime.now()
        cutoff = now - timedelta(minutes=minutes)
        
        # Get snapshots within time period
        recent_snapshots = []
        for key, data in self.history.items():
            if (data["symbol"] == symbol and 
                datetime.fromisoformat(data["timestamp"]) > cutoff):
                recent_snapshots.append(data)
        
        if len(recent_snapshots) < 2:
            return []
        
        # Compare first and last snapshot
        first = recent_snapshots[0]["options"]
        last = recent_snapshots[-1]["options"]
        
        spikes = []
        for i, current_opt in enumerate(last):
            if i < len(first):
                prev_opt = first[i]
                
                # Calculate changes
                price_change = self._calculate_change(
                    prev_opt.get('lastPrice', 0), 
                    current_opt.get('lastPrice', 0)
                )
                volume_change = self._calculate_change(
                    prev_opt.get('totalTradedVolume', 0),
                    current_opt.get('totalTradedVolume', 0)
                )
                
                # Detect significant changes
                if abs(price_change) > 20 or volume_change > 100:
                    price_diff = current_opt.get('lastPrice', 0) - prev_opt.get('lastPrice', 0)
                    volume_diff = current_opt.get('totalTradedVolume', 0) - prev_opt.get('totalTradedVolume', 0)
                    oi_diff = current_opt.get('openInterest', 0) - prev_opt.get('openInterest', 0)
                    
                    # Calculate value change (price * volume)
                    current_value = current_opt.get('lastPrice', 0) * current_opt.get('totalTradedVolume', 0)
                    previous_value = prev_opt.get('lastPrice', 0) * prev_opt.get('totalTradedVolume', 0)
                    value_diff = current_value - previous_value
                    value_change_pct = self._calculate_change(previous_value, current_value)
                    
                    spikes.append({
                        "strike": current_opt.get('strikePrice', 0),
                        "option_type": current_opt.get('optionType', 'Unknown'),
                        "price_change_pct": price_change,
                        "price_change_absolute": price_diff,
                        "volume_change_pct": volume_change,
                        "volume_change_absolute": volume_diff,
                        "value_change_pct": value_change_pct,
                        "value_change_absolute": value_diff,
                        "current_price": current_opt.get('lastPrice', 0),
                        "previous_price": prev_opt.get('lastPrice', 0),
                        "current_volume": current_opt.get('totalTradedVolume', 0),
                        "previous_volume": prev_opt.get('totalTradedVolume', 0),
                        "current_value": current_value,
                        "previous_value": previous_value,
                        "bid_price": current_opt.get('bidprice', 0),
                        "ask_price": current_opt.get('askPrice', 0),
                        "open_interest": current_opt.get('openInterest', 0),
                        "oi_change": current_opt.get('changeinOpenInterest', 0),
                        "oi_change_absolute": oi_diff,
                        "implied_volatility": current_opt.get('impliedVolatility', 0),
                        "time_period": f"{minutes} minutes",
                        "change_summary": f"₹{price_diff:.2f} ({price_change:.1f}%) | Value: ₹{value_diff:,.0f} ({value_change_pct:.1f}%) in {minutes}min",
                        "spike_type": "Price" if abs(price_change) > 20 else "Volume"
                    })
        
        return sorted(spikes, key=lambda x: abs(x['price_change_pct']), reverse=True)
    
    def _calculate_change(self, old_val: float, new_val: float) -> float:
        if old_val == 0:
            return 0
        return ((new_val - old_val) / old_val) * 100