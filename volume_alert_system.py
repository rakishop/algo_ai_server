import requests
import time
import json
import os
from datetime import datetime
from config import settings
from nse_client import NSEClient

class VolumeAlertSystem:
    def __init__(self):
        self.nse = NSEClient()
        self.volume_history_file = "volume_history.json"
        self.alert_threshold = 50  # 50% volume increase
        
    def load_volume_history(self):
        """Load previous volume data"""
        try:
            if os.path.exists(self.volume_history_file):
                with open(self.volume_history_file, 'r') as f:
                    data = json.load(f)
                    # Check if data is from previous day
                    if 'timestamp' in data:
                        data_date = datetime.fromisoformat(data['timestamp']).date()
                        if data_date < datetime.now().date():
                            print("Clearing old volume data from previous day")
                            return {}
                    return data
        except:
            pass
        return {}
    
    def save_volume_history(self, volume_data):
        """Save current volume data with timestamp"""
        try:
            data_with_timestamp = {
                **volume_data,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.volume_history_file, 'w') as f:
                json.dump(data_with_timestamp, f)
        except Exception as e:
            print(f"Error saving volume history: {e}")
    
    def is_market_open(self):
        """Check if market is open"""
        now = datetime.now()
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_start <= now <= market_end
    
    def get_current_volumes(self):
        """Get current volume data from NSE"""
        try:
            volume_data = self.nse.get_volume_gainers()
            active_data = self.nse.get_most_active_securities()
            
            current_volumes = {}
            
            # Process volume gainers
            if volume_data.get('data'):
                for stock in volume_data['data'][:20]:  # Top 20
                    symbol = stock.get('symbol')
                    volume = stock.get('quantityTraded', 0)
                    price = stock.get('lastPrice', 0)
                    change = stock.get('pChange', 0)
                    
                    if symbol and volume > 0:
                        current_volumes[symbol] = {
                            'volume': volume,
                            'price': price,
                            'change': change,
                            'timestamp': datetime.now().isoformat()
                        }
            
            # Add most active securities
            if active_data.get('data'):
                for stock in active_data['data'][:15]:  # Top 15
                    symbol = stock.get('symbol')
                    volume = stock.get('quantityTraded', 0)
                    price = stock.get('lastPrice', 0)
                    change = stock.get('pChange', 0)
                    
                    if symbol and volume > 0:
                        current_volumes[symbol] = {
                            'volume': volume,
                            'price': price,
                            'change': change,
                            'timestamp': datetime.now().isoformat()
                        }
            
            return current_volumes
            
        except Exception as e:
            print(f"Error getting volume data: {e}")
            return {}
    
    def detect_volume_spikes(self, current_volumes, previous_volumes):
        """Detect stocks with significant volume increases using AI/ML"""
        # Basic comparison (existing logic)
        basic_spikes = []
        for symbol, current_data in current_volumes.items():
            if symbol in previous_volumes:
                prev_volume = previous_volumes[symbol]['volume']
                curr_volume = current_data['volume']
                
                if prev_volume > 0:
                    volume_increase = ((curr_volume - prev_volume) / prev_volume) * 100
                    
                    if volume_increase >= self.alert_threshold:
                        basic_spikes.append({
                            'symbol': symbol,
                            'current_volume': curr_volume,
                            'previous_volume': prev_volume,
                            'volume_increase': volume_increase,
                            'price': current_data['price'],
                            'price_change': current_data['change'],
                            'detection_method': 'Basic'
                        })
        
        # Enhanced AI detection
        try:
            from enhanced_volume_detector import enhanced_volume_detection
            enhanced_results = enhanced_volume_detection(current_volumes)
            
            ai_spikes = []
            # Add statistical anomalies
            for anomaly in enhanced_results.get('statistical_anomalies', []):
                ai_spikes.append({
                    'symbol': anomaly['symbol'],
                    'current_volume': anomaly['current_volume'],
                    'previous_volume': anomaly['avg_volume'],
                    'volume_increase': (anomaly['volume_ratio'] - 1) * 100,
                    'price': anomaly['price'],
                    'price_change': anomaly['price_change'],
                    'detection_method': 'AI-Statistical',
                    'ai_score': anomaly['anomaly_strength'],
                    'z_score': anomaly['z_score']
                })
            
            # Combine and deduplicate
            all_spikes = basic_spikes + ai_spikes
            seen_symbols = set()
            unique_spikes = []
            
            for spike in sorted(all_spikes, key=lambda x: x.get('ai_score', x['volume_increase']), reverse=True):
                if spike['symbol'] not in seen_symbols:
                    seen_symbols.add(spike['symbol'])
                    unique_spikes.append(spike)
            
            return unique_spikes
            
        except Exception as e:
            print(f"AI detection failed, using basic: {e}")
            return sorted(basic_spikes, key=lambda x: x['volume_increase'], reverse=True)
    
    def send_telegram_alert(self, volume_spikes):
        """Send volume spike alert to Telegram"""
        if not volume_spikes:
            return False
        
        try:
            # Get IST time
            utc_now = datetime.utcnow()
            ist_time = utc_now + timedelta(hours=5, minutes=30)
            message = f"VOLUME SPIKE ALERT - {ist_time.strftime('%H:%M')}\n\n"
            message += f"HIGH VOLUME GAINERS ({len(volume_spikes)})\n\n"
            
            for i, spike in enumerate(volume_spikes[:5], 1):  # Top 5
                symbol = spike['symbol']
                vol_inc = spike['volume_increase']
                price = spike['price']
                price_change = spike['price_change']
                curr_vol = spike['current_volume']
                
                detection_method = spike.get('detection_method', 'Basic')
                ai_score = spike.get('ai_score', vol_inc)
                method_type = "AI" if "AI" in detection_method else "Basic"
                
                message += f"{i}. {symbol} ({method_type})\n"
                message += f"   Price: Rs{price:.1f} ({price_change:+.1f}%)\n"
                message += f"   Volume: +{vol_inc:.1f}%\n"
                message += f"   AI Score: {ai_score:.0f}\n"
                message += f"   Qty: {curr_vol:,}\n\n"
            
            message += "Volume spikes detected every 3min"
            
            # Send to Telegram
            chat_id = settings.telegram_chat_id or "-1002981590794"
            
            if not settings.telegram_bot_token:
                print("Telegram bot token not configured")
                return False
            
            url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
            data = {"chat_id": chat_id, "text": message}
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200 and response.json().get('ok'):
                print(f"Volume alert sent at {datetime.now().strftime('%H:%M:%S')}")
                return True
            else:
                print(f"Failed to send volume alert: {response.text}")
                return False
                
        except Exception as e:
            print(f"Error sending Telegram alert: {e}")
            return False
    
    def clear_cache_after_market(self):
        """Clear cached data after market closes"""
        try:
            if os.path.exists(self.volume_history_file):
                os.remove(self.volume_history_file)
                print(f"Cleared volume cache after market close")
        except Exception as e:
            print(f"Error clearing cache: {e}")
    
    def run_volume_check(self):
        """Run single volume check"""
        try:
            if not self.is_market_open():
                # Clear cache when market is closed
                self.clear_cache_after_market()
                print(f"Market closed at {datetime.now().strftime('%H:%M:%S')} - skipping volume check")
                return False
            
            print(f"Market is open - proceeding with volume check at {datetime.now().strftime('%H:%M:%S')}")
            
            print(f"Checking volume spikes at {datetime.now().strftime('%H:%M:%S')}")
            
            # Get current and previous volumes
            current_volumes = self.get_current_volumes()
            previous_volumes = self.load_volume_history()
            
            if not current_volumes:
                print("No volume data available")
                return False
            
            # Detect volume spikes
            volume_spikes = self.detect_volume_spikes(current_volumes, previous_volumes)
            
            if volume_spikes:
                # Double-check market is still open before sending alert
                if not self.is_market_open():
                    print(f"Market closed during processing - not sending alert")
                    return False
                    
                print(f"Volume spikes detected: {len(volume_spikes)} stocks")
                success = self.send_telegram_alert(volume_spikes)
                if success:
                    # Save current volumes for next comparison
                    self.save_volume_history(current_volumes)
                return success
            else:
                print("No significant volume spikes detected")
                # Still save current volumes for next comparison
                self.save_volume_history(current_volumes)
                return False
                
        except Exception as e:
            print(f"Error in volume check: {e}")
            return False
    
    def start_monitoring(self):
        """Start continuous volume monitoring every 3 minutes"""
        print("Starting volume spike monitoring every 3 minutes...")
        print("Press Ctrl+C to stop")
        
        while True:
            try:
                self.run_volume_check()
                print("Next volume check in 3 minutes...")
                time.sleep(180)  # 3 minutes = 180 seconds
            except KeyboardInterrupt:
                print("\\nStopping volume monitoring...")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

# Function to be called by scheduler
def send_volume_alert():
    """Function to be called by external scheduler"""
    alert_system = VolumeAlertSystem()
    return alert_system.run_volume_check()

if __name__ == "__main__":
    alert_system = VolumeAlertSystem()
    alert_system.start_monitoring()