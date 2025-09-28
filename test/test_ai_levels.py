import sys
sys.path.append('..')
from utils.ai_risk_calculator import AIRiskCalculator

# Test AI levels calculation
calculator = AIRiskCalculator()

print("Testing AI levels for BHARATGEAR...")
levels = calculator.calculate_ai_levels("BHARATGEAR", 141.62, "STRONG BUY")
print(f"Current: ₹141.62")
print(f"Stop Loss: ₹{levels['stop_loss']} ({levels['sl_percentage']}%)")
print(f"Target: ₹{levels['target']} ({levels['target_percentage']}%)")
print(f"R:R Ratio: {levels['risk_reward_ratio']}")
print(f"ATR: {levels['atr']}, Volatility: {levels['volatility']}%")
print(f"Support: ₹{levels['support']}, Resistance: ₹{levels['resistance']}")

print("\n" + "="*50)

print("Testing AI levels for TATAMOTORS...")
levels = calculator.calculate_ai_levels("TATAMOTORS", 673.95, "BUY")
print(f"Current: ₹673.95")
print(f"Stop Loss: ₹{levels['stop_loss']} ({levels['sl_percentage']}%)")
print(f"Target: ₹{levels['target']} ({levels['target_percentage']}%)")
print(f"R:R Ratio: {levels['risk_reward_ratio']}")
print(f"ATR: {levels['atr']}, Volatility: {levels['volatility']}%")
print(f"Support: ₹{levels['support']}, Resistance: ₹{levels['resistance']}")

print("\n" + "="*50)

print("Testing stock historical data...")
hist_data = calculator.get_stock_historical_data("BHARATGEAR")
if hist_data:
    print(f"Historical data found: {len(hist_data['close'])} days")
    print(f"Recent closes: {hist_data['close'][:3]}")
    print(f"52W High: {hist_data['52w_high']}, 52W Low: {hist_data['52w_low']}")
else:
    print("No historical data found - using fallback")