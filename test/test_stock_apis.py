import sys
sys.path.append('..')
from utils.ai_risk_calculator import AIRiskCalculator

# Test stock APIs
calculator = AIRiskCalculator()

# Test HDFCBANK stock chart data
print("Testing HDFCBANK chart data...")
chart_data = calculator.get_stock_chart_data("HDFCBANK")
if chart_data:
    print(f"SUCCESS: Chart data - Close: {chart_data['close_price']}, Points: {len(chart_data['prices'])}")
else:
    print("FAILED: Chart data")

# Test HDFCBANK historical data
print("\nTesting HDFCBANK historical data...")
hist_data = calculator.get_stock_historical_data("HDFCBANK")
if hist_data:
    print(f"SUCCESS: Historical data - Days: {len(hist_data['close'])}, 52W High: {hist_data['52w_high']}")
else:
    print("FAILED: Historical data")

# Test AI levels calculation
print("\nTesting AI levels for HDFCBANK...")
ai_levels = calculator.calculate_ai_levels("HDFCBANK", 945.05, "BUY")
print(f"AI Levels: SL={ai_levels['stop_loss']}, Target={ai_levels['target']}, R:R={ai_levels['risk_reward_ratio']}")