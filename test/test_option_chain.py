import sys
sys.path.append('..')
from utils.ai_risk_calculator import AIRiskCalculator

# Test option chain API
calculator = AIRiskCalculator()

# Test BANKNIFTY option chain
print("Testing BANKNIFTY option chain...")
data = calculator.get_option_chain_data("BANKNIFTY")

if data:
    print("SUCCESS: Option chain data fetched and saved")
else:
    print("FAILED: Failed to fetch option chain data")

# Test NIFTY option chain
print("\nTesting NIFTY option chain...")
data = calculator.get_option_chain_data("NIFTY")

if data:
    print("SUCCESS: Option chain data fetched and saved")
else:
    print("FAILED: Failed to fetch option chain data")