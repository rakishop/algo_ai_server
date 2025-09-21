import json

# Load and examine the first few records
with open("live_api_responses.json", 'r') as f:
    data = json.load(f)

# Check the structure of 52-week-extremes data
extremes_data = data['responses']['/api/v1/market/52-week-extremes']

print("52-week high data structure:")
if '52_week_high' in extremes_data and 'data' in extremes_data['52_week_high']:
    sample_stock = extremes_data['52_week_high']['data'][0]
    print("Sample stock fields:")
    for key, value in sample_stock.items():
        print(f"  {key}: {value}")

print("\n" + "="*50)

# Check daily movers structure
movers_data = data['responses']['/api/v1/market/daily-movers']
print("Daily movers structure:")
if 'gainers' in movers_data:
    print("Gainers keys:", list(movers_data['gainers'].keys()))
    if 'NIFTY' in movers_data['gainers'] and 'data' in movers_data['gainers']['NIFTY']:
        sample_gainer = movers_data['gainers']['NIFTY']['data'][0]
        print("Sample gainer fields:")
        for key, value in sample_gainer.items():
            print(f"  {key}: {value}")