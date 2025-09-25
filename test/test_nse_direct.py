from nse_client import NSEClient

nse = NSEClient()

print("Testing NSE endpoints...")

# Test different endpoints
endpoints = [
    ("52 Week High", nse.get_52week_high_stocks_data),
    ("Most Active", nse.get_most_active_securities),
    ("Volume Gainers", nse.get_volume_gainers),
    ("All Indices", nse.get_all_indices)
]

for name, func in endpoints:
    try:
        result = func()
        if "error" in result:
            print(f"{name}: ERROR - {result['error']}")
        else:
            data_count = len(result.get('data', []))
            print(f"{name}: SUCCESS - {data_count} items")
            if data_count > 0:
                print(f"  Sample: {list(result['data'][0].keys())[:5]}")
    except Exception as e:
        print(f"{name}: EXCEPTION - {e}")