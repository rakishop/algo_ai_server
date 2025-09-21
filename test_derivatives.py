from nse_client import NSEClient
import json

client = NSEClient()

# Test derivatives methods
print("=== Testing Derivatives Snapshot ===")
derivatives_snapshot = client.get_derivatives_snapshot()
print(json.dumps(derivatives_snapshot, indent=2)[:1000])

print("\n=== Testing Most Active Underlying ===")
active_underlying = client.get_most_active_underlying()
print(json.dumps(active_underlying, indent=2)[:1000])

print("\n=== Testing OI Spurts Contracts ===")
oi_contracts = client.get_oi_spurts_contracts()
print(json.dumps(oi_contracts, indent=2)[:1000])