import sys
import importlib

# Force reload the ai_endpoints module
if 'ai_endpoints' in sys.modules:
    importlib.reload(sys.modules['ai_endpoints'])
    print("ai_endpoints module reloaded")
else:
    print("ai_endpoints module not loaded")

# Test syntax
try:
    import ai_endpoints
    print("ai_endpoints imported successfully")
except Exception as e:
    print(f"Import error: {e}")

print("Stop the server and restart it to load new endpoints")