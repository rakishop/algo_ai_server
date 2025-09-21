try:
    from enhanced_ai_endpoints import EnhancedAIEndpoints
    print("✓ EnhancedAIEndpoints imported successfully")
except Exception as e:
    print(f"✗ Import error: {e}")

try:
    from fastapi import FastAPI
    from nse_client import NSEClient
    
    app = FastAPI()
    nse_client = NSEClient()
    enhanced = EnhancedAIEndpoints(app, nse_client)
    print("✓ EnhancedAIEndpoints initialized successfully")
    
    # Check if routes were added
    routes = [route.path for route in app.routes]
    ai_routes = [r for r in routes if '/ai/' in r]
    print(f"✓ AI routes found: {len(ai_routes)}")
    for route in ai_routes:
        print(f"  {route}")
        
except Exception as e:
    print(f"✗ Initialization error: {e}")
    import traceback
    traceback.print_exc()