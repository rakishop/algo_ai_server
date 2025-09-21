import os
import asyncio
from train_models import train_from_json_files, setup_training_data

async def startup_tasks():
    """Run startup tasks for deployment"""
    print("🚀 Running startup tasks...")
    
    # Train ML models on startup
    try:
        if not train_from_json_files():
            print("📡 No existing training data, collecting fresh data...")
            setup_training_data()
    except Exception as e:
        print(f"⚠️ Training warning: {e}")
    
    print("✅ Startup tasks completed")

if __name__ == "__main__":
    asyncio.run(startup_tasks())