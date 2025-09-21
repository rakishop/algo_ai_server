import uvicorn
from config import settings

if __name__ == "__main__":
    # Use PORT from environment for Render deployment
    import os
    port = int(os.environ.get("PORT", settings.port))
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=port,
        reload=settings.debug
    )