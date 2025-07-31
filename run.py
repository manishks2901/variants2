import uvicorn
import os
from decouple import config

if __name__ == "__main__":
    port = int(config("PORT", default=8000))
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
