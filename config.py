import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    server_url: str = None
    
    class Config:
        env_file = ".env"
    
    @property
    def base_url(self) -> str:
        if self.server_url:
            return self.server_url
        return f"http://localhost:{self.port}"

settings = Settings()