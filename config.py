import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    server_url: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    telegram_news_channel_id: Optional[str] = None
    twitter_bearer_token: Optional[str] = None
    twitter_api_key: Optional[str] = None
    twitter_api_secret: Optional[str] = None
    
    class Config:
        env_file = ".env"
        extra = "ignore"
    
    @property
    def base_url(self) -> str:
        return self.server_url or f"http://{self.host}:{self.port}"

settings = Settings()