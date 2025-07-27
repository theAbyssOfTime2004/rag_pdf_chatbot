from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "RAG PDF Chatbot"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str = "sqlite:///./rag_chatbot.db"
    
    # File Upload Settings
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB in bytes
    UPLOAD_FOLDER: str = "uploads"
    ALLOWED_EXTENSIONS: set = {"pdf"}
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    
    # LLM Settings
    OPENAI_API_KEY: Optional[str] = None
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    LLM_MODEL: str = "gpt-3.5-turbo"
    
    # Chunking Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

# Tạo upload folder nếu chưa có
os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)