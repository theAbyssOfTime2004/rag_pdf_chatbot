from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Project information
    PROJECT_NAME: str = "PDF RAG Chatbot"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # API configuration
    API_V1_STR: str = "/api/v1"
    
    # Database configuration
    DATABASE_URL: str = "sqlite:///./rag_chatbot.db"
    
    # File Upload Settings
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB in bytes
    UPLOAD_FOLDER: str = "uploads/documents"
    ALLOWED_EXTENSIONS: set = {"pdf"}
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    
    # Embedding Settings - FIXED
    EMBEDDING_MODEL: str = "nomic-ai/nomic-embed-text-v1"  # Correct Nomic model
    # Alternative models:
    # EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"           # Fast, 384D
    # EMBEDDING_MODEL: str = "all-mpnet-base-v2"          # High quality, 768D
    
    # PDF Processing Settings
    OCR_LANGUAGE: str = "en"
    OCR_CONFIDENCE_THRESHOLD: float = 0.7
    USE_GPU_OCR: bool = False
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_PDF_PAGES: int = 100
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Vector Store Settings
    VECTOR_DIMENSION: int = 768  # Nomic model dimension
    FAISS_INDEX_TYPE: str = "FlatIP"
    MAX_SEARCH_RESULTS: int = 20
    SIMILARITY_THRESHOLD: float = 0.3
    
    # Embedding Cache
    ENABLE_EMBEDDING_CACHE: bool = True
    CACHE_SIZE_LIMIT: int = 10000
    
    # LLM Settings
    LLM_MODEL: str = "llama2"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"

# Global settings instance
settings = Settings()

# Tạo upload folder nếu chưa có
os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)