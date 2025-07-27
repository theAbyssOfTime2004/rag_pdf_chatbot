from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# Tạo engine cho SQLite
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False}  # Cần cho SQLite
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class cho tất cả models
Base = declarative_base()

# Dependency để get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()