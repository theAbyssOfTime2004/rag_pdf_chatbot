from sqlalchemy import Column, String, Text, Integer, Float, JSON
from sqlalchemy.orm import relationship
from sqlalchemy import ForeignKey
from .base import BaseModel

class ChatSession(BaseModel):
    __tablename__ = "chat_sessions"
    
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    title = Column(String(500), nullable=True)
    
    # Relationship
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(BaseModel):
    __tablename__ = "chat_messages"
    
    session_id = Column(String(255), ForeignKey("chat_sessions.session_id"), nullable=False)
    message_type = Column(String(50), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    message_metadata = Column(JSON, nullable=True)  # Store context, sources, etc.
    feedback_score = Column(Integer, nullable=True)  # 1-5 rating
    
    # Relationship
    session = relationship("ChatSession", back_populates="messages")