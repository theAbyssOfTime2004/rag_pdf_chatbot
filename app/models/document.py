from sqlalchemy import Column, String, Text, Integer, Boolean, DateTime, LargeBinary
from sqlalchemy.orm import relationship
from sqlalchemy import ForeignKey, UniqueConstraint
from .base import BaseModel


class Document(BaseModel):
    __tablename__ = "documents"
    
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_path = Column(String(500), nullable=False)
    content_type = Column(String(100), nullable=False)
    processed = Column(Boolean, default=False)
    text_content = Column(Text, nullable=True)
    chunk_count = Column(Integer, default=0)
    processing_status = Column(String(50), default="pending")  # pending, processing, completed, failed
    
    # Relationship
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(BaseModel):
    __tablename__ = "document_chunks"
    
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=True)
    embedding = Column(LargeBinary, nullable=True)  # Serialized vector embedding
    
    __table_args__ = (
        UniqueConstraint('document_id', 'chunk_index', name='uq_document_chunk_index'),
    )

    # Relationship
    document = relationship("Document", back_populates="chunks")