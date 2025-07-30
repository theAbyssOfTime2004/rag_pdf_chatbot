from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime

class DocumentBase(BaseModel):
    filename: str = Field(..., description="Stored filename")
    original_filename: str = Field(..., description="Original uploaded filename")
    content_type: str = Field(..., description="MIME type")

class DocumentCreate(DocumentBase):
    """Schema for creating new document"""
    pass

class DocumentUpdate(BaseModel):
    """Schema for updating document"""
    original_filename: Optional[str] = None
    processing_status: Optional[str] = Field(None, description="pending, processing, completed, failed")
    processed: Optional[bool] = None
    
    model_config = ConfigDict(extra="forbid")

class DocumentResponse(DocumentBase):
    """Schema for document response"""
    id: int
    file_size: int = Field(..., description="File size in bytes")
    file_path: str = Field(..., description="Storage path")
    processed: bool = Field(..., description="Whether document is processed")
    text_content: Optional[str] = Field(None, description="Extracted text content")
    chunk_count: int = Field(0, description="Number of text chunks")
    processing_status: str = Field(..., description="Current processing status")
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class DocumentStatistics(BaseModel):
    """Schema for document statistics"""
    total_documents: int
    processed_documents: int
    pending_documents: int
    failed_documents: int
    total_size_bytes: int
    total_size_mb: float

class BulkDeleteRequest(BaseModel):
    """Schema for bulk delete request"""
    document_ids: List[int] = Field(..., min_length=1, max_length=100)

class BulkDeleteResponse(BaseModel):
    """Schema for bulk delete response"""
    deleted_count: int
    failed_count: int
    deleted_ids: List[int]
    failed_ids: List[int]

class DocumentChunkPreview(BaseModel):
    """Schema for document chunk preview"""
    id: int
    chunk_index: int
    page_number: Optional[int]
    text_preview: str