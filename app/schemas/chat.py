from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000, description="User's question")
    session_id: Optional[int] = Field(None, description="Chat session ID")
    max_tokens: Optional[int] = Field(512, ge=50, le=2048, description="Maximum tokens in response")
    include_sources: bool = Field(True, description="Include source documents in response")

class SourceInfo(BaseModel):
    content: str
    score: float
    chunk_id: Optional[int] = None
    document_id: Optional[int] = None
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    file_name: str = "Unknown"
    document_filename: Optional[str] = None

class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceInfo] = []
    total_sources: int
    session_id: Optional[int] = None
    message_id: Optional[int] = None
    model_used: str = "unknown"
    processing_info: Dict[str, Any] = {}

    class Config:
        from_attributes = True

class ChatSessionCreate(BaseModel):
    title: Optional[str] = Field(None, max_length=200, description="Session title")

class ChatSessionResponse(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int

    class Config:
        from_attributes = True

class ChatMessageResponse(BaseModel):
    id: int
    session_id: int
    role: str
    content: str
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = Field(None, alias="metadata_")

    class Config:
        from_attributes = True
        populate_by_name = True