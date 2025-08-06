from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Câu hỏi của người dùng")
    session_id: Optional[str] = Field(None, description="ID của phiên trò chuyện để duy trì ngữ cảnh")
    document_id: Optional[int] = Field(None, description="ID của tài liệu cụ thể để giới hạn tìm kiếm")

class Source(BaseModel):
    chunk_id: int
    document_id: int
    page_number: int
    similarity_score: float
    chunk_text: str

class ChatResponse(BaseModel):
    answer: str
    model_used: str
    sources: List[Source]  

class ChatHistory(BaseModel):
    question: str
    answer: str
    timestamp: datetime
    model_config = {
        "from_attributes": True
    }

class FeedbackRequest(BaseModel):
    message_id: int
    is_helpful: bool
    comment: Optional[str] = None

class ChatSessionCreate(BaseModel):
    title: Optional[str] = None

class ChatSessionResponse(BaseModel):
    id: int
    session_id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class ChatMessageCreate(BaseModel):
    message_type: str  # 'user' or 'assistant'
    content: str
    message_metadata: Optional[Dict[str, Any]] = None

class ChatMessageResponse(BaseModel):
    id: int
    session_id: str
    message_type: str
    content: str
    message_metadata: Optional[Dict[str, Any]]
    feedback_score: Optional[int]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True