from pydantic import BaseModel
from typing import Optional, Any, Dict
from datetime import datetime

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