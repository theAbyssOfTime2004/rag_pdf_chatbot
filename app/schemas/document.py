from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class DocumentBase(BaseModel):
    filename: str
    original_filename: str
    content_type: str

class DocumentCreate(DocumentBase):
    pass

class DocumentResponse(DocumentBase):
    id: int
    file_size: int
    file_path: str
    processed: bool
    text_content: Optional[str]
    chunk_count: int
    processing_status: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True