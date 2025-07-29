from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models.chat import ChatSession, ChatMessage
from app.schemas.chat import (
    ChatSessionResponse, 
    ChatMessageResponse, 
    ChatMessageCreate,
    ChatSessionCreate
)
import uuid

router = APIRouter()

@router.post("/sessions", response_model=ChatSessionResponse)
async def create_chat_session(
    session_data: ChatSessionCreate,
    db: Session = Depends(get_db)
):
    """Tạo phiên chat mới"""
    session = ChatSession(
        session_id=str(uuid.uuid4()),
        title=session_data.title or "New Chat"
    )
    
    db.add(session)
    db.commit()
    db.refresh(session)
    
    return session

@router.get("/sessions", response_model=List[ChatSessionResponse])
async def get_chat_sessions(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Lấy danh sách phiên chat"""
    sessions = db.query(ChatSession).offset(skip).limit(limit).all()
    return sessions

@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_chat_session(session_id: str, db: Session = Depends(get_db)):
    """Lấy thông tin phiên chat"""
    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return session

@router.post("/sessions/{session_id}/messages", response_model=ChatMessageResponse)
async def send_message(
    session_id: str,
    message_data: ChatMessageCreate,
    db: Session = Depends(get_db)
):
    """Gửi tin nhắn trong phiên chat"""
    # Kiểm tra session tồn tại
    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    # Tạo message
    message = ChatMessage(
        session_id=session_id,
        message_type=message_data.message_type,
        content=message_data.content,
        message_metadata=message_data.message_metadata
    )
    
    db.add(message)
    db.commit()
    db.refresh(message)
    
    return message

@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessageResponse])
async def get_chat_messages(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Lấy tin nhắn trong phiên chat"""
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.created_at).all()
    
    return messages