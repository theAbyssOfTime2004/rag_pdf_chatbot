from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import uuid
import json

from app.database import get_db
from app.models.chat import ChatSession, ChatMessage
from app.schemas.chat import (
    ChatSessionResponse, 
    ChatMessageResponse, 
    ChatMessageCreate,
    ChatSessionCreate,
    ChatRequest,
    ChatResponse,
    Source,
    FeedbackRequest
)
from app.services.vector_service import VectorService
from app.services.llm_service import LLMService
from app.core.config import settings
import logging
logger = logging.getLogger(__name__)

# --- Khởi tạo các service cốt lõi ---
# Trong ứng dụng thực tế, bạn có thể muốn sử dụng hệ thống Dependency Injection phức tạp hơn
vector_service = VectorService()
llm_service = LLMService()

router = APIRouter()

# --- Endpoint RAG Chính ---
@router.post("/ask", response_model=ChatResponse, summary="Hỏi đáp với RAG Pipeline")
async def ask_question(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Endpoint chính để thực hiện chu trình Retrieval-Augmented Generation (RAG).
    1.  **Retrieval**: Tìm kiếm các đoạn văn bản liên quan trong vector store.
    2.  **Augmentation**: Tạo context từ các kết quả tìm kiếm.
    3.  **Generation**: Gửi context và câu hỏi đến LLM để tạo câu trả lời.
    4.  **Persistence**: Lưu câu hỏi và câu trả lời vào lịch sử chat.
    """
    # --- 1. Retrieval ---
    try:
        search_results = await vector_service.search_similar_chunks(
            query=request.question,
            k=settings.MAX_SEARCH_RESULTS,
            document_ids=[request.document_id] if request.document_id else None
        )
        
        if not search_results:
            raise HTTPException(
                status_code=404, 
                detail="Không tìm thấy nội dung liên quan trong tài liệu để trả lời câu hỏi này."
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi trong quá trình tìm kiếm vector: {e}")

    # --- 2. Augmentation ---
    context = "\n\n---\n\n".join([result['chunk_text'] for result in search_results])
    
    # --- 3. Generation ---
    try:
        llm_result = await llm_service.generate_response(
            question=request.question,
            context=context
        )
        if not llm_result or not llm_result.get("success"):
            raise HTTPException(status_code=503, detail="LLM service không khả dụng hoặc gặp lỗi.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi trong quá trình LLM tạo câu trả lời: {e}")

    # --- 4. Persistence (Lưu vào Database) ---
    try:
        # Tìm hoặc tạo session
        session = db.query(ChatSession).filter(ChatSession.session_id == request.session_id).first()
        if not session:
            session = ChatSession(session_id=request.session_id or str(uuid.uuid4()), title=request.question[:50])
            db.add(session)
            db.flush() # flush để lấy session.id nếu là session mới

        # Lưu câu hỏi của người dùng
        user_message = ChatMessage(
            session_id=session.session_id,
            message_type='user',
            content=request.question
        )
        db.add(user_message)

        # Lưu câu trả lời của AI, kèm theo nguồn
        sources_for_db = [Source(**result).model_dump() for result in search_results]
        ai_message = ChatMessage(
            session_id=session.session_id,
            message_type='ai',
            content=llm_result["answer"],
            message_metadata={"sources": sources_for_db, "model_used": llm_result["model_used"]}
        )
        db.add(ai_message)
        
        db.commit()
    except Exception as e:
        db.rollback()
        # Không raise lỗi ở đây để người dùng vẫn nhận được câu trả lời, nhưng ghi log lại
        logger.error(f"Could not save chat history: {e}")


    return ChatResponse(
        answer=llm_result["answer"],
        model_used=llm_result["model_used"],
        sources = [Source(**r) for r in search_results]
    )


# --- Các Endpoint Quản Lý Session và Message Hiện Có ---

@router.post("/sessions", response_model=ChatSessionResponse, summary="Tạo phiên chat mới")
async def create_chat_session(
    session_data: ChatSessionCreate,
    db: Session = Depends(get_db)
):
    """Tạo một phiên chat mới và trả về thông tin của nó."""
    session = ChatSession(
        session_id=str(uuid.uuid4()),
        title=session_data.title or "New Chat"
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

@router.get("/sessions", response_model=List[ChatSessionResponse], summary="Lấy danh sách các phiên chat")
async def get_chat_sessions(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Lấy danh sách tất cả các phiên chat đã có."""
    sessions = db.query(ChatSession).order_by(ChatSession.created_at.desc()).offset(skip).limit(limit).all()
    return sessions

@router.get("/sessions/{session_id}", response_model=ChatSessionResponse, summary="Lấy thông tin một phiên chat")
async def get_chat_session(session_id: str, db: Session = Depends(get_db)):
    """Lấy thông tin chi tiết của một phiên chat cụ thể."""
    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return session

@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessageResponse], summary="Lấy lịch sử tin nhắn của một phiên chat")
async def get_chat_messages(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Lấy tất cả các tin nhắn trong một phiên chat, sắp xếp theo thời gian."""
    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
        
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.created_at).all()
    
    return messages

# --- Endpoint Feedback ---
@router.post("/feedback", status_code=204, summary="Gửi phản hồi về một tin nhắn")
async def submit_feedback(
    feedback: FeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    Cho phép người dùng gửi phản hồi (hữu ích/không hữu ích) cho một tin nhắn cụ thể của AI.
    """
    message = db.query(ChatMessage).filter(ChatMessage.id == feedback.message_id).first()
    
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
        
    if message.message_type != 'ai':
        raise HTTPException(status_code=400, detail="Feedback can only be submitted for AI messages")
        
    # Cập nhật điểm feedback: 1 cho hữu ích, -1 cho không hữu ích
    message.feedback_score = 1 if feedback.is_helpful else -1
    
    # Xử lý message_metadata một cách an toàn
    if feedback.comment:
        # Chuyển đổi message_metadata thành dict nếu là JSON string hoặc None
        metadata_dict = {}
        
        if message.message_metadata:
            if isinstance(message.message_metadata, dict):
                metadata_dict = message.message_metadata
            elif isinstance(message.message_metadata, str):
                try:
                    metadata_dict = json.loads(message.message_metadata)
                except json.JSONDecodeError:
                    metadata_dict = {}
        
        # Cập nhật feedback comment
        metadata_dict['feedback_comment'] = feedback.comment
        
        # Gán lại giá trị
        message.message_metadata = metadata_dict
    
    db.commit()
    
    return None # Trả về 204 No Content