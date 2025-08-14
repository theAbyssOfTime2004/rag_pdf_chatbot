from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
from typing import List, Optional
import time
import logging

from app.database import get_db
from app.schemas.chat import (
    ChatRequest, 
    ChatResponse, 
    ChatSessionCreate, 
    ChatSessionResponse,
    ChatMessageResponse
)
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)

router = APIRouter()

def get_chat_service(db: Session = Depends(get_db)) -> ChatService:
    """Dependency to get ChatService instance"""
    return ChatService(db=db)

@router.post("/sessions", response_model=ChatSessionResponse)
async def create_chat_session(
    session_data: ChatSessionCreate,
    chat_service: ChatService = Depends(get_chat_service)
):
    """Create a new chat session"""
    try:
        # ✅ FIX: Wrap sync DB operation in threadpool
        session = await run_in_threadpool(
            chat_service.create_chat_session_sync,  # We'll create this sync method
            session_data.title
        )
        
        return ChatSessionResponse(
            id=session.id,
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
            message_count=0
        )
        
    except Exception as e:
        logger.error(f"Failed to create chat session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create chat session: {str(e)}"
        )

@router.get("/sessions", response_model=List[ChatSessionResponse])
async def get_chat_sessions(
    limit: int = 20,
    chat_service: ChatService = Depends(get_chat_service)
):
    """Get recent chat sessions with optimized message counting"""
    try:
        # ✅ FIX: Use threadpool for sync DB operations
        sessions_with_counts = await run_in_threadpool(
            chat_service.get_chat_sessions_with_message_counts,  # Optimized method
            limit
        )
        
        result = []
        for session_data in sessions_with_counts:
            session, message_count = session_data
            result.append(ChatSessionResponse(
                id=session.id,
                title=session.title,
                created_at=session.created_at,
                updated_at=session.updated_at,
                message_count=message_count  # ✅ FIX: Single query instead of multiple
            ))
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get chat sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chat sessions: {str(e)}"
        )

@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_chat_session(
    session_id: int,
    chat_service: ChatService = Depends(get_chat_service)
):
    """Get specific chat session"""
    # ✅ FIX: Use threadpool for sync DB operation
    session_data = await run_in_threadpool(
        chat_service.get_chat_session_with_message_count,
        session_id
    )
    
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session {session_id} not found"
        )
    
    session, message_count = session_data
    
    return ChatSessionResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        message_count=message_count
    )

@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessageResponse])
async def get_chat_messages(
    session_id: int,
    limit: int = 50,
    chat_service: ChatService = Depends(get_chat_service)
):
    """Get messages for a chat session"""
    # ✅ FIX: Use threadpool for sync DB operations
    session_exists = await run_in_threadpool(
        chat_service.chat_session_exists,
        session_id
    )
    
    if not session_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session {session_id} not found"
        )
    
    messages = await run_in_threadpool(
        chat_service.get_chat_messages,
        session_id,
        limit
    )
    
    return [
        ChatMessageResponse(
            id=msg.id,
            session_id=msg.session_id,
            role=msg.role,
            content=msg.content,
            created_at=msg.created_at,
            metadata=msg.metadata_  # ✅ FIX: Will handle alias in schema
        )
        for msg in reversed(messages)  # Return in chronological order
    ]

@router.post("/query", response_model=ChatResponse, response_model_by_alias=True)
async def chat_with_documents(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    """
    Chat with documents using LlamaIndex QueryEngine.
    Main endpoint for RAG-based conversation.
    """
    try:
        logger.info(f"Received chat request: '{request.query[:100]}...'")
        
        # ✅ FIX: Validate session with threadpool if provided
        if request.session_id:
            session_exists = await run_in_threadpool(
                chat_service.chat_session_exists,
                request.session_id
            )
            if not session_exists:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Chat session {request.session_id} not found"
                )
        
        # ✅ FIX: Proper timing measurement
        start_time = time.perf_counter()
        
        # Process query using ChatService with LlamaIndex QueryEngine
        result = await chat_service.chat_with_documents(
            query=request.query,
            session_id=request.session_id,
            max_tokens=request.max_tokens or 512,
            include_sources=request.include_sources
        )
        
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        
        # Return response
        response = ChatResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result["sources"],
            total_sources=result["total_sources"],
            session_id=result.get("session_id"),
            message_id=result.get("message_id"),
            model_used=result.get("model_used", "unknown"),
            processing_info={
                "query_engine": "LlamaIndex",
                "retrieval_method": "FAISS Vector Search",
                "synthesis_method": "Tree Summarize",
                "processing_time_seconds": round(processing_time, 3)
            }
        )
        
        logger.info(f"Chat response generated in {processing_time:.3f}s. Answer length: {len(response.answer)}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat query: {str(e)}"
        )

@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: int,
    chat_service: ChatService = Depends(get_chat_service)
):
    """Delete a chat session and all its messages"""
    # ✅ FIX: Use threadpool for sync DB operation
    success = await run_in_threadpool(
        chat_service.delete_chat_session_sync,  # We'll create this sync method
        session_id
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session {session_id} not found"
        )
    
    return {"message": f"Chat session {session_id} deleted successfully"}

@router.get("/info")
async def get_chat_service_info(
    chat_service: ChatService = Depends(get_chat_service)
):
    """Get information about the chat service"""
    return chat_service.get_service_info()

@router.post("/test")
async def test_chat_service(
    chat_service: ChatService = Depends(get_chat_service)
):
    """Test endpoint to verify chat service functionality"""
    try:
        # Test basic functionality
        info = chat_service.get_service_info()
        
        # Try a simple query (without saving to session)
        start_time = time.perf_counter()
        test_result = await chat_service.chat_with_documents(
            query="What is the main topic of the documents?",
            session_id=None,
            include_sources=False
        )
        end_time = time.perf_counter()
        
        return {
            "status": "healthy",
            "service_info": info,
            "test_query": test_result["query"],
            "test_response_length": len(test_result["answer"]),
            "sources_available": test_result["total_sources"] > 0,
            "test_processing_time": round(end_time - start_time, 3)
        }
        
    except Exception as e:
        logger.error(f"Chat service test failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "service_info": chat_service.get_service_info()
        }