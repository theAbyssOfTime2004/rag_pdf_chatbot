import logging
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func

# LlamaIndex imports for RAG
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.faiss import FaissVectorStore

# Project imports
from app.core.config import settings
from app.services.llamaindex_service import LlamaIndexService
from app.services.vector_service import VectorService
from app.models.chat import ChatSession, ChatMessage

logger = logging.getLogger(__name__)

class ChatService:
    """
    Service quản lý chat và RAG using LlamaIndex QueryEngine.
    Optimized version with better DB operations and QueryEngine usage.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.llamaindex_service = LlamaIndexService()
        self.vector_service = VectorService(db=db)
        
        # Initialize LLM
        self._initialize_llm()
        
        # Initialize Query Engine (lazy loading)
        self._query_engine = None
        self._vector_index = None
        
        logger.info("ChatService initialized with optimized LlamaIndex QueryEngine")
    
    def _initialize_llm(self):
        """Initialize LLM for QueryEngine"""
        try:
            # Initialize Ollama LLM
            self.llm = Ollama(
                model=settings.LLM_MODEL,
                base_url=settings.LLM_BASE_URL,
                temperature=settings.LLM_TEMPERATURE,
                request_timeout=settings.LLM_REQUEST_TIMEOUT
            )
            logger.info(f"Initialized LLM: {settings.LLM_MODEL}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama LLM: {e}")
            # Fallback to mock LLM for development
            from llama_index.core.llms.mock import MockLLM
            self.llm = MockLLM()
            logger.info("Using MockLLM as fallback")
    
    async def _get_or_create_query_engine(self) -> RetrieverQueryEngine:
        """
        ✅ FIX: Simplified QueryEngine creation using as_query_engine()
        """
        try:
            if self._query_engine is None:
                logger.info("Creating new QueryEngine...")
                
                # Create FAISS VectorStore wrapper for LlamaIndex
                faiss_vector_store = FaissVectorStore(
                    faiss_index=self.vector_service.vector_store.index
                )
                
                # Create storage context
                storage_context = StorageContext.from_defaults(
                    vector_store=faiss_vector_store
                )
                
                # Create VectorStoreIndex
                self._vector_index = VectorStoreIndex.from_vector_store(
                    vector_store=faiss_vector_store,
                    storage_context=storage_context,
                    embed_model=self.llamaindex_service.embedding_model
                )
                
                # ✅ FIX: Use simplified as_query_engine() method
                self._query_engine = self._vector_index.as_query_engine(
                    similarity_top_k=settings.MAX_SEARCH_RESULTS,
                    response_mode=settings.CHAT_RESPONSE_MODE,
                    node_postprocessors=[
                        SimilarityPostprocessor(similarity_cutoff=settings.SIMILARITY_THRESHOLD)
                    ],
                    llm=self.llm,
                    use_async=True
                )
                
                logger.info("QueryEngine created successfully using as_query_engine()")
            
            return self._query_engine
            
        except Exception as e:
            logger.error(f"Failed to create QueryEngine: {e}")
            raise
    
    async def chat_with_documents(
        self, 
        query: str, 
        session_id: Optional[int] = None,
        max_tokens: int = 512,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """Chat với tài liệu sử dụng LlamaIndex QueryEngine"""
        try:
            logger.info(f"Processing chat query: '{query[:100]}...'")
            
            # Get or create query engine
            query_engine = await self._get_or_create_query_engine()
            
            # Execute query using LlamaIndex
            logger.info("Executing query with LlamaIndex QueryEngine...")
            response = await query_engine.aquery(query)
            
            # Extract response data
            answer = str(response.response)
            source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
            
            # Process source information
            sources = []
            if include_sources and source_nodes:
                sources = self._process_source_nodes(source_nodes)
            
            # Save to chat history if session provided
            chat_message = None
            if session_id:
                chat_message = self._save_chat_message_sync(
                    session_id=session_id,
                    user_query=query,
                    assistant_response=answer,
                    sources=sources
                )
            
            result = {
                "query": query,
                "answer": answer,
                "sources": sources,
                "total_sources": len(sources),
                "session_id": session_id,
                "message_id": chat_message.id if chat_message else None,
                "model_used": settings.LLM_MODEL
            }
            
            logger.info(f"Chat query completed. Answer length: {len(answer)} chars, Sources: {len(sources)}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process chat query: {e}")
            return {
                "query": query,
                "answer": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "total_sources": 0,
                "error": str(e),
                "session_id": session_id
            }
    
    def _process_source_nodes(self, source_nodes: List[NodeWithScore]) -> List[Dict[str, Any]]:
        """Process source nodes from QueryEngine response"""
        sources = []
        
        for node_with_score in source_nodes:
            try:
                node = node_with_score.node
                score = node_with_score.score
                
                # Extract metadata
                metadata = node.metadata or {}
                
                source_info = {
                    "content": node.get_content()[:500],  # Limit content length
                    "score": float(score) if score is not None else 0.0,
                    "chunk_id": metadata.get("chunk_id"),
                    "document_id": metadata.get("document_id"),
                    "page_number": metadata.get("page_number"),
                    "chunk_index": metadata.get("chunk_index"),
                    "file_name": metadata.get("file_name", "Unknown"),
                    "node_id": node.node_id
                }
                
                # Get additional document info from database if available
                if source_info["document_id"]:
                    doc_info = self._get_document_info(source_info["document_id"])
                    if doc_info:
                        source_info.update(doc_info)
                
                sources.append(source_info)
                
            except Exception as e:
                logger.warning(f"Failed to process source node: {e}")
                continue
        
        # Sort by relevance score
        sources.sort(key=lambda x: x["score"], reverse=True)
        return sources
    
    # ✅ FIX: Add optimized sync methods for better DB operations
    
    def create_chat_session_sync(self, title: Optional[str] = None) -> ChatSession:
        """Create new chat session - sync version for threadpool"""
        try:
            session = ChatSession(
                title=title or settings.DEFAULT_CHAT_TITLE,
                metadata_={"created_by": "user"}
            )
            
            self.db.add(session)
            self.db.commit()
            self.db.refresh(session)
            
            logger.info(f"Created new chat session: {session.id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to create chat session: {e}")
            self.db.rollback()
            raise
    
    def get_chat_sessions_with_message_counts(self, limit: int = 20) -> List[Tuple[ChatSession, int]]:
        """
        ✅ FIX: Get chat sessions with message counts in single optimized query
        """
        try:
            # Single query with LEFT JOIN and COUNT
            result = (
                self.db.query(
                    ChatSession,
                    func.count(ChatMessage.id).label('message_count')
                )
                .outerjoin(ChatMessage, ChatSession.id == ChatMessage.session_id)
                .group_by(ChatSession.id)
                .order_by(ChatSession.updated_at.desc())
                .limit(limit)
                .all()
            )
            
            return [(session, count) for session, count in result]
            
        except Exception as e:
            logger.error(f"Failed to get chat sessions with counts: {e}")
            return []
    
    def get_chat_session_with_message_count(self, session_id: int) -> Optional[Tuple[ChatSession, int]]:
        """Get chat session with message count in single query"""
        try:
            result = (
                self.db.query(
                    ChatSession,
                    func.count(ChatMessage.id).label('message_count')
                )
                .outerjoin(ChatMessage, ChatSession.id == ChatMessage.session_id)
                .filter(ChatSession.id == session_id)
                .group_by(ChatSession.id)
                .first()
            )
            
            return result if result else None
            
        except Exception as e:
            logger.error(f"Failed to get chat session {session_id}: {e}")
            return None
    
    def chat_session_exists(self, session_id: int) -> bool:
        """Check if chat session exists - optimized"""
        try:
            return self.db.query(
                self.db.query(ChatSession).filter(ChatSession.id == session_id).exists()
            ).scalar()
        except Exception as e:
            logger.error(f"Failed to check session existence {session_id}: {e}")
            return False
    
    def delete_chat_session_sync(self, session_id: int) -> bool:
        """Delete chat session - sync version for threadpool"""
        try:
            # Delete messages first
            self.db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).delete()
            
            # Delete session
            deleted = self.db.query(ChatSession).filter(
                ChatSession.id == session_id
            ).delete()
            
            self.db.commit()
            
            if deleted > 0:
                logger.info(f"Deleted chat session {session_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete chat session {session_id}: {e}")
            self.db.rollback()
            return False
    
    def _save_chat_message_sync(
        self,
        session_id: int,
        user_query: str,
        assistant_response: str,
        sources: List[Dict[str, Any]]
    ) -> ChatMessage:
        """Save chat message to database - sync version"""
        try:
            # Create user message
            user_message = ChatMessage(
                session_id=session_id,
                role="user",
                content=user_query,
                metadata_={"query_length": len(user_query)}
            )
            self.db.add(user_message)
            self.db.flush()
            
            # Create assistant message
            assistant_message = ChatMessage(
                session_id=session_id,
                role="assistant",
                content=assistant_response,
                metadata_={
                    "sources_count": len(sources),
                    "response_length": len(assistant_response),
                    "model": settings.LLM_MODEL,
                    "sources": [{"chunk_id": s.get("chunk_id"), "score": s.get("score")} for s in sources[:5]]
                }
            )
            self.db.add(assistant_message)
            self.db.commit()
            
            logger.info(f"Saved chat messages for session {session_id}")
            return assistant_message
            
        except Exception as e:
            logger.error(f"Failed to save chat message: {e}")
            self.db.rollback()
            raise
    
    def _get_document_info(self, document_id: int) -> Optional[Dict[str, Any]]:
        """Get additional document information from database"""
        try:
            from app.models.document import Document
            
            document = self.db.query(Document).filter(
                Document.id == document_id
            ).first()
            
            if document:
                return {
                    "document_filename": document.filename,
                    "document_title": document.original_filename,
                    "file_size": document.file_size,
                    "upload_date": document.created_at.isoformat() if document.created_at else None
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get document info for ID {document_id}: {e}")
            return None
    
    def get_chat_session(self, session_id: int) -> Optional[ChatSession]:
        """Get chat session by ID"""
        return self.db.query(ChatSession).filter(
            ChatSession.id == session_id
        ).first()
    
    def get_chat_messages(self, session_id: int, limit: int = 50) -> List[ChatMessage]:
        """Get chat messages for a session"""
        return (
            self.db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
            .all()
        )
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        return {
            "service_name": "ChatService",
            "llm_model": settings.LLM_MODEL,
            "embedding_model": settings.EMBEDDING_MODEL,
            "query_engine": "LlamaIndex as_query_engine()",
            "vector_store": "FAISS",
            "max_search_results": settings.MAX_SEARCH_RESULTS,
            "similarity_threshold": settings.SIMILARITY_THRESHOLD,
            "response_mode": settings.CHAT_RESPONSE_MODE,
            "has_query_engine": self._query_engine is not None
        }
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'llamaindex_service'):
                self.llamaindex_service.cleanup()
            if hasattr(self, 'vector_service'):
                self.vector_service.cleanup()
            
            self._query_engine = None
            self._vector_index = None
            
            logger.info("ChatService cleanup completed")
            
        except Exception as e:
            logger.warning(f"Error during ChatService cleanup: {e}")