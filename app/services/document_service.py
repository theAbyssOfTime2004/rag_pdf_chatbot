import logging
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from typing import List, Optional, Dict, Any
from fastapi import HTTPException, UploadFile
from pathlib import Path

from app.models.document import Document, DocumentChunk
from app.schemas.document import DocumentResponse, DocumentCreateRequest
from app.services.vector_service import VectorService
from app.services.llamaindex_service import LlamaIndexService  # THÊM
from app.utils.file_handler import save_upload_file, get_file_path, validate_file_type
from app.core.config import settings

logger = logging.getLogger(__name__)

class DocumentService:
    """
    Service quản lý tài liệu - được refactor để sử dụng LlamaIndexService
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.vector_service = VectorService(db=db)
        self.llama_index_service = LlamaIndexService()  # THÊM
        logger.info("DocumentService initialized with LlamaIndexService")

    async def process_and_store_document(self, file: UploadFile) -> Document:
        """
        Xử lý và lưu trữ tài liệu hoàn chỉnh - refactored với LlamaIndexService
        """
        try:
            # Validate file type
            if not self.llama_index_service.is_allowed_file(file.filename):
                raise HTTPException(
                    status_code=400,
                    detail=f"File type not supported: {file.filename}"
                )

            # Read file content
            file_content = await file.read()
            
            # Validate file size
            if len(file_content) > settings.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE} bytes"
                )

            # Save file to disk
            file_path = await save_upload_file(file, file_content)
            logger.info(f"File saved to: {file_path}")

            # Create document record in database
            db_document = Document(
                filename=file.filename,
                original_filename=file.filename,
                file_path=str(file_path),
                file_size=len(file_content),
                content_type=file.content_type or "application/octet-stream",
                processing_status="processing"
            )
            
            self.db.add(db_document)
            self.db.commit()
            self.db.refresh(db_document)
            logger.info(f"Created document record with ID: {db_document.id}")

            # === REFACTOR START: Sử dụng LlamaIndexService ===
            logger.info(f"Starting LlamaIndex processing for: {file.filename}")

            # Process file using LlamaIndexService
            processing_result = await self.llama_index_service.process_uploaded_file_complete(
                file_content=file_content,
                filename=file.filename,
                generate_embeddings=True
            )

            if processing_result["status"] != "success":
                db_document.processing_status = "failed"
                db_document.error_message = processing_result.get("error", "Unknown processing error")
                self.db.commit()
                raise HTTPException(status_code=500, detail=db_document.error_message)

            nodes = processing_result["nodes"]
            embeddings = processing_result["embeddings"]

            if not nodes:
                db_document.processing_status = "failed"
                db_document.error_message = "No content extracted from the document."
                self.db.commit()
                raise HTTPException(status_code=400, detail="No content could be extracted from the document")

            # Create DocumentChunk records
            chunks_to_add = []
            chunk_texts = []
            
            for i, node in enumerate(nodes):
                chunk_content = node.get_content()
                chunk_texts.append(chunk_content)
                
                chunk = DocumentChunk(
                    document_id=db_document.id,
                    content=chunk_content,
                    chunk_index=i,
                    page_number=node.metadata.get("page_label"),
                    metadata_=node.metadata  # Store full node metadata
                )
                chunks_to_add.append(chunk)

            # Add chunks to database
            self.db.add_all(chunks_to_add)
            self.db.flush()  # flush để chunks có ID

            # Get chunk IDs
            chunk_ids = [chunk.id for chunk in chunks_to_add]

            # Add embeddings to vector store
            await self.vector_service.add_chunks_to_index(
                document_id=db_document.id,
                chunk_ids=chunk_ids,
                chunk_texts=chunk_texts,
                embeddings=embeddings
            )
            # === REFACTOR END ===

            # Update document status
            db_document.processing_status = "completed"
            db_document.total_chunks = len(chunks_to_add)
            self.db.commit()
            self.db.refresh(db_document)
            
            logger.info(f"Successfully processed document ID: {db_document.id} with {len(chunks_to_add)} chunks")
            return db_document

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing document {file.filename}: {e}", exc_info=True)
            
            # Update document status to failed
            if 'db_document' in locals():
                db_document.processing_status = "failed"
                db_document.error_message = str(e)
                self.db.commit()
            
            raise HTTPException(
                status_code=500,
                detail=f"Document processing failed: {str(e)}"
            )

    def get_document_by_id(self, document_id: int) -> Optional[Document]:
        """Lấy tài liệu theo ID"""
        return self.db.query(Document).filter(Document.id == document_id).first()

    def get_documents(self, skip: int = 0, limit: int = 100) -> List[Document]:
        """Lấy danh sách tài liệu"""
        return (
            self.db.query(Document)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_document_chunks(self, document_id: int) -> List[DocumentChunk]:
        """Lấy các chunks của tài liệu"""
        return (
            self.db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
            .all()
        )

    async def delete_document(self, document_id: int) -> bool:
        """Xóa tài liệu và các chunks liên quan"""
        try:
            document = self.get_document_by_id(document_id)
            if not document:
                return False

            # Delete from vector store
            await self.vector_service.remove_document_from_index(document_id)

            # Delete chunks from database
            self.db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).delete()

            # Delete document from database
            self.db.delete(document)
            self.db.commit()

            # Delete physical file
            try:
                file_path = Path(document.file_path)
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted file: {file_path}")
            except Exception as e:
                logger.warning(f"Could not delete file {document.file_path}: {e}")

            logger.info(f"Successfully deleted document ID: {document_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            self.db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete document: {str(e)}"
            )

    def search_documents(self, query: str, limit: int = 10) -> List[Document]:
        """Tìm kiếm tài liệu theo tên"""
        return (
            self.db.query(Document)
            .filter(
                or_(
                    Document.filename.ilike(f"%{query}%"),
                    Document.original_filename.ilike(f"%{query}%")
                )
            )
            .limit(limit)
            .all()
        )

    def get_processing_stats(self) -> Dict[str, Any]:
        """Lấy thống kê xử lý tài liệu"""
        from sqlalchemy import func
        
        stats = (
            self.db.query(
                func.count(Document.id).label('total_documents'),
                func.count(Document.id).filter(Document.processing_status == 'completed').label('completed'),
                func.count(Document.id).filter(Document.processing_status == 'processing').label('processing'),
                func.count(Document.id).filter(Document.processing_status == 'failed').label('failed'),
                func.sum(Document.file_size).label('total_size'),
                func.sum(Document.total_chunks).label('total_chunks')
            )
            .first()
        )
        
        return {
            "total_documents": stats.total_documents or 0,
            "completed": stats.completed or 0,
            "processing": stats.processing or 0,
            "failed": stats.failed or 0,
            "total_size_bytes": stats.total_size or 0,
            "total_chunks": stats.total_chunks or 0
        }