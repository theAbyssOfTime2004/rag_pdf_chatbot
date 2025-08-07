from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from typing import List, Optional, Dict, Any
from fastapi import HTTPException, UploadFile
from app.models.document import Document, DocumentChunk
from app.schemas.document import DocumentCreate, DocumentUpdate
from app.utils.file_handler import FileHandler
from app.core.config import settings
import logging
import os
from datetime import datetime

class DocumentService:
    def __init__(self, db: Session):
        self.db = db
        self.file_handler = FileHandler()
    
    async def upload_document(
        self, 
        file: UploadFile, 
        user_id: Optional[str] = None
    ) -> Document:
        """
        Upload và lưu document với validation
        """
        try:
            # 1. Validate file
            await self.file_handler.validate_file(file)
            
            # 2. Save file to disk
            file_info = await self.file_handler.save_file(file)
            
            # 3. Create document record
            document = Document(
                filename=file_info["filename"],
                original_filename=file.filename,
                file_size=file_info["file_size"],
                file_path=file_info["file_path"],
                content_type=file.content_type,
                processing_status="pending",
                processed=False
            )
            
            self.db.add(document)
            self.db.commit()
            self.db.refresh(document)
            
            logging.info(f"Document uploaded successfully: {document.id}")
            return document
            
        except Exception as e:
            self.db.rollback()
            # Cleanup file if database fails
            if 'file_info' in locals():
                self.file_handler.delete_file(file_info["file_path"])
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    def get_documents(
        self, 
        skip: int = 0, 
        limit: int = 100,
        status: Optional[str] = None,
        search: Optional[str] = None
    ) -> List[Document]:
        """
        Lấy danh sách documents với filtering
        """
        query = self.db.query(Document)
        
        # Filter by status
        if status:
            query = query.filter(Document.processing_status == status)
        
        # Search by filename
        if search:
            query = query.filter(
                or_(
                    Document.filename.contains(search),
                    Document.original_filename.contains(search)
                )
            )
        
        return query.offset(skip).limit(limit).all()
    
    def get_document_by_id(self, document_id: int) -> Optional[Document]:
        """
        Lấy document theo ID
        """
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    
    def update_document(
        self, 
        document_id: int, 
        document_update: DocumentUpdate
    ) -> Document:
        """
        Cập nhật thông tin document
        """
        document = self.get_document_by_id(document_id)
        
        update_data = document_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(document, field, value)
        
        self.db.commit()
        self.db.refresh(document)
        
        logging.info(f"Document updated: {document_id}")
        return document
    
    def delete_document(self, document_id: int) -> bool:
        """
        Xóa document và file liên quan
        """
        document = self.get_document_by_id(document_id)
        
        try:
            # 1. Delete file from disk
            if os.path.exists(document.file_path):
                os.remove(document.file_path)
                logging.info(f"File deleted: {document.file_path}")
            
            # 2. Delete from database (cascade will handle chunks)
            self.db.delete(document)
            self.db.commit()
            
            logging.info(f"Document deleted successfully: {document_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to delete document: {str(e)}"
            )
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """
        Thống kê documents
        """
        total_docs = self.db.query(Document).count()
        processed_docs = self.db.query(Document).filter(Document.processed == True).count()
        pending_docs = self.db.query(Document).filter(
            Document.processing_status == "pending"
        ).count()
        
        total_size = self.db.query(Document).with_entities(
            self.db.func.sum(Document.file_size)
        ).scalar() or 0
        
        return {
            "total_documents": total_docs,
            "processed_documents": processed_docs,
            "pending_documents": pending_docs,
            "failed_documents": total_docs - processed_docs - pending_docs,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }
    
    async def trigger_manual_processing(self, document_id: int) -> Dict[str, Any]:
        """Manually trigger processing for a pending document"""
        
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if document.processing_status == 'completed':
            return {"message": "Document already processed", "status": "completed"}
        
        # Reset status và trigger processing
        document.processing_status = 'processing'
        self.db.commit()
        
        try:
            from app.services.pdf_processing_service import PDFProcessingService
            
            pdf_processor = PDFProcessingService()
            result = await pdf_processor.process_document(document_id, self.db)
            
            if result.get("status") == "success":
                return {"message": "Processing completed successfully", "status": "completed"}
            else:
                return {"message": f"Processing failed: {result.get('error')}", "status": "failed"}
                
        except Exception as e:
            # Update document status to failed
            document.processing_status = 'failed'
            document.processing_error = str(e)
            self.db.commit()
            
            logging.error(f"Document processing failed for ID {document_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {e}")