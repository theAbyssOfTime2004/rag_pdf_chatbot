from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from app.database import get_db
from app.models.document import Document, DocumentChunk
from app.schemas.document import (
    DocumentResponse, 
    DocumentCreate, 
    DocumentUpdate,
    DocumentStatistics,
    BulkDeleteRequest,
    BulkDeleteResponse
)
from app.services.document_service import DocumentService
import logging

router = APIRouter()

def get_document_service(db: Session = Depends(get_db)) -> DocumentService:
    """Dependency để inject DocumentService"""
    return DocumentService(db)

@router.get("/", response_model=List[DocumentResponse])
async def get_documents(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    status: Optional[str] = Query(None, description="Filter by processing status"),
    search: Optional[str] = Query(None, description="Search in filename"),
    document_service: DocumentService = Depends(get_document_service)
):
    """
    Lấy danh sách documents với filtering và pagination
    """
    try:
        documents = document_service.get_documents(
            skip=skip, 
            limit=limit, 
            status=status, 
            search=search
        )
        return documents
    except Exception as e:
        logging.error(f"Error getting documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")

@router.get("/statistics", response_model=DocumentStatistics)
async def get_document_statistics(
    document_service: DocumentService = Depends(get_document_service)
):
    """
    Lấy thống kê tổng quan về documents
    """
    try:
        stats = document_service.get_document_statistics()
        return stats
    except Exception as e:
        logging.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    document_service: DocumentService = Depends(get_document_service)
):
    """
    Lấy thông tin chi tiết document theo ID
    """
    try:
        document = document_service.get_document_by_id(document_id)
        return document
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document")

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to upload"),
    document_service: DocumentService = Depends(get_document_service)
):
    """
    Upload PDF document với security validation
    """
    try:
        document = await document_service.upload_document(file)
        
        # Schedule background processing (for future PDF processing task)
        # background_tasks.add_task(process_document_background, document.id)
        
        return document
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail="Upload failed")

@router.put("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: int,
    document_update: DocumentUpdate,
    document_service: DocumentService = Depends(get_document_service)
):
    """
    Cập nhật thông tin document
    """
    try:
        document = document_service.update_document(document_id, document_update)
        return document
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error updating document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update document")

@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    document_service: DocumentService = Depends(get_document_service)
):
    """
    Xóa document và file liên quan
    """
    try:
        success = document_service.delete_document(document_id)
        if success:
            return {"message": "Document deleted successfully", "document_id": document_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete document")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

@router.post("/bulk-delete", response_model=BulkDeleteResponse)
async def bulk_delete_documents(
    bulk_request: BulkDeleteRequest,
    document_service: DocumentService = Depends(get_document_service)
):
    """
    Xóa nhiều documents cùng lúc
    """
    deleted_ids = []
    failed_ids = []
    
    for document_id in bulk_request.document_ids:
        try:
            success = document_service.delete_document(document_id)
            if success:
                deleted_ids.append(document_id)
            else:
                failed_ids.append(document_id)
        except Exception as e:
            logging.error(f"Error deleting document {document_id}: {str(e)}")
            failed_ids.append(document_id)
    
    return BulkDeleteResponse(
        deleted_count=len(deleted_ids),
        failed_count=len(failed_ids),
        deleted_ids=deleted_ids,
        failed_ids=failed_ids
    )

@router.get("/{document_id}/chunks")
async def get_document_chunks(
    document_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """
    Lấy chunks của document (cho debugging/preview)
    """
    # Verify document exists
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    chunks = db.query(DocumentChunk).filter(
        DocumentChunk.document_id == document_id
    ).offset(skip).limit(limit).all()
    
    return {
        "document_id": document_id,
        "total_chunks": document.chunk_count,
        "chunks": [
            {
                "id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "text_preview": chunk.chunk_text[:200] + "..." if len(chunk.chunk_text) > 200 else chunk.chunk_text
            }
            for chunk in chunks
        ]
    }

@router.post("/{document_id}/process")
async def process_document(
    document_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Trigger PDF processing for uploaded document
    """
    from app.services.pdf_processing_service import PDFProcessingService
    
    pdf_service = PDFProcessingService()
    background_tasks.add_task(pdf_service.process_document, document_id, db)
    
    return {
        "message": "PDF processing started",
        "document_id": document_id,
        "status": "processing"
    }