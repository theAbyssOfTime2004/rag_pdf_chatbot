from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models.document import Document, DocumentChunk
from app.schemas.document import DocumentResponse, DocumentCreate

router = APIRouter()

@router.get("/", response_model=List[DocumentResponse])
async def get_documents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Lấy danh sách tất cả documents"""
    documents = db.query(Document).offset(skip).limit(limit).all()
    return documents

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: int, db: Session = Depends(get_db)):
    """Lấy thông tin document theo ID"""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload PDF document"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Tạo document record
    document = Document(
        filename=file.filename,
        original_filename=file.filename,
        file_size=0,  # Sẽ tính sau
        file_path=f"uploads/{file.filename}",
        content_type=file.content_type,
        processing_status="pending"
    )
    
    db.add(document)
    db.commit()
    db.refresh(document)
    
    return {
        "message": "Document uploaded successfully",
        "document_id": document.id,
        "filename": document.filename
    }

@router.delete("/{document_id}")
async def delete_document(document_id: int, db: Session = Depends(get_db)):
    """Xóa document"""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    db.delete(document)
    db.commit()
    
    return {"message": "Document deleted successfully"}