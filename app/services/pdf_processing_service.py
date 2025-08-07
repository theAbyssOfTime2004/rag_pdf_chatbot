from typing import Dict, List, Optional
import asyncio
import logging
from pathlib import Path
from app.utils.pdf_utils import PDFUtils
from app.services.ocr_service import OCRService
from app.services.text_processor import TextProcessor
from app.models.document import Document, DocumentChunk
from app.database import get_db
from sqlalchemy.orm import Session

class PDFProcessingService:
    """
    Main service để orchestrate PDF processing pipeline
    """
    
    def __init__(self):
        self.pdf_utils = PDFUtils()
        self.ocr_service = OCRService()
        self.text_processor = TextProcessor()
        self.logger = logging.getLogger(__name__)
    
    async def process_document(self, document_id: int, db: Session):
        """
        Complete PDF processing pipeline
        """
        try:
            # 1. Get document from database
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            # 2. Update status to processing
            document.processing_status = "processing"
            db.commit()
            
            # 3. Validate PDF
            is_valid, validation_msg = self.pdf_utils.validate_pdf(document.file_path)
            if not is_valid:
                raise ValueError(f"Invalid PDF: {validation_msg}")
            
            # 4. Detect PDF type and extract text
            pdf_type = self.pdf_utils.detect_pdf_type(document.file_path)
            
            if pdf_type == "text-based":
                extraction_result = self.pdf_utils.extract_text_fast(document.file_path)
            else:
                extraction_result = await self.ocr_service.extract_text_from_pdf(document.file_path)
            
            if not extraction_result["success"]:
                raise ValueError(f"Text extraction failed: {extraction_result.get('error', 'Unknown error')}")
            
            # 5. Combine all pages into full text
            full_text = "\n\n".join(extraction_result["text_pages"])
            
            # 6. Clean text
            cleaned_text = self.text_processor.clean_text(full_text)
            
            # 7. Split into chunks
            chunks_data = self.text_processor.split_into_chunks(
                cleaned_text,
                metadata={"document_id": document_id}
            )
            
            # 8. Save chunks to database
            await self._save_chunks_to_db(document_id, chunks_data, extraction_result["page_info"], db)
            
            # 9. Update document status
            document.text_content = cleaned_text
            document.chunk_count = len(chunks_data)
            document.processed = True
            document.processing_status = "completed"
            db.commit()

            # 10. Add index chunks to vector store
            from app.services.vector_service import VectorService
            vector_service = VectorService()
            
            result = await vector_service.index_document_chunks(document_id, db)
            
            if result['success']:
                logging.info(f"✅ Vector indexing completed for document {document_id}")
            else:
                raise Exception(f"Vector indexing failed: {result.get('error')}")
            
            return {
                "status": "success",
                "document_id": document_id,
                "pdf_type": pdf_type,
                "extraction_method": extraction_result["extraction_method"],
                "total_pages": extraction_result["total_pages"],
                "chunks_created": len(chunks_data),
                "text_length": len(cleaned_text)
            }
            
        except Exception as e:
            # Update document status to failed
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.processing_status = 'failed'
                document.processing_error = str(e)
                db.commit()
            
            self.logger.error(f"PDF processing failed for document {document_id}: {e}")
            return {
                "status": "failed",
                "document_id": document_id,
                "error": str(e)
            }
    
    async def _save_chunks_to_db(
        self, 
        document_id: int, 
        chunks_data: List[Dict], 
        page_info: List[Dict],
        db: Session
    ):
        """
        Save chunks to database
        """
        # Delete existing chunks
        db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
        
        # Create new chunks
        for chunk_data in chunks_data:
            # Estimate page number from chunk position
            page_number = self._estimate_page_number(
                chunk_data["start_pos"], 
                chunk_data["end_pos"], 
                page_info
            )
            
            chunk = DocumentChunk(
                document_id=document_id,
                chunk_text=chunk_data["chunk_text"],
                chunk_index=chunk_data["chunk_index"],
                page_number=page_number
            )
            db.add(chunk)
        
        db.commit()
    
    def _estimate_page_number(self, start_pos: int, end_pos: int, page_info: List[Dict]) -> Optional[int]:
        """
        Estimate page number based on text position
        """
        total_chars = 0
        chunk_middle = (start_pos + end_pos) // 2
        
        for page in page_info:
            page_chars = page.get("text_length", 0)
            if total_chars <= chunk_middle <= total_chars + page_chars:
                return page.get("page_number")
            total_chars += page_chars
        
        return 1  # Default to page 1