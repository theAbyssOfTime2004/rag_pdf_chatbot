import PyPDF2
import pdfplumber
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from PIL import Image
import cv2
import numpy as np

class PDFUtils:
    """
    Smart PDF processing utilities với hybrid approach
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_pdf_type(self, pdf_path: str) -> str:
        """
        Detect if PDF is text-based or image-based
        Returns: 'text-based', 'image-based', or 'mixed'
        """
        try:
            text_density = self._calculate_text_density(pdf_path)
            
            if text_density > 0.5:  # >50% pages có text đầy đủ
                return "text-based"
            elif text_density > 0.1:  # 10-50% có text
                return "mixed"
            else:  # <10% có text
                return "image-based"
                
        except Exception as e:
            self.logger.warning(f"Error detecting PDF type: {e}")
            return "image-based"  # Fallback to OCR
    
    def _calculate_text_density(self, pdf_path: str) -> float:
        """
        Calculate ratio of pages with meaningful text content
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                pages_with_text = 0
                
                # Sample max 5 pages để tăng tốc
                sample_pages = min(5, total_pages)
                
                for i in range(sample_pages):
                    page = reader.pages[i]
                    text = page.extract_text().strip()
                    
                    # Check if page có meaningful text
                    if len(text) > 100 and self._has_meaningful_text(text):
                        pages_with_text += 1
                
                return pages_with_text / sample_pages
                
        except Exception as e:
            self.logger.error(f"Error calculating text density: {e}")
            return 0.0
    
    def _has_meaningful_text(self, text: str) -> bool:
        """
        Check if text contains meaningful content (not just random characters)
        """
        if len(text) < 50:
            return False
            
        # Check ratio of alphanumeric characters
        alphanum_count = sum(c.isalnum() for c in text)
        alphanum_ratio = alphanum_count / len(text)
        
        # Check for common words/patterns
        common_words = ['the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with']
        text_lower = text.lower()
        word_matches = sum(1 for word in common_words if word in text_lower)
        
        return alphanum_ratio > 0.7 and word_matches > 2
    
    def extract_text_fast(self, pdf_path: str) -> Dict[str, any]:
        """
        Fast text extraction using pdfplumber (better than PyPDF2)
        """
        try:
            text_content = []
            page_info = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    
                    if page_text:
                        text_content.append(page_text)
                        page_info.append({
                            "page_number": page_num,
                            "text_length": len(page_text),
                            "extraction_method": "text-based"
                        })
                    else:
                        text_content.append("")
                        page_info.append({
                            "page_number": page_num,
                            "text_length": 0,
                            "extraction_method": "failed"
                        })
            
            return {
                "text_pages": text_content,
                "page_info": page_info,
                "total_pages": len(text_content),
                "extraction_method": "pdfplumber",
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Fast text extraction failed: {e}")
            return {
                "text_pages": [],
                "page_info": [],
                "total_pages": 0,
                "extraction_method": "failed",
                "success": False,
                "error": str(e)
            }
    
    def get_pdf_info(self, pdf_path: str) -> Dict[str, any]:
        """
        Get basic PDF information
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                info = reader.metadata
                
                return {
                    "page_count": len(reader.pages),
                    "title": info.get('/Title', 'Unknown') if info else 'Unknown',
                    "author": info.get('/Author', 'Unknown') if info else 'Unknown',
                    "creator": info.get('/Creator', 'Unknown') if info else 'Unknown',
                    "encrypted": reader.is_encrypted,
                    "file_size": Path(pdf_path).stat().st_size
                }
                
        except Exception as e:
            self.logger.error(f"Error getting PDF info: {e}")
            return {
                "page_count": 0,
                "title": "Unknown",
                "author": "Unknown", 
                "creator": "Unknown",
                "encrypted": False,
                "file_size": 0,
                "error": str(e)
            }
    
    def validate_pdf(self, pdf_path: str) -> Tuple[bool, str]:
        """
        Validate PDF file integrity
        """
        try:
            # Check file exists
            if not Path(pdf_path).exists():
                return False, "File does not exist"
            
            # Check file size
            file_size = Path(pdf_path).stat().st_size
            if file_size == 0:
                return False, "File is empty"
            
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                return False, "File too large (>100MB)"
            
            # Try to open with PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Check if encrypted
                if reader.is_encrypted:
                    return False, "PDF is password protected"
                
                # Check if has pages
                if len(reader.pages) == 0:
                    return False, "PDF has no pages"
                
                # Try to read first page
                reader.pages[0].extract_text()
            
            return True, "Valid PDF"
            
        except Exception as e:
            return False, f"Invalid PDF: {str(e)}"