from typing import List, Dict, Optional
import re
import logging
from app.core.config import settings

class TextProcessor:
    """
    Text processing service for chunking and cleaning
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.chunk_size = getattr(settings, 'CHUNK_SIZE', 1000)
        self.chunk_overlap = getattr(settings, 'CHUNK_OVERLAP', 200)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\"\'\/]', '', text)
        
        # Fix common OCR errors
        text = self._fix_ocr_errors(text)
        
        # Normalize line breaks
        text = text.replace('\n\n', '\n').replace('\r', '')
        
        return text.strip()
    
    def _fix_ocr_errors(self, text: str) -> str:
        """
        Fix common OCR recognition errors
        """
        # Common OCR corrections
        corrections = {
            r'\bl\b': 'I',  # lowercase l → I
            r'\b0\b': 'O',  # zero → O (context dependent)
            r'rn': 'm',     # rn → m
            r'vv': 'w',     # vv → w
        }
        
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def split_into_chunks(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Split text into overlapping chunks
        """
        if not text or len(text) < 50:
            return []
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If not at end of text, try to break at sentence boundary
            if end < len(text):
                end = self._find_sentence_boundary(text, end)
            
            # Extract chunk
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_data = {
                    "chunk_text": chunk_text,
                    "chunk_index": chunk_index,
                    "start_pos": start,
                    "end_pos": end,
                    "chunk_length": len(chunk_text)
                }
                
                # Add metadata if provided
                if metadata:
                    chunk_data.update(metadata)
                
                chunks.append(chunk_data)
                chunk_index += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start <= 0:
                start = end
        
        return chunks
    
    def _find_sentence_boundary(self, text: str, preferred_end: int) -> int:
        """
        Find good sentence boundary near preferred end position
        """
        # Look for sentence endings in a window around preferred position
        window_start = max(0, preferred_end - 100)
        window_end = min(len(text), preferred_end + 100)
        window_text = text[window_start:window_end]
        
        # Find sentence boundaries (., !, ?)
        sentence_endings = []
        for match in re.finditer(r'[.!?]\s+', window_text):
            abs_pos = window_start + match.end()
            sentence_endings.append(abs_pos)
        
        if sentence_endings:
            # Choose closest to preferred position
            closest = min(sentence_endings, key=lambda x: abs(x - preferred_end))
            return closest
        
        # Fallback: look for word boundary
        for i in range(preferred_end, max(0, preferred_end - 100), -1):
            if text[i].isspace():
                return i
        
        return preferred_end