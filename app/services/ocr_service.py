from paddleocr import PaddleOCR
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Optional
import logging
from pathlib import Path
import asyncio
import concurrent.futures
from app.core.config import settings

class OCRService:
    """
    OCR service using PaddleOCR for image-based PDFs
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._ocr = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    
    @property
    def ocr(self):
        """Lazy load OCR model"""
        if self._ocr is None:
            try:
                self._ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=getattr(settings, 'OCR_LANGUAGE', 'en'),
                    use_gpu=getattr(settings, 'USE_GPU_OCR', False),
                    show_log=False
                )
                self.logger.info("PaddleOCR initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize PaddleOCR: {e}")
                raise
        return self._ocr
    
    async def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract text from PDF using OCR (async)
        """
        try:
            # Run OCR in thread pool để không block event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, 
                self._extract_text_sync, 
                pdf_path
            )
            return result
            
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            return {
                "text_pages": [],
                "page_info": [],
                "total_pages": 0,
                "extraction_method": "ocr_failed",
                "success": False,
                "error": str(e)
            }
    
    def _extract_text_sync(self, pdf_path: str) -> Dict[str, any]:
        """
        Synchronous OCR extraction
        """
        text_pages = []
        page_info = []
        
        # OCR entire PDF
        results = self.ocr.ocr(pdf_path, cls=True)
        
        for page_num, page_result in enumerate(results, 1):
            if page_result is None:
                # Empty page
                text_pages.append("")
                page_info.append({
                    "page_number": page_num,
                    "text_length": 0,
                    "extraction_method": "ocr_empty",
                    "confidence_avg": 0.0
                })
                continue
            
            # Extract text and confidence scores
            page_text = ""
            confidences = []
            
            for line in page_result:
                if len(line) >= 2:
                    text_info = line[1]
                    if len(text_info) >= 2:
                        text = text_info[0]
                        confidence = text_info[1]
                        
                        # Filter by confidence threshold
                        conf_threshold = getattr(settings, 'OCR_CONFIDENCE_THRESHOLD', 0.7)
                        if confidence >= conf_threshold:
                            page_text += text + "\n"
                            confidences.append(confidence)
            
            text_pages.append(page_text.strip())
            page_info.append({
                "page_number": page_num,
                "text_length": len(page_text),
                "extraction_method": "ocr",
                "confidence_avg": np.mean(confidences) if confidences else 0.0,
                "lines_detected": len(page_result)
            })
        
        return {
            "text_pages": text_pages,
            "page_info": page_info,
            "total_pages": len(text_pages),
            "extraction_method": "paddleocr",
            "success": True
        }
    
    def preprocess_image(self, image_path: str) -> str:
        """
        Preprocess image for better OCR results
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Noise removal
            denoised = cv2.medianBlur(gray, 3)
            
            # Threshold to binary
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Save preprocessed image
            preprocessed_path = image_path.replace('.', '_processed.')
            cv2.imwrite(preprocessed_path, thresh)
            
            return preprocessed_path
            
        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {e}")
            return image_path  # Return original if preprocessing fails
    
    def cleanup(self):
        """
        Cleanup resources
        """
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)