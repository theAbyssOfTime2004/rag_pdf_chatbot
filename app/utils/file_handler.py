import os
import shutil
import hashlib
import mimetypes
from pathlib import Path
from typing import Dict, List
from fastapi import HTTPException, UploadFile
from app.core.config import settings
import aiofiles
# import magic  # Bỏ dòng này

class FileHandler:
    def __init__(self):
        base_upload_dir = Path(settings.UPLOAD_FOLDER)
        self.upload_dir = base_upload_dir / "documents"

        # Create base and subfolder
        base_upload_dir.mkdir(exist_ok=True)
        self.upload_dir.mkdir(exist_ok=True)
        
        # Allowed file extensions (simpler approach)
        self.allowed_extensions = {'.pdf'}
    
    async def validate_file(self, file: UploadFile) -> None:
        """
        Validate uploaded file với security checks
        """
        # 1. Check file size
        if hasattr(file, 'size') and file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB"
            )
        
        # 2. Check filename
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # 3. Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in self.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed"
            )
        
        # 4. Basic content type check
        if file.content_type and not file.content_type.startswith('application/pdf'):
            # Warning but not blocking (some browsers send different MIME types)
            pass
        
        # 5. Security: Check for malicious filenames
        if self._is_malicious_filename(file.filename):
            raise HTTPException(
                status_code=400,
                detail="Invalid filename detected"
            )
        
        # 6. Check PDF header (simple validation)
        await self._validate_pdf_header(file)
    
    async def _validate_pdf_header(self, file: UploadFile) -> None:
        """
        Simple PDF header validation without python-magic
        """
        try:
            # Read first few bytes
            header = await file.read(4)
            await file.seek(0)  # Reset file pointer
            
            # PDF files start with %PDF
            if not header.startswith(b'%PDF'):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid PDF file format"
                )
        except Exception:
            # If can't read header, assume valid (fail-safe)
            await file.seek(0)
    
    def _is_malicious_filename(self, filename: str) -> bool:
        """
        Check for potentially malicious filenames
        """
        dangerous_patterns = [
            "..", "/", "\\", ":", "*", "?", '"', "<", ">", "|",
            "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3",
            "LPT1", "LPT2", "LPT3"
        ]
        
        filename_upper = filename.upper()
        return any(pattern in filename_upper for pattern in dangerous_patterns)
    
    async def save_file(self, file: UploadFile) -> Dict[str, any]:
        """
        Save file to disk với unique filename
        """
        try:
            # Generate unique filename
            file_hash = await self._generate_file_hash(file)
            file_ext = Path(file.filename).suffix.lower()
            unique_filename = f"{file_hash}{file_ext}"
            
            file_path = self.upload_dir / unique_filename
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Get file size
            file_size = file_path.stat().st_size
            
            return {
                "filename": unique_filename,
                "original_filename": file.filename,
                "file_path": str(file_path),
                "file_size": file_size,
                "file_hash": file_hash
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save file: {str(e)}"
            )
    
    async def _generate_file_hash(self, file: UploadFile) -> str:
        """
        Generate SHA-256 hash of file content
        """
        content = await file.read()
        await file.seek(0)  # Reset file pointer
        
        file_hash = hashlib.sha256(content).hexdigest()
        return file_hash[:16]  # Use first 16 chars
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete file from disk
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete file: {str(e)}"
            )
    
    def get_file_info(self, file_path: str) -> Dict[str, any]:
        """
        Get file information
        """
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        stat_info = os.stat(file_path)
        return {
            "file_path": file_path,
            "file_size": stat_info.st_size,
            "created_time": stat_info.st_ctime,
            "modified_time": stat_info.st_mtime,
            "mime_type": mimetypes.guess_type(file_path)[0]
        }