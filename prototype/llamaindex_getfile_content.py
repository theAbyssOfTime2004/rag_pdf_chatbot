"""
File Processing Prototype using LlamaIndex
Chỉ thực hiện đến bước 2: Upload file và trích xuất text
Không thực hiện chunking hay vector processing
"""
# Ngăn auto-loading
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


import tempfile
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# LlamaIndex imports
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.readers.file import (
    PyMuPDFReader,
    DocxReader,
    PptxReader,
    PandasCSVReader,        
    HTMLTagReader
)
from llama_index.readers.json import JSONReader

# Disable embedding auto-loading
Settings.embed_model = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration

ALLOWED_EXTENSIONS = {
    "txt", "docx", "pdf", "pptx", "csv", 
    "xlsx", "xls", "json", "html", "htm"
}

# LlamaIndex reader mapping
LLAMAINDEX_READERS = {
    "pdf": PyMuPDFReader(),
    "docx": DocxReader(),
    "pptx": PptxReader(),
    "csv": PandasCSVReader(),
    "json": JSONReader(),
    "html": HTMLTagReader(),
    "htm": HTMLTagReader(),
}

# Pydantic Models

class FileProcessingResult(BaseModel):
    filename: str
    file_size: int
    content_type: str
    file_extension: str
    extracted_text: str
    text_length: int
    status: str
    message: str

class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
    detail: Optional[str] = None

# File Processing Class

class LlamaIndexFileProcessor:
    """
    File processor using LlamaIndex readers
    Chỉ thực hiện trích xuất text, không chunking
    """
    
    def __init__(self):
        self.readers = LLAMAINDEX_READERS
        
    def is_allowed_file(self, filename: str) -> bool:
        """Check if file extension is supported"""
        if not filename or '.' not in filename:
            return False
        extension = filename.rsplit('.', 1)[1].lower()
        return extension in ALLOWED_EXTENSIONS
    
    def get_file_extension(self, filename: str) -> str:
        """Extract file extension"""
        if not filename or '.' not in filename:
            return ""
        return filename.rsplit('.', 1)[1].lower()

    async def process_uploaded_file(self, file: UploadFile) -> FileProcessingResult:
        """
        Process uploaded file and extract text using LlamaIndex
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            FileProcessingResult with extracted text
            
        Raises:
            HTTPException: If file processing fails
        """
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
            
        if not self.is_allowed_file(file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"File type not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        filename = file.filename
        file_extension = self.get_file_extension(filename)
        content_type = file.content_type or "unknown"
        
        logger.info(f"Processing file: {filename} (type: {content_type})")
        
        # Read file content
        try:
            file_content = await file.read()
            file_size = len(file_content)
            
            if file_size == 0:
                raise HTTPException(status_code=400, detail="File is empty")
                
            logger.info(f"File size: {file_size} bytes")
            
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
        
        # Create temporary file for LlamaIndex processing
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=f".{file_extension}",
                prefix="llamaindex_"
            ) as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name
            
            logger.info(f"Created temporary file: {temp_path}")
            
            # Extract text using LlamaIndex
            extracted_text = await self._extract_text_with_llamaindex(
                temp_path, file_extension
            )
            
            # Prepare result
            result = FileProcessingResult(
                filename=filename,
                file_size=file_size,
                content_type=content_type,
                file_extension=file_extension,
                extracted_text=extracted_text,
                text_length=len(extracted_text),
                status="success",
                message=f"Successfully extracted {len(extracted_text)} characters from {filename}"
            )
            
            logger.info(f"Successfully processed {filename}: {len(extracted_text)} characters extracted")
            return result
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error processing file: {str(e)}"
            )
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    logger.info(f"Cleaned up temporary file: {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_path}: {e}")

    def _excel_to_temp_csvs(self, xl_path: str) -> list[str]:
        """Convert .xlsx/.xls to one or more temporary CSV files (one per sheet)."""
        tmp_csv_paths = []
        try:
            sheets = pd.read_excel(xl_path, sheet_name=None)
            for sheet_name, df in sheets.items():
                safe = str(sheet_name).replace("/", "_").replace("\\", "_").replace(" ", "_")
                tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=f"__{safe}.csv", prefix="xlsx2csv_")
                df.to_csv(tmp_csv.name, index=False)
                tmp_csv_paths.append(tmp_csv.name)
                tmp_csv.close()  # Đóng file handle
            return tmp_csv_paths
        except Exception as e:
            # Cleanup nếu có lỗi
            for path in tmp_csv_paths:
                try:
                    os.unlink(path)
                except:
                    pass
            raise Exception(f"Failed to convert Excel to CSV: {e}")

    async def _extract_text_with_llamaindex(self, file_path: str, extension: str) -> str:
        """Extract text using appropriate LlamaIndex reader"""
        try:
            # 1) Plain text
            if extension == "txt":
                reader = SimpleDirectoryReader(
                    input_files=[file_path],
                    file_extractor={".txt": None}
                )
                documents = reader.load_data()

            # 2) Excel -> CSV -> PandasCSVReader
            elif extension in {"xlsx", "xls"}:
                csv_paths = self._excel_to_temp_csvs(file_path)  
                reader = PandasCSVReader(concat_rows=False)
                documents = []
                try:
                    for cp in csv_paths:
                        # PandasCSVReader cần Path object
                        documents.extend(reader.load_data(file=Path(cp)))
                finally:
                    # Clean up temp CSVs
                    for cp in csv_paths:
                        try:
                            os.unlink(cp)
                        except Exception as e:
                            logger.warning(f"Failed to cleanup temp CSV {cp}: {e}")

            # 3) Other file types 
            elif extension in self.readers:
                reader = self.readers[extension]
                
                # Different readers have different APIs
                if extension == "pdf":
                    # PyMuPDFReader takes file path directly
                    documents = reader.load_data(file_path)
                elif extension == "csv":
                    # PandasCSVReader needs file= parameter
                    documents = reader.load_data(file=Path(file_path))
                elif extension == "json":
                    # JSONReader needs file= parameter  
                    documents = reader.load_data(input_file=file_path)
                else:
                    # DocxReader, PptxReader, HTMLTagReader
                    try:
                        documents = reader.load_data(file=Path(file_path))
                    except TypeError:
                        # Fallback for readers that don't use file= parameter
                        documents = reader.load_data(file_path)

            else:
                raise ValueError(f"No reader available for extension: {extension}")

            # Extract text from documents
            if not documents:
                logger.warning(f"No documents loaded from {file_path}")
                return ""

            extracted_text_parts = []
            for doc in documents:
                text = getattr(doc, 'text', None) or getattr(doc, 'content', None)
                if text and text.strip():
                    extracted_text_parts.append(text.strip())

            extracted_text = "\n\n".join(extracted_text_parts)
            
            if not extracted_text:
                logger.warning(f"No text content extracted from {file_path}")
                return ""

            logger.info(f"Extracted {len(extracted_text)} chars using reader for .{extension}")
            return extracted_text

        except Exception as e:
            logger.error(f"LlamaIndex extraction failed for {file_path}: {e}")
            raise Exception(f"Text extraction failed: {e}")

# FastAPI Application


app = FastAPI(
    title="File Processing Prototype with LlamaIndex",
    description="Upload files and extract text using LlamaIndex readers",
    version="1.0.0"
)

# Initialize processor
processor = LlamaIndexFileProcessor()

@app.post("/process-file", response_model=FileProcessingResult)
async def process_file_endpoint(file: UploadFile = File(...)):
    """
    Upload and process a file to extract text
    
    Supported formats: txt, docx, pdf, pptx, csv, xlsx, xls, json, html
    """
    try:
        result = await processor.process_uploaded_file(file)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_file_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats"""
    return {
        "supported_extensions": list(ALLOWED_EXTENSIONS),
        "total_formats": len(ALLOWED_EXTENSIONS),
        "llamaindex_readers": list(LLAMAINDEX_READERS.keys())
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "file-processing-prototype",
        "llamaindex_version": "0.9.x"
    }

@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "File Processing Prototype with LlamaIndex",
        "endpoints": {
            "process_file": "POST /process-file",
            "supported_formats": "GET /supported-formats", 
            "health": "GET /health"
        },
        "description": "Upload files to extract text using LlamaIndex readers (no chunking or vector processing)"
    }

# Main execution

if __name__ == "__main__":
    uvicorn.run(
        "llamaindex_getfile_content:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )