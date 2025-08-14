"""
LlamaIndex Document Processing Prototype
Thực hiện Document Chunking và Embedding Generation using LlamaIndex
Thay thế cho LangChain-based processing trong process_document.py
"""

import os
import tempfile
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from uuid import UUID, uuid4
import hashlib
from pathlib import Path

# Disable auto-loading models
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# LlamaIndex imports
from llama_index.core import Settings, Document
from llama_index.core.node_parser import TokenTextSplitter, SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import BaseNode

# Load .env file
load_dotenv()

# Disable embedding auto-loading
# Settings.embed_model = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# =================== PYDANTIC MODELS ===================

class DocumentMetadata(BaseModel):
    source_type: str  # "web", "file", "text", "qa", "history"
    content_type: str  # "text/plain", "application/pdf", etc.
    file_id: Optional[UUID] = None
    text_id: Optional[UUID] = None
    crawl_id: Optional[UUID] = None
    created_at: datetime = None
    full_content: Optional[str] = None
    custom_metadata: Dict[str, Any] = {}

    def __init__(self, **data):
        if 'created_at' not in data or data['created_at'] is None:
            data['created_at'] = datetime.now()
        super().__init__(**data)

class DocumentChunk(BaseModel):
    content: str
    chunk_id: str
    chunk_index: int
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = None

class ProcessingResult(BaseModel):
    status: str
    message: str
    chunks: List[DocumentChunk]
    chunk_count: int
    total_tokens: int

class EmbeddingResult(BaseModel):
    status: str
    message: str
    chunks_with_embeddings: List[DocumentChunk]
    embedding_count: int

# =================== HELPER FUNCTIONS ===================

def generate_chunk_id(content: str, source_id: str, chunk_index: int) -> str:
    """Generate a unique ID for each chunk based on content and metadata"""
    unique_string = f"{content}{source_id}{chunk_index}"
    return hashlib.md5(unique_string.encode()).hexdigest()

# =================== LLAMAINDEX PROCESSOR ===================

from .llamaindex_getfile_content import LlamaIndexFileProcessor
from app.rag.schemas import FileDataSource, WebsiteDataSource, TextDataSource
from typing import List

class LlamaIndexDocumentProcessor:
    """
    Document processor using LlamaIndex for chunking and embedding generation
    Replaces LangChain-based processing
    """
    
    def __init__(
        self, 
        chunk_size: int = CHUNK_SIZE, 
        chunk_overlap: int = CHUNK_OVERLAP,
        embedding_model: str = EMBEDDING_MODEL
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize LlamaIndex text splitter
        self.text_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" ",
            backup_separators=["\n"]
        )
        
        # Initialize node parser with text splitter
        self.node_parser = self.text_splitter
        
        # Initialize embedding model
        self.embedding_model = OpenAIEmbedding(
            model=embedding_model,
            dimensions=EMBEDDING_DIMENSION,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # ADD: File processor integration
        self.file_processor = LlamaIndexFileProcessor()
        
        logger.info(f"Initialized LlamaIndex processor with chunk_size={chunk_size}, overlap={chunk_overlap}")

    def _create_document_metadata(
        self,
        source_type: str,
        content_type: str,
        source_id: Optional[str] = None,
        **kwargs
    ) -> DocumentMetadata:
        """Create DocumentMetadata object"""
        metadata_dict = {
            "source_type": source_type,
            "content_type": content_type,
            **kwargs
        }
        
        # Set appropriate ID field based on source type
        if source_type == "web" and "crawl_id" in kwargs:
            metadata_dict["crawl_id"] = kwargs["crawl_id"]
        elif source_type == "file" and "file_id" in kwargs:
            metadata_dict["file_id"] = kwargs["file_id"]
        elif source_type == "text" and "text_id" in kwargs:
            metadata_dict["text_id"] = kwargs["text_id"]
            
        return DocumentMetadata(**metadata_dict)

    async def process_file_content(
        self,
        file_name: str,
        file_content: Union[bytes, str],
        file_type: str,
        file_id: Optional[UUID] = None,
        generate_embeddings: bool = True
    ) -> ProcessingResult:
        """Process file content into chunks using LlamaIndex"""
        
        try:
            # Convert bytes to string if necessary
            if isinstance(file_content, bytes):
                content = file_content.decode("utf-8", errors="ignore")
                logger.info(f"Decoded bytes to string, length: {len(content)}")
            else:
                content = file_content
                logger.info(f"Content is already string, length: {len(content)}")

            if len(content) == 0:
                logger.warning("Empty content received")
                return ProcessingResult(
                    status="error",
                    message="Empty content",
                    chunks=[],
                    chunk_count=0,
                    total_tokens=0
                )

            # Create LlamaIndex Document
            document = Document(
                text=content,
                metadata={
                    "file_name": file_name,
                    "file_type": file_type,
                    "file_id": str(file_id) if file_id else None,
                    "source_type": "file"
                }
            )

            # Parse into nodes (chunks)
            nodes = self.node_parser.get_nodes_from_documents([document])
            logger.info(f"Split content into {len(nodes)} chunks")

            # Create DocumentChunk objects
            chunks = []
            total_tokens = 0
            
            for idx, node in enumerate(nodes):
                # Create metadata
                doc_metadata = self._create_document_metadata(
                    source_type="file",
                    content_type=file_type,
                    file_id=file_id,
                    full_content=content
                )
                
                # Generate chunk
                chunk = DocumentChunk(
                    content=node.get_content(),
                    chunk_id=generate_chunk_id(node.get_content(), file_name, idx),
                    chunk_index=idx,
                    metadata=doc_metadata
                )
                
                # Generate embedding if requested
                if generate_embeddings:
                    embedding = await self.embedding_model.aget_text_embedding(node.get_content())
                    chunk.embedding = embedding
                
                chunks.append(chunk)
                # Estimate tokens (rough calculation)
                total_tokens += len(node.get_content().split())

            return ProcessingResult(
                status="success",
                message=f"Successfully processed {file_name} into {len(chunks)} chunks",
                chunks=chunks,
                chunk_count=len(chunks),
                total_tokens=total_tokens
            )

        except Exception as e:
            logger.error(f"Error processing file content: {str(e)}")
            return ProcessingResult(
                status="error",
                message=f"Error processing file: {str(e)}",
                chunks=[],
                chunk_count=0,
                total_tokens=0
            )

    async def process_website_content(
        self,
        url: str,
        content: str,
        crawl_id: Optional[UUID] = None,
        generate_embeddings: bool = True
    ) -> ProcessingResult:
        """Process website content into chunks using LlamaIndex"""
        
        try:
            # Create LlamaIndex Document
            document = Document(
                text=content,
                metadata={
                    "url": url,
                    "crawl_id": str(crawl_id) if crawl_id else None,
                    "source_type": "web"
                }
            )

            # Parse into nodes (chunks)
            nodes = self.node_parser.get_nodes_from_documents([document])
            logger.info(f"Split website content into {len(nodes)} chunks")

            # Create DocumentChunk objects
            chunks = []
            total_tokens = 0
            
            for idx, node in enumerate(nodes):
                # Create metadata
                doc_metadata = self._create_document_metadata(
                    source_type="web",
                    content_type="text/html",
                    crawl_id=crawl_id,
                    full_content=content
                )
                
                # Generate chunk
                chunk = DocumentChunk(
                    content=node.get_content(),
                    chunk_id=generate_chunk_id(node.get_content(), url, idx),
                    chunk_index=idx,
                    metadata=doc_metadata
                )
                
                # Generate embedding if requested
                if generate_embeddings:
                    embedding = await self.embedding_model.aget_text_embedding(node.get_content())
                    chunk.embedding = embedding
                
                chunks.append(chunk)
                total_tokens += len(node.get_content().split())

            return ProcessingResult(
                status="success",
                message=f"Successfully processed website {url} into {len(chunks)} chunks",
                chunks=chunks,
                chunk_count=len(chunks),
                total_tokens=total_tokens
            )

        except Exception as e:
            logger.error(f"Error processing website content: {str(e)}")
            return ProcessingResult(
                status="error",
                message=f"Error processing website: {str(e)}",
                chunks=[],
                chunk_count=0,
                total_tokens=0
            )

    async def process_text_content(
        self,
        content: str,
        text_id: Optional[UUID] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
        generate_embeddings: bool = True
    ) -> ProcessingResult:
        """Process direct text input into chunks using LlamaIndex"""
        
        try:
            # Create LlamaIndex Document
            document = Document(
                text=content,
                metadata={
                    "text_id": str(text_id) if text_id else None,
                    "source_type": "text",
                    **(custom_metadata or {})
                }
            )

            # Parse into nodes (chunks)
            nodes = self.node_parser.get_nodes_from_documents([document])
            logger.info(f"Split text content into {len(nodes)} chunks")

            # Create DocumentChunk objects
            chunks = []
            total_tokens = 0
            
            for idx, node in enumerate(nodes):
                # Create metadata
                doc_metadata = self._create_document_metadata(
                    source_type="text",
                    content_type="text/plain",
                    text_id=text_id,
                    full_content=content,
                    custom_metadata=custom_metadata or {}
                )
                
                # Generate chunk
                chunk = DocumentChunk(
                    content=node.get_content(),
                    chunk_id=generate_chunk_id(node.get_content(), "text", idx),
                    chunk_index=idx,
                    metadata=doc_metadata
                )
                
                # Generate embedding if requested
                if generate_embeddings:
                    embedding = await self.embedding_model.aget_text_embedding(node.get_content())
                    chunk.embedding = embedding
                
                chunks.append(chunk)
                total_tokens += len(node.get_content().split())

            return ProcessingResult(
                status="success",
                message=f"Successfully processed text into {len(chunks)} chunks",
                chunks=chunks,
                chunk_count=len(chunks),
                total_tokens=total_tokens
            )

        except Exception as e:
            logger.error(f"Error processing text content: {str(e)}")
            return ProcessingResult(
                status="error",
                message=f"Error processing text: {str(e)}",
                chunks=[],
                chunk_count=0,
                total_tokens=0
            )

    async def process_qa_content(
        self,
        question: str,
        answer: str,
        training_id: str,
        text_id: Optional[UUID] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
        generate_embeddings: bool = True
    ) -> ProcessingResult:
        """
        Process QA pairs - no chunking, preserve semantic relationship
        Store complete question as single chunk
        """
        
        try:
            content = question.strip()
            
            # Prepare QA metadata
            qa_metadata = {
                "training_id": training_id,
                "question": question,
                "answer": answer,
                **(custom_metadata or {})
            }
            
            # Create metadata
            doc_metadata = self._create_document_metadata(
                source_type="qa",
                content_type="text/plain",
                text_id=text_id,
                custom_metadata=qa_metadata
            )
            
            # Create single chunk (no splitting for QA)
            chunk = DocumentChunk(
                content=content,
                chunk_id=generate_chunk_id(content, training_id, 0),
                chunk_index=0,
                metadata=doc_metadata
            )
            
            # Generate embedding if requested
            if generate_embeddings:
                embedding = await self.embedding_model.aget_text_embedding(content)
                chunk.embedding = embedding
            
            return ProcessingResult(
                status="success",
                message=f"Successfully processed QA pair with training_id {training_id}",
                chunks=[chunk],
                chunk_count=1,
                total_tokens=len(content.split())
            )

        except Exception as e:
            logger.error(f"Error processing QA content: {str(e)}")
            return ProcessingResult(
                status="error",
                message=f"Error processing QA: {str(e)}",
                chunks=[],
                chunk_count=0,
                total_tokens=0
            )

    async def generate_embeddings_for_chunks(
        self, 
        chunks: List[DocumentChunk]
    ) -> EmbeddingResult:
        """Generate embeddings for existing chunks"""
        
        try:
            chunks_with_embeddings = []
            
            for chunk in chunks:
                if chunk.embedding is None:
                    # Generate embedding
                    embedding = await self.embedding_model.aget_text_embedding(chunk.content)
                    chunk.embedding = embedding
                
                chunks_with_embeddings.append(chunk)
            
            return EmbeddingResult(
                status="success",
                message=f"Successfully generated embeddings for {len(chunks)} chunks",
                chunks_with_embeddings=chunks_with_embeddings,
                embedding_count=len(chunks_with_embeddings)
            )

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return EmbeddingResult(
                status="error",
                message=f"Error generating embeddings: {str(e)}",
                chunks_with_embeddings=[],
                embedding_count=0
            )

    # MODIFY: Existing method để support multiple sources
    async def process_multiple_files(
        self,
        files: List[FileDataSource],
        generate_embeddings: bool = True
    ) -> List[ProcessingResult]:
        """Process multiple files - compatible với RAG system"""
        results = []
        
        for file_data in files:
            # Convert FileDataSource to UploadFile-like object
            if isinstance(file_data.file_content, str):
                content = file_data.file_content.encode('utf-8')
            else:
                content = file_data.file_content
                
            # Create mock UploadFile
            from fastapi import UploadFile
            from io import BytesIO
            
            mock_file = UploadFile(
                filename=file_data.file_name,
                file=BytesIO(content),
                content_type=file_data.file_type
            )
            
            # Process through complete pipeline
            result = await self.process_uploaded_file_complete(
                file=mock_file,
                generate_embeddings=generate_embeddings
            )
            results.append(result)
            
        return results

    async def process_uploaded_file_complete(
        self,
        file: UploadFile,
        generate_embeddings: bool = True
    ) -> ProcessingResult:
        """COMPLETE PIPELINE: Extract → Chunk → Embed"""
        try:
            logger.info(f"🔄 Starting complete pipeline for {file.filename}")
            
            # STEP 1: Extract text
            extraction_result = await self.file_processor.process_uploaded_file(file)
            
            if extraction_result.status != "success":
                return ProcessingResult(
                    status="error",
                    message=f"Text extraction failed: {extraction_result.message}",
                    chunks=[],
                    chunk_count=0,
                    total_tokens=0
                )
            
            logger.info(f"✅ Step 1: Extracted {extraction_result.text_length} characters")
            
            # STEP 2: Process into chunks
            processing_result = await self.process_file_content(
                file_name=file.filename,
                file_content=extraction_result.extracted_text,
                file_type=extraction_result.content_type,
                file_id=uuid4(),
                generate_embeddings=generate_embeddings
            )
            
            # Enrich metadata với extraction info
            for chunk in processing_result.chunks:
                chunk.metadata.custom_metadata.update({
                    "extraction_method": "llamaindex_readers",
                    "original_file_size": extraction_result.file_size,
                    "extraction_status": extraction_result.status
                })
            
            logger.info(f"✅ Complete pipeline: {processing_result.chunk_count} chunks created")
            return processing_result
            
        except Exception as e:
            logger.error(f"❌ Pipeline error: {str(e)}")
            return ProcessingResult(
                status="error",
                message=f"Pipeline failed: {str(e)}",
                chunks=[],
                chunk_count=0,
                total_tokens=0
            )

# =================== FASTAPI APPLICATION ===================

app = FastAPI(
    title="LlamaIndex Document Processing Prototype",
    description="Document chunking and embedding generation using LlamaIndex",
    version="1.0.0"
)

# Initialize processor
processor = LlamaIndexDocumentProcessor()

@app.get("/")
async def root():
    return {
        "message": "LlamaIndex Document Processing Prototype",
        "features": [
            "Document chunking using LlamaIndex TokenTextSplitter",
            "Embedding generation using OpenAI text-embedding-3-small",
            "Support for file, website, text, and QA content processing"
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "processor": "LlamaIndex", "embedding_model": EMBEDDING_MODEL}

@app.post("/process_file") 
async def process_file_endpoint(
    file: UploadFile = File(...),
    generate_embeddings: bool = True
):
    # SỬ DỤNG complete pipeline thay vì đọc raw bytes
    result = await processor.process_uploaded_file_complete(
        file=file,
        generate_embeddings=generate_embeddings
    )
    return result

@app.post("/process_text")
async def process_text_endpoint(
    content: str,
    generate_embeddings: bool = True,
    custom_metadata: Optional[Dict[str, Any]] = None
):
    """Process text content into chunks with optional embeddings"""
    
    result = await processor.process_text_content(
        content=content,
        text_id=uuid4(),
        custom_metadata=custom_metadata,
        generate_embeddings=generate_embeddings
    )
    
    return result

@app.post("/process_qa")
async def process_qa_endpoint(
    question: str,
    answer: str,
    training_id: str,
    generate_embeddings: bool = True,
    custom_metadata: Optional[Dict[str, Any]] = None
):
    """Process QA pair (no chunking) with optional embeddings"""
    
    result = await processor.process_qa_content(
        question=question,
        answer=answer,
        training_id=training_id,
        text_id=uuid4(),
        custom_metadata=custom_metadata,
        generate_embeddings=generate_embeddings
    )
    
    return result

@app.post("/generate_embeddings")
async def generate_embeddings_endpoint(chunks: List[DocumentChunk]):
    """Generate embeddings for existing chunks"""
    
    result = await processor.generate_embeddings_for_chunks(chunks)
    return result

# ADD: Compatibility endpoints với RAG system
@app.post("/process_files_rag_compatible")
async def process_files_rag_compatible(
    files: List[FileDataSource],
    generate_embeddings: bool = True
):
    """
    RAG-compatible endpoint - input format giống app/rag/routes.py
    """
    results = await processor.process_multiple_files(
        files=files,
        generate_embeddings=generate_embeddings
    )
    
    # Format response giống RAG system
    all_chunk_ids = []
    total_chunks = 0
    errors = []
    
    for result in results:
        if result.status == "success":
            chunk_ids = [chunk.chunk_id for chunk in result.chunks]
            all_chunk_ids.extend(chunk_ids)
            total_chunks += result.chunk_count
        else:
            errors.append({
                "message": result.message,
                "status": result.status
            })
    
    return {
        "status": "success" if not errors else "partial_success",
        "chunk_ids": all_chunk_ids,
        "total_chunks": total_chunks,
        "message": f"Processed {total_chunks} chunks using LlamaIndex",
        "errors": errors if errors else None,
        "method": "llamaindex_complete_pipeline"
    }

