import logging
import os
import tempfile
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Disable auto-loading to prevent conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# LlamaIndex imports
from llama_index.core import SimpleDirectoryReader, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import (
    PyMuPDFReader,
    DocxReader,
    PptxReader,
    PandasCSVReader,
    HTMLTagReader
)
from llama_index.readers.json import JSONReader

# Project imports
from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

class LlamaIndexService:
    """
    Service trung tâm xử lý tài liệu bằng LlamaIndex.
    Kết hợp khả năng từ prototype file processing và document processing.
    
    Chức năng:
    - Đọc file đa định dạng (PDF, DOCX, TXT, CSV, HTML, etc.)
    - Chia chunks (nodes) với SentenceSplitter
    - Tạo embeddings với HuggingFace models
    - Hỗ trợ caching và batch processing
    """
    
    # Supported file extensions và reader mapping
    ALLOWED_EXTENSIONS = {
        "txt", "docx", "pdf", "pptx", "csv", 
        "xlsx", "xls", "json", "html", "htm"
    }
    
    LLAMAINDEX_READERS = {
        "pdf": PyMuPDFReader(),
        "docx": DocxReader(),
        "pptx": PptxReader(),
        "csv": PandasCSVReader(),
        "json": JSONReader(),
        "html": HTMLTagReader(),
        "htm": HTMLTagReader(),
    }
    
    def __init__(self):
        logger.info("Initializing LlamaIndexService...")
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbedding(
            model_name=settings.EMBEDDING_MODEL,
            cache_folder="./embeddings_cache",
            trust_remote_code=True  # For nomic models
        )
        
        # Configure text splitter (NodeParser)
        self.text_splitter = SentenceSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separator=" ",
            backup_separators=["\n", "\n\n", ".", "!", "?"]
        )
        
        # Thread executor for blocking operations
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Disable global LlamaIndex settings to avoid conflicts
        Settings.embed_model = None
        
        logger.info(f"LlamaIndexService initialized with model: {settings.EMBEDDING_MODEL}")
        logger.info(f"Chunk size: {settings.CHUNK_SIZE}, Overlap: {settings.CHUNK_OVERLAP}")
    
    def is_allowed_file(self, filename: str) -> bool:
        """Check if file extension is supported"""
        if not filename or '.' not in filename:
            return False
        extension = filename.rsplit('.', 1)[1].lower()
        return extension in self.ALLOWED_EXTENSIONS
    
    def get_file_extension(self, filename: str) -> str:
        """Extract file extension"""
        if not filename or '.' not in filename:
            return ""
        return filename.rsplit('.', 1)[1].lower()
    
    def load_and_chunk_file(self, file_path: Path) -> List[BaseNode]:
        """
        Đọc file, trích xuất văn bản và chia thành các chunks (nodes).
        
        Args:
            file_path: Path to the file to be processed
            
        Returns:
            List of LlamaIndex BaseNode objects with content and metadata
        """
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Get file extension
            file_extension = self.get_file_extension(str(file_path))
            
            # Load documents using appropriate reader
            documents = self._load_documents_with_reader(file_path, file_extension)
            
            if not documents:
                logger.warning(f"No content extracted from {file_path}")
                return []
            
            # Add file metadata to documents
            for doc in documents:
                doc.metadata.update({
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_extension": file_extension,
                    "source_type": "file"
                })
            
            # Split documents into nodes (chunks)
            nodes = self.text_splitter.get_nodes_from_documents(documents)
            
            # Add additional metadata to nodes
            for i, node in enumerate(nodes):
                node.metadata.update({
                    "chunk_index": i,
                    "node_id": self._generate_node_id(node.get_content(), str(file_path), i)
                })
            
            logger.info(f"Successfully processed {file_path}: {len(nodes)} nodes created")
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to load and chunk file {file_path}: {e}", exc_info=True)
            raise
    
    def _load_documents_with_reader(self, file_path: Path, file_extension: str) -> List[Document]:
        """Load documents using appropriate LlamaIndex reader"""
        try:
            # Special handling for Excel files
            if file_extension in ["xlsx", "xls"]:
                return self._handle_excel_file(file_path)
            
            # Special handling for text files
            if file_extension == "txt":
                return self._handle_text_file(file_path)
            
            # Use specific reader if available
            if file_extension in self.LLAMAINDEX_READERS:
                reader = self.LLAMAINDEX_READERS[file_extension]
                documents = reader.load_data(file=file_path)
            else:
                # Fallback to SimpleDirectoryReader
                reader = SimpleDirectoryReader(input_files=[file_path])
                documents = reader.load_data()
            
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {e}")
            raise
    
    def _handle_excel_file(self, file_path: Path) -> List[Document]:
        """Handle Excel files by converting to CSV first"""
        try:
            import pandas as pd
            
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            documents = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Convert DataFrame to text
                csv_content = df.to_string(index=False)
                
                # Create document
                doc = Document(
                    text=csv_content,
                    metadata={
                        "sheet_name": sheet_name,
                        "file_type": "excel",
                        "rows": len(df),
                        "columns": len(df.columns)
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to process Excel file {file_path}: {e}")
            # Fallback to SimpleDirectoryReader
            reader = SimpleDirectoryReader(input_files=[file_path])
            return reader.load_data()
    
    def _handle_text_file(self, file_path: Path) -> List[Document]:
        """Handle text files with proper encoding detection"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    
                    doc = Document(
                        text=content,
                        metadata={
                            "encoding": encoding,
                            "file_type": "text"
                        }
                    )
                    return [doc]
                    
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, fallback
            logger.warning(f"Could not decode {file_path} with standard encodings")
            reader = SimpleDirectoryReader(input_files=[file_path])
            return reader.load_data()
            
        except Exception as e:
            logger.error(f"Failed to process text file {file_path}: {e}")
            raise
    
    async def generate_embeddings_for_nodes(self, nodes: List[BaseNode]) -> List[List[float]]:
        """
        Tạo embeddings cho một danh sách các nodes.
        
        Args:
            nodes: List of LlamaIndex BaseNode objects
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        try:
            if not nodes:
                return []
            
            # Extract text content from nodes
            texts = [node.get_content() for node in nodes]
            
            # Generate embeddings using async method
            embeddings = await self._generate_embeddings_async(texts)
            
            logger.info(f"Generated {len(embeddings)} embeddings for {len(nodes)} nodes")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
            raise
    
    async def _generate_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings asynchronously to avoid blocking"""
        try:
            # Run embedding generation in thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                self.executor,
                self._generate_embeddings_sync,
                texts
            )
            return embeddings
            
        except Exception as e:
            logger.error(f"Async embedding generation failed: {e}")
            raise
    
    def _generate_embeddings_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous embedding generation"""
        try:
            # Use batch processing for efficiency
            if len(texts) == 1:
                # Single text
                embedding = self.embedding_model.get_text_embedding(texts[0])
                return [embedding]
            else:
                # Batch processing
                embeddings = self.embedding_model.get_text_embedding_batch(texts, show_progress=False)
                return embeddings
                
        except Exception as e:
            logger.error(f"Sync embedding generation failed: {e}")
            raise
    
    async def process_uploaded_file_complete(
        self, 
        file_content: bytes, 
        filename: str, 
        generate_embeddings: bool = True
    ) -> Dict[str, Any]:
        """
        Xử lý hoàn chỉnh file upload: trích xuất văn bản, chia chunks, tạo embeddings.
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            generate_embeddings: Whether to generate embeddings
            
        Returns:
            Dict containing processing results
        """
        try:
            # Validate file
            if not self.is_allowed_file(filename):
                raise ValueError(f"Unsupported file type: {filename}")
            
            # Create temporary file
            file_extension = self.get_file_extension(filename)
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
                
                # Process file
                nodes = self.load_and_chunk_file(Path(temp_path))
                
                result = {
                    "status": "success",
                    "filename": filename,
                    "file_size": len(file_content),
                    "file_extension": file_extension,
                    "nodes": nodes,
                    "chunk_count": len(nodes),
                    "embeddings": None
                }
                
                # Generate embeddings if requested
                if generate_embeddings and nodes:
                    embeddings = await self.generate_embeddings_for_nodes(nodes)
                    result["embeddings"] = embeddings
                    result["embedding_count"] = len(embeddings)
                
                logger.info(f"Successfully processed {filename}: {len(nodes)} chunks created")
                return result
                
            finally:
                # Cleanup temporary file
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"Failed to process uploaded file {filename}: {e}")
            return {
                "status": "error",
                "filename": filename,
                "error": str(e),
                "nodes": [],
                "chunk_count": 0
            }
    
    async def process_text_content(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        generate_embeddings: bool = True
    ) -> Dict[str, Any]:
        """
        Xử lý nội dung text trực tiếp (không qua file).
        
        Args:
            content: Text content to process
            metadata: Optional metadata to attach
            generate_embeddings: Whether to generate embeddings
            
        Returns:
            Dict containing processing results
        """
        try:
            # Create document from text
            doc_metadata = metadata or {}
            doc_metadata.update({
                "source_type": "text",
                "content_length": len(content)
            })
            
            document = Document(text=content, metadata=doc_metadata)
            
            # Split into nodes
            nodes = self.text_splitter.get_nodes_from_documents([document])
            
            # Add node metadata
            for i, node in enumerate(nodes):
                node.metadata.update({
                    "chunk_index": i,
                    "node_id": self._generate_node_id(node.get_content(), "text_content", i)
                })
            
            result = {
                "status": "success",
                "content_length": len(content),
                "nodes": nodes,
                "chunk_count": len(nodes),
                "embeddings": None
            }
            
            # Generate embeddings if requested
            if generate_embeddings and nodes:
                embeddings = await self.generate_embeddings_for_nodes(nodes)
                result["embeddings"] = embeddings
                result["embedding_count"] = len(embeddings)
            
            logger.info(f"Successfully processed text content: {len(nodes)} chunks created")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process text content: {e}")
            return {
                "status": "error",
                "error": str(e),
                "nodes": [],
                "chunk_count": 0
            }
    
    def _generate_node_id(self, content: str, source_id: str, chunk_index: int) -> str:
        """Generate unique ID for a node"""
        unique_string = f"{content[:100]}{source_id}{chunk_index}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """
        Tạo embedding cho câu hỏi/query.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding as list of floats
        """
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self.executor,
                self.embedding_model.get_text_embedding,
                query
            )
            
            logger.info(f"Generated query embedding for: '{query[:50]}...'")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get embedding model dimension"""
        return self.embedding_model.embed_dim
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        return {
            "service_name": "LlamaIndexService",
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_dimension": self.get_embedding_dimension(),
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "supported_extensions": list(self.ALLOWED_EXTENSIONS),
            "available_readers": list(self.LLAMAINDEX_READERS.keys())
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        logger.info("LlamaIndexService cleanup completed")