from typing import List, Optional, Dict, Any
import numpy as np
import logging
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.core.config import settings
from app.core.vector_store import FAISSVectorStore
from app.models.document import Document, DocumentChunk

logger = logging.getLogger(__name__)

class VectorService:
    """
    Vector service để quản lý FAISS vector store
    Refactored để nhận embeddings từ LlamaIndexService thay vì tự tạo
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.logger = logger  # ✅ THÊM: Add logger instance
        
        # Khởi tạo FAISS vector store
        dimension = settings.EMBEDDING_DIMENSION if hasattr(settings, 'EMBEDDING_DIMENSION') else 768
        self.vector_store = FAISSVectorStore(dimension=dimension)
        
        # Load existing index if available
        self._load_existing_index()
        self.logger.info("VectorService initialized (without EmbeddingService)")

    def _load_existing_index(self):
        """Load existing FAISS index from disk"""
        try:
            index_path = Path("vector_indexes")
            if index_path.exists():
                self.vector_store.load_index()
                self.logger.info("Loaded existing FAISS index")
            else:
                self.logger.info("No existing index found, will create new one")
        except Exception as e:
            self.logger.warning(f"Failed to load existing index: {e}")

    async def add_chunks_to_index(
        self, 
        document_id: int, 
        chunk_ids: List[int], 
        chunk_texts: List[str], 
        embeddings: List[List[float]]
    ):
        """
        Thêm các chunks với embeddings đã tạo sẵn vào index.
        
        Args:
            document_id: ID của document
            chunk_ids: List các chunk ID
            chunk_texts: List nội dung text của chunks
            embeddings: List embeddings đã được tạo sẵn
        """
        try:
            if not embeddings or len(embeddings) != len(chunk_ids):
                raise ValueError(f"Embedding count ({len(embeddings)}) doesn't match chunk count ({len(chunk_ids)})")

            self.logger.info(f"Adding {len(chunk_ids)} chunks to vector index for document {document_id}")

            # Convert embeddings to numpy array
            vectors = np.array(embeddings, dtype=np.float32)
            
            # Create metadata for each chunk
            metadata = []
            for i, (chunk_id, text) in enumerate(zip(chunk_ids, chunk_texts)):
                metadata.append({
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "text": text[:500],  # Limit text length in metadata
                    "chunk_index": i
                })

            # Add vectors to store
            self.vector_store.add_vectors(vectors, metadata, document_id)
            
            # Save index to disk
            self.vector_store.save_index()
            
            self.logger.info(f"Successfully added {len(vectors)} vectors for document {document_id}")

        except Exception as e:
            self.logger.error(f"Failed to add chunks to index for document {document_id}: {e}")
            raise

    async def search_similar_chunks(
        self, 
        query_embedding: List[float], 
        top_k: int = 5,
        document_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm chunks tương tự dựa trên query embedding đã được tạo sẵn.
        
        Args:
            query_embedding: Embedding của query (đã được tạo sẵn)
            top_k: Số lượng kết quả trả về
            document_ids: Filter theo document IDs (optional)
            
        Returns:
            List các chunks tương tự với scores và metadata
        """
        try:
            if not query_embedding:
                return []

            # Convert query to numpy array
            query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            
            # Search in vector store
            results = self.vector_store.search(query_vector, top_k)
            
            if not results:
                return []

            # Process results
            similar_chunks = []
            for score, metadata in results:
                chunk_id = metadata.get("chunk_id")
                if not chunk_id:
                    continue

                # Get full chunk data from database
                chunk = self.db.query(DocumentChunk).filter(
                    DocumentChunk.id == chunk_id
                ).first()
                
                if not chunk:
                    continue

                # Filter by document IDs if specified
                if document_ids and chunk.document_id not in document_ids:
                    continue

                # Get document info
                document = self.db.query(Document).filter(
                    Document.id == chunk.document_id
                ).first()

                chunk_data = {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "similarity_score": float(score),
                    "document_filename": document.filename if document else "Unknown",
                    "metadata": metadata
                }
                similar_chunks.append(chunk_data)

            self.logger.info(f"Found {len(similar_chunks)} similar chunks")
            return similar_chunks

        except Exception as e:
            self.logger.error(f"Failed to search similar chunks: {e}")
            return []

    async def remove_document_from_index(self, document_id: int):
        """Xóa tất cả vectors của một document khỏi index"""
        try:
            removed_count = self.vector_store.remove_by_document_id(document_id)
            
            if removed_count > 0:
                self.vector_store.save_index()
                self.logger.info(f"Removed {removed_count} vectors for document {document_id}")
            else:
                self.logger.warning(f"No vectors found for document {document_id}")

        except Exception as e:
            self.logger.error(f"Failed to remove document {document_id} from index: {e}")
            raise

    def get_index_stats(self) -> Dict[str, Any]:
        """Lấy thống kê về vector index"""
        try:
            stats = self.vector_store.get_stats()
            return {
                "total_vectors": stats.get("total_vectors", 0),
                "dimension": stats.get("dimension", 0),
                "index_size_mb": stats.get("index_size_mb", 0),
                "documents_indexed": len(stats.get("document_ids", [])),
                "backend": "FAISS"
            }
        except Exception as e:
            self.logger.error(f"Failed to get index stats: {e}")
            return {"error": str(e)}

    async def rebuild_index(self):
        """Rebuild toàn bộ vector index từ database"""
        try:
            self.logger.info("Starting index rebuild...")
            
            # Clear existing index
            self.vector_store.clear_index()
            
            # Get all completed documents
            documents = self.db.query(Document).filter(
                Document.processing_status == "completed"
            ).all()
            
            total_processed = 0
            
            for document in documents:
                try:
                    # Get chunks for this document
                    chunks = self.db.query(DocumentChunk).filter(
                        DocumentChunk.document_id == document.id
                    ).order_by(DocumentChunk.chunk_index).all()
                    
                    if not chunks:
                        continue
                    
                    # Note: Để rebuild index, chúng ta cần embeddings
                    # Hiện tại chỉ log warning vì không có embeddings sẵn
                    self.logger.warning(f"Cannot rebuild embeddings for document {document.id} - embeddings not stored")
                    
                except Exception as e:
                    self.logger.error(f"Failed to process document {document.id} during rebuild: {e}")
                    continue
            
            self.vector_store.save_index()
            self.logger.info(f"Index rebuild completed. Processed {total_processed} documents")
            
        except Exception as e:
            self.logger.error(f"Failed to rebuild index: {e}")
            raise

    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self.vector_store, 'cleanup'):
                self.vector_store.cleanup()
            self.logger.info("VectorService cleanup completed")
        except Exception as e:
            self.logger.warning(f"Error during VectorService cleanup: {e}")