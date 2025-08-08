from typing import List, Optional, Dict, Any
import numpy as np
import logging
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.core.config import settings
from app.core.vector_store import FAISSVectorStore
from app.services.embedding_service import EmbeddingService
from app.models.document import Document, DocumentChunk

# ✅ SỬA: Move function outside class và add missing import
import re

def _keyword_overlap_score(query: str, text: str) -> float:
    """Helper function for keyword overlap calculation"""
    q = set(re.findall(r'\w+', query.lower()))
    t = set(re.findall(r'\w+', text.lower()))
    return (len(q & t) / len(q)) if q else 0.0

class VectorService:
    """
    High-level vector service for document indexing and search
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.embedding_service = EmbeddingService()
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize FAISS vector store"""
        try:
            dimension = self.embedding_service.get_dimension()
            self.vector_store = FAISSVectorStore(
                dimension=dimension,
                index_path="vector_indexes"
            )
            
            # ✅ THÊM: Debug file existence
            from pathlib import Path
            index_file = Path("vector_indexes/faiss_index.index")
            metadata_file = Path("vector_indexes/faiss_index_metadata.pkl")
            
            self.logger.info(f"🔍 Working directory: {Path.cwd()}")
            self.logger.info(f"📁 Index file exists: {index_file.exists()}")
            self.logger.info(f"📁 Metadata file exists: {metadata_file.exists()}")
            
            # Try to load existing index
            if self.vector_store.load_index("faiss_index"):
                self.logger.info(f"📚 Loaded existing vector index with {self.vector_store.index.ntotal} vectors")
            else:
                self.logger.info("🆕 Failed to load existing index or no index found")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize vector store: {e}")
            raise
    
    async def index_document_chunks(self, document_id: int, db: Session) -> Dict[str, Any]:
        """
        Index all chunks of a document into vector store
        """
        try:
            # Get document and its chunks
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            chunks = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).order_by(DocumentChunk.chunk_index).all()
            
            if not chunks:
                raise ValueError(f"No chunks found for document {document_id}")
            
            self.logger.info(f"🔄 Indexing {len(chunks)} chunks for document {document_id}")
            
            # Extract text content from chunks
            chunk_texts = [chunk.chunk_text for chunk in chunks]
            
            # Generate embeddings for all chunks
            embeddings = await self.embedding_service.batch_text_to_embeddings(chunk_texts)
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Prepare metadata for each chunk
            metadata = []
            for chunk in chunks:
                meta = {
                    'chunk_id': chunk.id,
                    'document_id': chunk.document_id,
                    'chunk_index': chunk.chunk_index,
                    'page_number': chunk.page_number,
                    'text_length': len(chunk.chunk_text),
                    'chunk_text': chunk.chunk_text  # Store full text for retrieval
                }
                metadata.append(meta)
            
            # Add vectors to FAISS index
            vector_ids = self.vector_store.add_vectors(
                vectors=embeddings_array,
                metadata=metadata,
                document_id=document_id
            )
            
            # Save index to disk
            self.vector_store.save_index()
            
            result = {
                'document_id': document_id,
                'chunks_indexed': len(chunks),
                'vector_ids': vector_ids,
                'success': True
            }
            
            self.logger.info(f"✅ Successfully indexed {len(chunks)} chunks for document {document_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to index document {document_id}: {e}")
            return {
                'document_id': document_id,
                'success': False,
                'error': str(e)
            }
    
    async def search_similar_chunks(
        self,
        query: str,
        k: int = 5,
        document_ids=None,
        similarity_threshold: float | None = None
    ):
        if not self.vector_store or not self.vector_store.index:
            self.logger.warning("No vector index available for search")
            return []

        try:
            emb = await self.embedding_service.text_to_embedding(query)
            # lấy nhiều ứng viên hơn để re-rank
            candidates = self.vector_store.search(
                np.array(emb, dtype=np.float32), 
                k=max(k * 10, 30),
                document_ids=document_ids
            )
            
            if not candidates:
                self.logger.info("🔍 Raw search returned 0 results")
                return []

            top = candidates[0]['similarity']
            base_thr = settings.SIMILARITY_THRESHOLD if similarity_threshold is None else similarity_threshold
            dyn_thr = max(base_thr, top - 0.15)

            filtered = [r for r in candidates if r['similarity'] >= dyn_thr]

            # ✅ SỬA: Use correct function name
            for r in filtered:
                txt = r['metadata'].get('chunk_text', '')
                r['rerank'] = r['similarity'] + 0.2 * _keyword_overlap_score(query, txt)

            filtered.sort(key=lambda x: x['rerank'], reverse=True)

            results = []
            for r in filtered[:k]:
                m = r['metadata']
                results.append({
                    "chunk_id": m.get("chunk_id"),
                    "document_id": r.get("document_id") or m.get("document_id"),
                    "chunk_index": m.get("chunk_index"),
                    "page_number": m.get("page_number"),
                    "similarity_score": r["similarity"],
                    "chunk_text": m.get("chunk_text", "")
                })

            self.logger.info(f"🔍 Search query: '{query[:50]}...' returned {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Search failed: {e}")
            return []
    
    async def reindex_all_documents(self, db: Session) -> Dict[str, Any]:
        """
        Reindex all processed documents
        """
        try:
            # Get all processed documents
            documents = db.query(Document).filter(
                and_(
                    Document.processed == True,
                    Document.chunk_count > 0
                )
            ).all()
            
            if not documents:
                return {
                    'success': True,
                    'message': 'No documents to reindex',
                    'documents_processed': 0
                }
            
            self.logger.info(f"🔄 Reindexing {len(documents)} documents...")
            
            # Clear existing index
            self.vector_store = FAISSVectorStore(
                dimension=self.embedding_service.get_dimension(),
                index_path="vector_indexes"
            )
            
            success_count = 0
            failed_documents = []
            
            for document in documents:
                try:
                    result = await self.index_document_chunks(document.id, db)
                    if result['success']:
                        success_count += 1
                    else:
                        failed_documents.append(document.id)
                except Exception as e:
                    self.logger.error(f"Failed to reindex document {document.id}: {e}")
                    failed_documents.append(document.id)
            
            return {
                'success': True,
                'documents_processed': success_count,
                'failed_documents': failed_documents,
                'total_documents': len(documents)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Reindexing failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def delete_document_vectors(self, document_id: int) -> bool:
        """
        Delete vectors for a specific document
        Note: FAISS doesn't support efficient deletion, so we recreate index
        """
        try:
            if not self.vector_store or not self.vector_store.index:
                return True  # Nothing to delete
            
            # Find vectors to keep (all except for this document)
            vectors_to_keep = []
            metadata_to_keep = []
            doc_mapping_to_keep = {}
            
            for i, meta in enumerate(self.vector_store.metadata):
                if meta.get('document_id') != document_id:
                    # Keep this vector
                    vector = self.vector_store.index.reconstruct(i)
                    vectors_to_keep.append(vector)
                    metadata_to_keep.append(meta)
                    doc_mapping_to_keep[len(vectors_to_keep) - 1] = meta.get('document_id')
            
            if vectors_to_keep:
                # Recreate index with remaining vectors
                self.vector_store = FAISSVectorStore(
                    dimension=self.embedding_service.get_dimension(),
                    index_path="vector_indexes"
                )
                
                vectors_array = np.array(vectors_to_keep, dtype=np.float32)
                # Group by document_id for proper addition
                doc_groups = {}
                for i, meta in enumerate(metadata_to_keep):
                    doc_id = meta['document_id']
                    if doc_id not in doc_groups:
                        doc_groups[doc_id] = {'vectors': [], 'metadata': []}
                    doc_groups[doc_id]['vectors'].append(vectors_array[i])
                    doc_groups[doc_id]['metadata'].append(meta)
                
                # Add vectors back by document groups
                for doc_id, group in doc_groups.items():
                    group_vectors = np.array(group['vectors'])
                    self.vector_store.add_vectors(
                        vectors=group_vectors,
                        metadata=group['metadata'],
                        document_id=doc_id
                    )
                
                # Save updated index
                self.vector_store.save_index()
            else:
                # No vectors left, create empty index
                self.vector_store = FAISSVectorStore(
                    dimension=self.embedding_service.get_dimension(),
                    index_path="vector_indexes"
                )
            
            self.logger.info(f"🗑️  Deleted vectors for document {document_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to delete vectors for document {document_id}: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector index
        """
        try:
            if not self.vector_store or not self.vector_store.index:
                return {
                    'total_vectors': 0,
                    'index_dimension': self.embedding_service.get_dimension(),
                    'documents_indexed': 0,
                    'index_type': 'empty'
                }
            
            # Count unique documents
            unique_docs = set()
            for meta in self.vector_store.metadata:
                if 'document_id' in meta:
                    unique_docs.add(meta['document_id'])
            
            return {
                'total_vectors': self.vector_store.index.ntotal,
                'index_dimension': self.vector_store.dimension,
                'documents_indexed': len(unique_docs),
                'index_type': type(self.vector_store.index).__name__,
                'embedding_model': self.embedding_service.get_model_info()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get index stats: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """
        Cleanup resources
        """
        if hasattr(self, 'embedding_service'):
            self.embedding_service.cleanup()