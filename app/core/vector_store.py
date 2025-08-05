# app/core/vector_store.py
import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

class FAISSVectorStore:
    """
    FAISS-based vector store với persistence
    """
    
    def __init__(self, dimension: int, index_path: str = "vector_indexes"):
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.index_path.mkdir(exist_ok=True)
        
        self.index = None
        self.metadata = []  # Store chunk metadata
        self.document_mapping = {}  # Map vector_id -> document_id
        
        self.logger = logging.getLogger(__name__)
    
    def create_index(self, index_type: str = "Flat") -> None:
        """Create FAISS index"""
        if index_type == "Flat":
            # Exact search (good for small datasets)
            self.index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "IVF":
            # Approximate search (good for large datasets)
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)  # 100 clusters
        
        self.logger.info(f"Created FAISS index: {index_type}, dimension: {self.dimension}")
    
    def add_vectors(
        self, 
        vectors: np.ndarray, 
        metadata: List[Dict],
        document_id: int
    ) -> List[int]:
        """Add vectors to index"""
        if self.index is None:
            self.create_index()
        
        # Normalize vectors (for cosine similarity)
        vectors_normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Add to FAISS index
        start_id = self.index.ntotal
        self.index.add(vectors_normalized.astype('float32'))
        
        # Store metadata
        vector_ids = list(range(start_id, self.index.ntotal))
        for i, meta in enumerate(metadata):
            self.metadata.append(meta)
            self.document_mapping[start_id + i] = document_id
        
        self.logger.info(f"Added {len(vectors)} vectors for document {document_id}")
        return vector_ids
    
    def search(
        self, 
        query_vector: np.ndarray, 
        k: int = 5,
        document_ids: Optional[List[int]] = None
    ) -> List[Dict]:
        """Search for similar vectors"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Normalize query vector
        query_normalized = query_vector / np.linalg.norm(query_vector)
        query_normalized = query_normalized.reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_normalized, k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # No more results
                break
            
            # Filter by document_ids if specified
            if document_ids and self.document_mapping.get(idx) not in document_ids:
                continue
            
            result = {
                'vector_id': int(idx),
                'distance': float(distance),
                'similarity': 1 / (1 + distance),  # Convert distance to similarity
                'metadata': self.metadata[idx],
                'document_id': self.document_mapping.get(idx)
            }
            results.append(result)
        
        return results
    
    def save_index(self, filename: str = "faiss_index") -> bool:
        """Save index và metadata"""
        try:
            # Save FAISS index
            index_file = self.index_path / f"{filename}.index"
            faiss.write_index(self.index, str(index_file))
            
            # Save metadata
            metadata_file = self.index_path / f"{filename}_metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'document_mapping': self.document_mapping,
                    'dimension': self.dimension
                }, f)
            
            self.logger.info(f"Saved vector index to {index_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")
            return False
    
    def load_index(self, filename: str = "faiss_index") -> bool:
        """Load index và metadata"""
        try:
            # Load FAISS index
            index_file = self.index_path / f"{filename}.index"
            if not index_file.exists():
                self.logger.warning(f"Index file not found: {index_file}")
                return False
            
            self.index = faiss.read_index(str(index_file))
            
            # Load metadata
            metadata_file = self.index_path / f"{filename}_metadata.pkl"
            with open(metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.document_mapping = data['document_mapping']
                self.dimension = data['dimension']
            
            self.logger.info(f"Loaded vector index from {index_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load index: {e}")
            return False