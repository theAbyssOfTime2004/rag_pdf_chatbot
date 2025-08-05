# app/services/embedding_service.py
from typing import List, Optional, Dict
import numpy as np
import hashlib
import pickle
import asyncio
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings

# Try to import Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class EmbeddingService:
    """
    Open-source embedding service using Nomic Embed Text v1
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache_dir = Path("embeddings_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Model initialization
        self.embedding_model = None
        self.embedding_dimension = None
        self.model_name = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize embedding model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize open-source embedding model"""
        model_name = getattr(settings, 'EMBEDDING_MODEL', 'nomic-ai/nomic-embed-text-v1')
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            # Use Sentence Transformers (including Nomic model)
            self._setup_sentence_transformers(model_name)
        else:
            # Fallback to simple embeddings
            self._setup_fallback_embeddings()
            self.logger.warning("⚠️  sentence-transformers not available. Using fallback embeddings.")
    
    def _setup_sentence_transformers(self, model_name: str):
        """Setup Sentence Transformers (including Nomic model)"""
        try:
            self.model_name = model_name
            
            self.logger.info(f"🔄 Loading open-source model: {model_name}")
            
            # Special handling for Nomic model
            if 'nomic-ai' in model_name:
                self.logger.info("🧠 Loading Nomic Embed Text v1 model...")
                self.embedding_model = SentenceTransformer(
                    model_name, 
                    trust_remote_code=True  # Required for Nomic model
                )
                # Nomic model dimension is 768
                self.embedding_dimension = 768
                self.logger.info("✅ Nomic Embed Text v1 loaded successfully!")
            else:
                # Standard Sentence Transformers models
                self.embedding_model = SentenceTransformer(model_name)
                self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            
            self.logger.info(f"✅ Initialized open-source embeddings: {model_name}, dimension: {self.embedding_dimension}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup Sentence Transformers: {e}")
            self._setup_fallback_embeddings()
    
    def _setup_fallback_embeddings(self):
        """Fallback to simple hash-based embeddings"""
        self.model_name = 'fallback'
        self.embedding_dimension = 384  # Standard dimension
        self.logger.warning("⚠️  Using fallback embedding method - install sentence-transformers for better results")
    
    async def text_to_embedding(self, text: str) -> List[float]:
        """Convert single text to embedding vector"""
        if not text or not text.strip():
            return [0.0] * self.embedding_dimension
        
        # Check cache first
        if getattr(settings, 'ENABLE_EMBEDDING_CACHE', True):
            cache_key = self._get_cache_key(text)
            cached_embedding = self._load_from_cache(cache_key)
            if cached_embedding is not None:
                return cached_embedding
        
        # Generate embedding
        try:
            if self.embedding_model is not None:
                # Use Sentence Transformers
                embedding = await self._sentence_transformer_embedding(text)
            else:
                # Use fallback method
                embedding = self._fallback_embedding(text)
            
            # Cache result
            if getattr(settings, 'ENABLE_EMBEDDING_CACHE', True):
                self._save_to_cache(cache_key, embedding)
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate embedding: {e}")
            return [0.0] * self.embedding_dimension
    
    async def batch_text_to_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Convert batch of texts to embeddings (more efficient)"""
        if not texts:
            return []
        
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text (if caching enabled)
        if getattr(settings, 'ENABLE_EMBEDDING_CACHE', True):
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    embeddings.append([0.0] * self.embedding_dimension)
                    continue
                    
                cache_key = self._get_cache_key(text)
                cached = self._load_from_cache(cache_key)
                if cached is not None:
                    embeddings.append(cached)
                else:
                    embeddings.append(None)  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            # No caching - process all
            uncached_texts = [t for t in texts if t and t.strip()]
            uncached_indices = list(range(len(texts)))
            embeddings = [None] * len(texts)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            try:
                if self.embedding_model is not None:
                    # Use Sentence Transformers batch processing
                    new_embeddings = await self._sentence_transformer_batch_embedding(uncached_texts)
                else:
                    # Use fallback method
                    new_embeddings = [self._fallback_embedding(text) for text in uncached_texts]
                
                # Fill in the placeholders and cache
                for idx, embedding in zip(uncached_indices, new_embeddings):
                    embeddings[idx] = embedding
                    if getattr(settings, 'ENABLE_EMBEDDING_CACHE', True):
                        cache_key = self._get_cache_key(texts[idx])
                        self._save_to_cache(cache_key, embedding)
                        
            except Exception as e:
                self.logger.error(f"❌ Failed to generate batch embeddings: {e}")
                # Fill remaining with zero vectors
                for idx in uncached_indices:
                    if embeddings[idx] is None:
                        embeddings[idx] = [0.0] * self.embedding_dimension
        
        return embeddings
    
    async def compute_similarity(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Compute similarity matrix for embeddings"""
        try:
            if self.embedding_model is not None and 'nomic-ai' in self.model_name:
                # Use Nomic model's built-in similarity function
                loop = asyncio.get_event_loop()
                embeddings_array = np.array(embeddings)
                similarities = await loop.run_in_executor(
                    self.executor,
                    self.embedding_model.similarity,
                    embeddings_array,
                    embeddings_array
                )
                return similarities.tolist()
            else:
                # Compute cosine similarity manually
                embeddings_array = np.array(embeddings)
                # Normalize vectors
                norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
                normalized = embeddings_array / norms
                # Compute similarity
                similarities = np.dot(normalized, normalized.T)
                return similarities.tolist()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to compute similarity: {e}")
            # Return identity matrix as fallback
            size = len(embeddings)
            return [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    
    async def _sentence_transformer_embedding(self, text: str) -> List[float]:
        """Generate embedding using Sentence Transformers (including Nomic)"""
        try:
            # Run in thread to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self.executor,
                self.embedding_model.encode,
                text
            )
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"❌ Sentence Transformer embedding failed: {e}")
            raise
    
    async def _sentence_transformer_batch_embedding(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings using Sentence Transformers (including Nomic)"""
        try:
            # Run in thread to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                self.executor,
                self.embedding_model.encode,
                texts
            )
            return embeddings.tolist()
        except Exception as e:
            self.logger.error(f"❌ Sentence Transformer batch embedding failed: {e}")
            raise
    
    def _fallback_embedding(self, text: str) -> List[float]:
        """Simple fallback embedding using hash-based method"""
        # Simple hash-based embedding (not recommended for production)
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to float vector
        embedding = []
        for i in range(0, len(hash_bytes), 4):
            chunk = hash_bytes[i:i+4]
            val = int.from_bytes(chunk + b'\x00' * (4 - len(chunk)), 'little')
            embedding.append((val % 1000) / 1000.0 - 0.5)  # Normalize to [-0.5, 0.5]
        
        # Pad or truncate to desired dimension
        while len(embedding) < self.embedding_dimension:
            embedding.extend(embedding[:min(len(embedding), self.embedding_dimension - len(embedding))])
        
        return embedding[:self.embedding_dimension]
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        key_string = f"{self.model_name}:{text}"
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[List[float]]:
        """Load embedding from cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.debug(f"Cache load failed: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: List[float]) -> None:
        """Save embedding to cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            self.logger.debug(f"Cache save failed: {e}")
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dimension
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information"""
        return {
            'model_type': 'open_source',
            'model_name': self.model_name,
            'dimension': self.embedding_dimension,
            'supports_similarity': 'nomic-ai' in (self.model_name or ''),
            'backend': 'sentence_transformers' if self.embedding_model else 'fallback'
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)