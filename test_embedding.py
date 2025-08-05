# test_opensource_embedding.py
import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.embedding_service import EmbeddingService

async def test_opensource_embedding():
    """Test Open Source Embedding Service"""
    print("🧪 Testing Open Source Embedding Service")
    print("="*50)
    
    # Initialize service
    embedding_service = EmbeddingService()
    
    # Test data
    sentences = [
        "That is a happy person",
        "That is a happy dog", 
        "That is a very happy person",
        "Today is a sunny day"
    ]
    
    model_info = embedding_service.get_model_info()
    print(f"📊 Model Info: {model_info}")
    print(f"📏 Embedding Dimension: {embedding_service.get_dimension()}")
    print(f"🔧 Backend: {model_info['backend']}")
    
    # Test single embedding
    print("\n🔍 Testing single embedding...")
    single_embedding = await embedding_service.text_to_embedding(sentences[0])
    print(f"   Text: '{sentences[0]}'")
    print(f"   Embedding length: {len(single_embedding)}")
    print(f"   Sample values: {single_embedding[:5]}")
    
    # Test batch embeddings
    print("\n📦 Testing batch embeddings...")
    batch_embeddings = await embedding_service.batch_text_to_embeddings(sentences)
    print(f"   Processed {len(batch_embeddings)} sentences")
    
    # Test similarity computation
    print("\n🔗 Testing similarity computation...")
    similarities = await embedding_service.compute_similarity(batch_embeddings)
    print(f"   Similarity matrix shape: {len(similarities)}x{len(similarities[0])}")
    
    # Show similarity results
    print("\n   Similarity Matrix:")
    for i, row in enumerate(similarities):
        print(f"   Sentence {i+1}: {[f'{val:.3f}' for val in row]}")
    
    # Test caching
    print("\n💾 Testing embedding cache...")
    cached_embedding = await embedding_service.text_to_embedding(sentences[0])
    print(f"   Cached result matches: {single_embedding == cached_embedding}")
    
    # Cleanup
    embedding_service.cleanup()
    print("\n✅ Open source embedding test completed!")

if __name__ == "__main__":
    asyncio.run(test_opensource_embedding())