
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from tqdm import tqdm

from setup.config import SetupConfig

BATCH_SIZE = 32  # Process 32 products at a time

def __create_product_text(product):
    """Combine relevant fields into searchable text"""
    parts = [
        product.get('name', ''),
        product.get('sub_title', ''),
        product.get('description', ''),
        product.get('brand', '')
    ]
    # Filter out empty strings and join
    return ' '.join(part for part in parts if part)

def updateProductEmbeddings(config: SetupConfig):
    print("Loading embedding model...")
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    
    print("Connecting to MongoDB...")
    client = MongoClient(config.MONGO_URI)
    db = client[config.DATABASE_NAME]
    collection = db[config.PRODUCTS_COLLECTION]
    
    # Get all products that don't have embeddings yet
    products = list(collection.find({"embedding": {"$exists": False}}))
    total = len(products)
    
    print(f"Found {total} products without embeddings")
    
    if total == 0:
        print("All products already have embeddings!")
        return
    
    # Process in batches
    for i in tqdm(range(0, total, BATCH_SIZE), desc="Embedding products"):
        batch = products[i:i + BATCH_SIZE]
        
        # Prepare texts for this batch
        texts = [__create_product_text(p) for p in batch]
        
        # Generate embeddings for entire batch at once
        embeddings = model.encode(texts, show_progress_bar=False)
        
        # Store embeddings back to MongoDB
        for product, embedding in zip(batch, embeddings):
            collection.update_one(
                {"_id": product["_id"]},
                {"$set": {"embedding": embedding.tolist()}}
            )
    
    print(f"\n✓ Successfully embedded {total} products!")
    
    # Verify
    embedded_count = collection.count_documents({"embedding": {"$exists": True}})
    print(f"Total products with embeddings: {embedded_count}")
