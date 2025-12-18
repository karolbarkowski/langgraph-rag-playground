import time
from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from bs4 import BeautifulSoup
from tqdm import tqdm
from typing import List, Dict
from datetime import datetime

from setup.config import SetupConfig

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# URLs to index
URLS_TO_INDEX = [
    "https://www.zappos.com/c/shipping-and-returns"
]

def __fetch_url_content(url: str, browser) -> Dict:
    """Fetch and extract main content from a URL using Playwright"""
    try:
        print(f"Fetching: {url}")
        
        # Create a new page
        page = browser.new_page()
        
        # Navigate to URL
        page.goto(url, wait_until="networkidle", timeout=30000)
        
        # Wait a bit for any lazy-loaded content
        page.wait_for_timeout(2000)
        
        # Get the HTML content
        html_content = page.content()
        
        # Get page title
        title = page.title()
        
        # Close the page
        page.close()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
        
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks_text = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks_text if chunk)
        
        return {
            "url": url,
            "title": title,
            "content": text,
            "fetched_at": datetime.utcnow().isoformat(),
            "status": "success",
            "content_length": len(text)
        }
        
    except Exception as e:
        print(f"Error fetching {url}: {str(e)}")
        return {
            "url": url,
            "status": "error",
            "error": str(e)
        }

def __chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If this isn't the last chunk, try to break at a sentence or word boundary
        if end < len(text):
            # Look for sentence end (. ! ?)
            last_period = text.rfind('.', start, end)
            last_question = text.rfind('?', start, end)
            last_exclamation = text.rfind('!', start, end)
            
            break_point = max(last_period, last_question, last_exclamation)
            
            if break_point > start:
                end = break_point + 1
            else:
                # No sentence boundary, break at space
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start forward, accounting for overlap
        start = end - overlap if end < len(text) else end
    
    return chunks

def index_documents(config: SetupConfig):
    """Main indexing pipeline"""
    print("="*60)
    print("DOCUMENT INDEXING PIPELINE")
    print("="*60)
    print(f"Collection: {config.DOCUMENTS_RETURN_POLOCY_COLLECTION_NAME}")
    print(f"Chunk size: {CHUNK_SIZE} chars, Overlap: {CHUNK_OVERLAP} chars")
    print(f"URLs to process: {len(URLS_TO_INDEX)}")
    print("="*60)
    
    # Load embedding model
    print("\nLoading embedding model...")
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    print(f"Model loaded (dimension: {model.get_sentence_embedding_dimension()})")
    
    # Connect to MongoDB
    print("\nConnecting to MongoDB...")
    client = MongoClient(config.MONGO_URI)
    db = client[config.DATABASE_NAME]
    collection = db[config.DOCUMENTS_RETURN_POLOCY_COLLECTION_NAME]
    print(f"Connected to collection: {config.DOCUMENTS_RETURN_POLOCY_COLLECTION_NAME}")
    
    # Start Playwright browser
    print("\nStarting browser...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        print("Browser started")
        
        # Process each URL
        total_chunks = 0
        failed_urls = []
        
        for url in tqdm(URLS_TO_INDEX, desc="Processing URLs"):
            # Fetch content
            result = __fetch_url_content(url, browser)
            
            if result["status"] == "error":
                failed_urls.append(url)
                continue
            
            print(f"\n  Fetched {result['content_length']} characters from {url}")
            
            # Delete old chunks for this URL (re-indexing)
            deleted = collection.delete_many({"url": url})
            if deleted.deleted_count > 0:
                print(f"  Removed {deleted.deleted_count} old chunks")
            
            # Chunk the content
            chunks = __chunk_text(result["content"])
            print(f"  Created {len(chunks)} chunks")
            
            # Embed and store each chunk
            for i, chunk_content  in enumerate(chunks):
                # Generate embedding
                embedding = model.encode(chunk_content ).tolist()
                
                # Create document
                doc = {
                    "url": url,
                    "title": result["title"],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "content": chunk_content,
                    "embedding": embedding,
                    "fetched_at": result["fetched_at"],
                    "chunk_size": len(chunk_content ),
                    "metadata": {
                        "collection_type": "url_document",
                        "knowledge_base": "return_policy"
                    }
                }
                
                collection.insert_one(doc)
                total_chunks += 1
            
            # Be nice to servers
            time.sleep(1)
        
        browser.close()
    
    print(f"\n{'='*60}")
    print("INDEXING COMPLETE")
    print(f"{'='*60}")
    print(f"✓ Total chunks indexed: {total_chunks}")
    print(f"✓ Successful URLs: {len(URLS_TO_INDEX) - len(failed_urls)}")
    if failed_urls:
        print(f"✗ Failed URLs: {len(failed_urls)}")
        for url in failed_urls:
            print(f"  - {url}")
    
    # Show sample
    print(f"\nSample chunk from collection:")
    sample = collection.find_one()
    if sample:
        print(f"  URL: {sample['url']}")
        print(f"  Title: {sample['title']}")
        print(f"  Chunk {sample['chunk_index']}/{sample['total_chunks']}")
        print(f"  Content preview: {sample['content'][:150]}...")
        print(f"  Embedding dimension: {len(sample['embedding'])}")
