from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager
import concurrent.futures
from src.product_rag_langgraph import run_product_rag, initialize_models
from src.document_rag import run_document_rag, initialize_models as init_document_models

executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for request/response validation
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User search query")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "blue Nike running shoes under $100"
            }
        }

class ProductResponse(BaseModel):
    name: str
    brand: Optional[str]
    color: Optional[str]
    price: float
    currency: Optional[str]
    description: Optional[str]
    url: Optional[str]
    images: Optional[List[str]]
    score: float

class SearchResponse(BaseModel):
    llm_response: str = Field(..., alias="LlmResponse")
    products: List[Dict] = Field(..., alias="Products")
    query: str
    timestamp: str
    processing_time_ms: float

    class Config:
        populate_by_name = True

class DocumentSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User question about documents")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the return policy for shoes?"
            }
        }

class DocumentSearchResponse(BaseModel):
    llm_response: str = Field(..., alias="LlmResponse")
    documents: List[Dict] = Field(..., alias="Documents")
    query: str
    timestamp: str
    processing_time_ms: float

    class Config:
        populate_by_name = True

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize models once
    logger.info("Starting up - initializing models...")
    initialize_models()
    init_document_models()
    logger.info("Models initialized successfully")
    yield
    # Shutdown: cleanup if needed
    logger.info("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Product RAG API",
    description="Semantic product search with AI-powered recommendations",
    version="1.0.0",
    lifespan=lifespan
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "rag-api",
        "endpoints": ["product_search", "document_search"]
    }

# Main search endpoint
@app.post("/search/products", response_model=SearchResponse)
async def search_products(request: SearchRequest):
    """
    Search for products using semantic search and get AI-powered recommendations

    - **query**: Natural language search query (e.g., "comfortable running shoes under $100")
    """
    start_time = datetime.utcnow()

    try:
        logger.info(f"[PRODUCTS] Received search request: '{request.query}'")

        # Run the RAG pipeline in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, run_product_rag, request.query)

        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds() * 1000

        logger.info(f"[PRODUCTS] Search completed in {processing_time:.2f}ms - Found {len(result['Products'])} products")

        return SearchResponse(
            LlmResponse=result["LlmResponse"],
            Products=result["Products"],
            query=request.query,
            timestamp=end_time.isoformat(),
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"[PRODUCTS] Error processing search request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Document search endpoint
@app.post("/search/documents", response_model=DocumentSearchResponse)
async def search_documents(request: DocumentSearchRequest):
    """
    Search documentation and get AI-powered answers

    - **query**: Natural language question (e.g., "What is the return policy?")
    """
    start_time = datetime.utcnow()

    try:
        logger.info(f"[DOCUMENTS] Received search request: '{request.query}'")

        # Run the RAG pipeline in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, run_document_rag, request.query)

        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds() * 1000

        logger.info(f"[DOCUMENTS] Search completed in {processing_time:.2f}ms - Found {len(result['Documents'])} document chunks")

        return DocumentSearchResponse(
            LlmResponse=result["LlmResponse"],
            Documents=result["Documents"],
            query=request.query,
            timestamp=end_time.isoformat(),
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"[DOCUMENTS] Error processing search request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """API information endpoint"""
    return {
        "name": "Product RAG API",
        "version": "1.0.0",
        "endpoints": {
            "search": "/search (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
