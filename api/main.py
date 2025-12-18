from fastapi import FastAPI, HTTPException
from typing import Optional
from datetime import datetime
from contextlib import asynccontextmanager
from api.products_search.ProductsSearchRequest import ProductsSearchRequest
from api.products_search.ProductsSearchResponse import ProductsSearchResponse
from api.documents_search.DocumentSearchRequest import DocumentSearchRequest
from api.documents_search.DocumentSearchResponse import DocumentSearchResponse
from rag_workflows.products_rag import run_product_rag, initialize_models as init_product_models
from rag_workflows.documents_rag import run_document_rag, initialize_models as init_document_models
import concurrent.futures
import asyncio
import logging

executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
title = "RAG API Playground"
version = "1.0.0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize models once
    logger.info("Starting up - initializing models...")
    init_product_models()
    init_document_models()
    logger.info("Models initialized successfully")
    yield
    # Shutdown: cleanup if needed
    logger.info("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title=title,
    version=version,
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "rag-api",
    }

@app.post("/search/products", response_model=ProductsSearchResponse)
async def search_products(request: ProductsSearchRequest):
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

        return ProductsSearchResponse(
            LlmResponse=result["LlmResponse"],
            Products=result["Products"],
            query=request.query,
            timestamp=end_time.isoformat(),
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"[PRODUCTS] Error processing search request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
