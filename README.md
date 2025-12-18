# LangGraph RAG Playground

A RAG (Retrieval-Augmented Generation) API for semantic search on products and documents using LangGraph and vector embeddings.

## Quick Start

### 1. Environment Variables

Copy [.env.example](.env.example) to `.env` and configure:

```bash
cp .env.example .env
```

Required variables:

- `MONGO_URI` - MongoDB connection string
- `DATABASE_NAME` - Database name
- `PRODUCTS_COLLECTION` - Products collection name
- `PRODUCTS_VECTOR_INDEX_NAME` - Vector index for products
- `DOCUMENTS_RETURN_POLOCY_COLLECTION_NAME` - Documents collection name
- `DOCUMENTS_VECTOR_INDEX_NAME` - Vector index for documents

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Setup

```bash
python setup/setup.py
```

This will:

- Export sample Nike products set to MongoDB
- Generate product embeddings
- Index and embed a sample policy documents page from the zappos.com website

### 4. Start API

```bash
python api/main.py
```

API runs on `http://localhost:8000`

## API Endpoints

- `GET /docs` - OpenAPI docs
- `POST /search/products` - Semantic product search
- `POST /search/documents` - Document search with AI answers
- `GET /health` - Health check
