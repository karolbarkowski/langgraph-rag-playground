import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import TypedDict, List, Optional, Literal
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag_workflows.config import langgraphConfig
import re

# Define State Schema
class ProductRAGState(TypedDict):
    """State that flows through the graph"""
    original_query: str
    semantic_query: str
    filters: dict
    query_embedding: list
    max_retry_count: int
    current_retry_counter: int
    products: List[dict]
    llm_response: str
    error: Optional[str]

# Initialize models (we'll load these once globally)
embedding_model = None
llm_model = None
mongo_collection = None

def initialize_models():
    """Load models once at startup"""
    global embedding_model, llm_model, mongo_collection
    
    if embedding_model is None:
        print("Loading embedding model...")
        embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    
    if llm_model is None:
        print("Initializing LLM...")
        llm_model = ChatOllama(model="llama3.1", temperature=0.7)
    
    if mongo_collection is None:
        print("Connecting to MongoDB...")
        client = MongoClient(langgraphConfig.MONGO_URI)
        db = client[langgraphConfig.DATABASE_NAME]
        mongo_collection = db[langgraphConfig.PRODUCTS_COLLECTION]

# Node 1: Validate Input (PURE - no state initialization)
def validate_input(state: ProductRAGState) -> ProductRAGState:
    """Validate the user query - pure validation only"""
    query = state.get("original_query", "").strip()
    
    if not query:
        state["error"] = "Query cannot be empty"
        return state
    
    if len(query) < 3:
        state["error"] = "Query too short (minimum 3 characters)"
        return state
    
    # No error - validation passed
    state["error"] = None
    print(f"✓ Input validated: '{query}'")
    return state

# Conditional Edge: Check Validation
def check_validation(state: ProductRAGState) -> Literal["initialize_state", "handle_validation_error"]:
    """Route based on validation result"""
    if state.get("error"):
        return "handle_validation_error"
    return "initialize_state"

# Node 2: Initialize State
def initialize_state(state: ProductRAGState) -> ProductRAGState:
    """Initialize all state fields after successful validation"""
    state["current_retry_counter"] = 0
    state["max_retry_count"] = 1
    state["filters"] = {}
    state["products"] = []
    state["llm_response"] = ""
    state["query_embedding"] = []
    state["semantic_query"] = ""
    
    print("✓ State initialized")
    return state

# Node 3: Handle Validation Error
def handle_validation_error(state: ProductRAGState) -> ProductRAGState:
    """Handle validation failures"""
    error_msg = state.get("error", "Invalid input")
    state["llm_response"] = f"Error: {error_msg}"
    state["products"] = []
    print(f"✗ Validation failed: {error_msg}")
    return state

# Node 4: Extract Filters (Simple MVP version)
def extract_filters(state: ProductRAGState) -> ProductRAGState:
    """Extract filters from query using simple regex/keyword matching"""
    query = state["original_query"].lower()
    filters = {}
    semantic_parts = [state["original_query"]]  # Start with full query
    
    # Extract brand (hardcoded common brands for MVP)
    brands = ["nike", "adidas", "puma", "reebok", "under armour", "new balance"]
    for brand in brands:
        if brand in query:
            filters["brand"] = brand.title()
            # Remove brand from semantic query
            semantic_parts = [part.replace(brand, "").strip() for part in semantic_parts]
    
    # Extract price (simple patterns like "under $100", "below 50")
    price_patterns = [
        (r"under \$?(\d+)", lambda x: {"$lt": int(x)}),
        (r"below \$?(\d+)", lambda x: {"$lt": int(x)}),
        (r"less than \$?(\d+)", lambda x: {"$lt": int(x)}),
        (r"over \$?(\d+)", lambda x: {"$gt": int(x)}),
        (r"above \$?(\d+)", lambda x: {"$gt": int(x)}),
    ]
    
    for pattern, price_fn in price_patterns:
        match = re.search(pattern, query)
        if match:
            filters["price"] = price_fn(match.group(1))
            break
    
    # Extract colors (hardcoded common colors)
    colors = ["blue", "red", "black", "white", "green", "yellow", "pink", "grey", "gray"]
    for color in colors:
        if color in query:
            filters["color"] = color.title()
            semantic_parts = [part.replace(color, "").strip() for part in semantic_parts]
    
    # Clean up semantic query
    semantic_query = " ".join(semantic_parts).strip()
    semantic_query = re.sub(r'\s+', ' ', semantic_query)  # Remove extra spaces
    
    state["filters"] = filters
    state["semantic_query"] = semantic_query if semantic_query else state["original_query"]
    
    print(f"✓ Filters extracted: {filters}")
    print(f"✓ Semantic query: '{state['semantic_query']}'")
    
    return state

# Node 5: Embed Query
def embed_query(state: ProductRAGState) -> ProductRAGState:
    """Generate embedding for semantic query"""
    if (embedding_model is None):
        raise Exception("embedding_model cannot be undefined")
    
    embedding = embedding_model.encode(state["semantic_query"]).tolist()
    state["query_embedding"] = embedding
    
    print(f"✓ Query embedded (dimension: {len(embedding)})")
    return state

# Node 6: Search Products
def search_products(state: ProductRAGState) -> ProductRAGState:
    """Perform vector search with filters"""
    num_candidates = 100 if state["current_retry_counter"] == 0 else 300
    filters = state["filters"] if state["current_retry_counter"] == 0 else None
    embeddings = state["query_embedding"]
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": langgraphConfig.PRODUCTS_VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": embeddings,
                "numCandidates": num_candidates,
                "limit": 10
            }
        },
        {
            "$project": {
                "_id": {"$toString": "$_id"},
                "name": 1,
                "brand": 1,
                "color": 1,
                "price": 1,
                "currency": 1,
                "description": 1,
                "url": 1,
                "images": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    
    if filters:
        pipeline.insert(1, {"$match": filters})
    
    if (mongo_collection is None):
        raise Exception('mongo_collection cannot be undefined')
    
    products = list(mongo_collection.aggregate(pipeline))
    state["products"] = products
    
    search_type = "with filters" if filters else "without filters (fallback)"
    print(f"✓ Search complete: {len(products)} products found ({search_type}, candidates={num_candidates})")
    
    return state

# Conditional Edge: Check Results
def check_results(state: ProductRAGState) -> Literal["generate_response", "retry_search", "no_results"]:
    """Decide next step based on search results"""
    num_products = len(state["products"])
    
    if num_products >= 3:
        return "generate_response"
    
    if state["current_retry_counter"] < state["max_retry_count"]:
        print(f"⚠ Only {num_products} products found. Retrying with relaxed criteria...")
        return "retry_search"
    
    if num_products > 0:
        # We have some products, just not many
        return "generate_response"
    
    return "no_results"

# Node 7: Retry Search (increments counter)
def retry_search(state: ProductRAGState) -> ProductRAGState:
    """Increment retry counter and go back to search"""
    state["current_retry_counter"] += 1
    print(f"↻ Retry attempt {state['current_retry_counter']}/{state['max_retry_count']}")
    return state

# Node 8: Generate Response
def generate_response(state: ProductRAGState) -> ProductRAGState:
    """Use LLM to generate natural language response"""
    products = state["products"]
    
    # Format products for LLM context
    if not products:
        context = "No products found matching the search criteria."
    else:
        formatted = []
        for i, p in enumerate(products[:5], 1):  # Top 5 for context
            product_text = f"{i}. {p['name']}"
            if p.get('brand'):
                product_text += f" by {p['brand']}"
            product_text += f" - ${p['price']}"
            if p.get('color'):
                product_text += f" ({p['color']})"
            if p.get('description'):
                desc = p['description'][:150] + "..." if len(p['description']) > 150 else p['description']
                product_text += f"\n   {desc}"
            formatted.append(product_text)
        context = "\n\n".join(formatted)
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful shopping assistant. Based on the user's query and the product search results provided, give personalized product recommendations.

Be conversational and helpful. Highlight why specific products match their needs. If the user mentioned specific requirements (price, color, brand), address those directly.

IMPORTANT: Provide complete recommendations based on the information available. Do not ask follow-up questions. If some details are unclear, make reasonable assumptions and provide your best recommendations anyway.

Product Search Results:
{products}"""),
        ("human", "{query}")
    ])

    if (llm_model is None):
        raise Exception("llm_model cannot be undefined")
    
    chain = prompt | llm_model | StrOutputParser()
    
    print("🤖 Generating LLM response...")
    response = chain.invoke({
        "query": state["original_query"],
        "products": context
    })
    
    state["llm_response"] = response
    print("✓ Response generated")
    
    return state

# Node 9: Handle No Results
def handle_no_results(state: ProductRAGState) -> ProductRAGState:
    """Generate appropriate response when no products found"""
    state["llm_response"] = f"I couldn't find any products matching '{state['original_query']}'. Try adjusting your search terms or browsing our catalog."
    print("⚠ No results - generated fallback message")
    return state

# Build the Graph
def create_product_rag_graph():
    """Construct the LangGraph workflow"""
    workflow = StateGraph(ProductRAGState)
    
    # Add nodes
    workflow.add_node("validate_input", validate_input)
    workflow.add_node("initialize_state", initialize_state)
    workflow.add_node("handle_validation_error", handle_validation_error)
    workflow.add_node("extract_filters", extract_filters)
    workflow.add_node("embed_query", embed_query)
    workflow.add_node("search_products", search_products)
    workflow.add_node("retry_search", retry_search)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("handle_no_results", handle_no_results)
    
    # Add edges
    workflow.set_entry_point("validate_input")
    
    # Conditional edge after validation
    workflow.add_conditional_edges(
        "validate_input",
        check_validation,
        {
            "initialize_state": "initialize_state",
            "handle_validation_error": "handle_validation_error"
        }
    )
    
    # Continue normal flow after initialization
    workflow.add_edge("initialize_state", "extract_filters")
    workflow.add_edge("extract_filters", "embed_query")
    workflow.add_edge("embed_query", "search_products")
    
    # Conditional edge after search
    workflow.add_conditional_edges(
        "search_products",
        check_results,
        {
            "generate_response": "generate_response",
            "retry_search": "retry_search",
            "no_results": "handle_no_results"
        }
    )
    
    # Retry loops back to search
    workflow.add_edge("retry_search", "search_products")
    
    # All terminal nodes end the graph
    workflow.add_edge("generate_response", END)
    workflow.add_edge("handle_no_results", END)
    workflow.add_edge("handle_validation_error", END)
    
    return workflow.compile()

# Main execution function
def run_product_rag(query: str) -> dict:
    """Execute the RAG pipeline and return structured output"""
    initialize_models()
    
    graph = create_product_rag_graph()
    
    # Initial state with all required fields
    initial_state = ProductRAGState(
        original_query=query,
        semantic_query="",
        filters={},
        query_embedding=[],
        max_retry_count=1,
        current_retry_counter=0,
        products=[],
        llm_response="",
        error=None
    )

    # Run the graph
    print(f"\n{'='*60}")
    print(f"Processing query: '{query}'")
    print(f"{'='*60}\n")

    final_state = graph.invoke(initial_state)
    
    # Format output as required structure
    output = {
        "LlmResponse": final_state.get("llm_response", ""),
        "Products": final_state.get("products", [])
    }
    
    return output

# Test it
if __name__ == "__main__":
    # Test 1: Invalid query (should trigger validation error)
    result = run_product_rag("")
    print(f"\n{'='*60}")
    print("FINAL OUTPUT (INVALID QUERY):")
    print(f"{'='*60}")
    print(f"\nLLM Response:\n{result['LlmResponse']}")
    print(f"\nProducts Found: {len(result['Products'])}")
    
    # Test 2: Simple query
    print("\n\n")
    result = run_product_rag("comfortable running shoes")
    print(f"\n{'='*60}")
    print("FINAL OUTPUT:")
    print(f"{'='*60}")
    print(f"\nLLM Response:\n{result['LlmResponse']}")
    print(f"\nProducts Found: {len(result['Products'])}")
    
    # Test 3: Query with filters
    # print("\n\n")
    # result2 = run_product_rag("blue Nike shoes under $100")
    # print(f"\n{'='*60}")
    # print("FINAL OUTPUT:")
    # print(f"{'='*60}")
    # print(f"\nLLM Response:\n{result2['LlmResponse']}")
    # print(f"\nProducts Found: {len(result2['Products'])}")