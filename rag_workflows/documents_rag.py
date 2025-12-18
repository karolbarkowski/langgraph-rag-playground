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

# Define State Schema
class DocumentRAGState(TypedDict):
    """State that flows through the graph"""
    original_query: str
    query_embedding: list
    documents: List[dict]  # Retrieved document chunks
    llm_response: str
    error: Optional[str]

# Initialize models (global, loaded once)
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
        mongo_collection = db[langgraphConfig.DOCUMENTS_RETURN_POLOCY_COLLECTION_NAME]

# Node 1: Validate Input
def validate_input(state: DocumentRAGState) -> DocumentRAGState:
    """Validate the user query"""
    query = state.get("original_query", "").strip()
    
    if not query:
        state["error"] = "Query cannot be empty"
        return state
    
    if len(query) < 3:
        state["error"] = "Query too short (minimum 3 characters)"
        return state
    
    state["error"] = None
    print(f"✓ Input validated: '{query}'")
    return state

# Conditional Edge: Check Validation
def check_validation(state: DocumentRAGState) -> Literal["initialize_state", "handle_validation_error"]:
    """Route based on validation result"""
    if state.get("error"):
        return "handle_validation_error"
    return "initialize_state"

# Node 2: Initialize State
def initialize_state(state: DocumentRAGState) -> DocumentRAGState:
    """Initialize all state fields after successful validation"""
    state["documents"] = []
    state["llm_response"] = ""
    state["query_embedding"] = []
    
    print("✓ State initialized")
    return state

# Node 3: Handle Validation Error
def handle_validation_error(state: DocumentRAGState) -> DocumentRAGState:
    """Handle validation failures"""
    error_msg = state.get("error", "Invalid input")
    state["llm_response"] = f"Error: {error_msg}"
    state["documents"] = []
    print(f"✗ Validation failed: {error_msg}")
    return state

# Node 4: Embed Query
def embed_query(state: DocumentRAGState) -> DocumentRAGState:
    """Generate embedding for query"""
    if (embedding_model is None):
        raise Exception("embedding_model cannot be undefined")
    
    embedding = embedding_model.encode(state["original_query"]).tolist()
    state["query_embedding"] = embedding
    
    print(f"✓ Query embedded (dimension: {len(embedding)})")
    return state

# Node 5: Search Documents
def search_documents(state: DocumentRAGState) -> DocumentRAGState:
    """Perform vector search on document chunks"""
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": langgraphConfig.DOCUMENTS_VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": state["query_embedding"],
                "numCandidates": 100,
                "limit": 5  # Top 5 most relevant chunks
            }
        },
        {
            "$project": {
                "_id": {"$toString": "$_id"},
                "url": 1,
                "title": 1,
                "content": 1,
                "chunk_index": 1,
                "total_chunks": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    if (mongo_collection is None):
        raise Exception('mongo_collection cannot be undefined')
    
    documents = list(mongo_collection.aggregate(pipeline))
    state["documents"] = documents
    
    print(f"✓ Search complete: {len(documents)} document chunks found")
    
    return state

# Node 6: Generate Response
def generate_response(state: DocumentRAGState) -> DocumentRAGState:
    """Use LLM to generate answer from retrieved documents"""
    documents = state["documents"]
    
    # Format documents for LLM context
    if not documents:
        context = "No relevant documents found."
    else:
        formatted = []
        for i, doc in enumerate(documents, 1):
            doc_text = f"[Document {i}] From: {doc.get('title', 'Unknown')}\n"
            doc_text += f"URL: {doc.get('url', 'N/A')}\n"
            doc_text += f"Content: {doc['content']}\n"
            formatted.append(doc_text)
        context = "\n---\n".join(formatted)
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on the provided documentation.

Use the document excerpts below to answer the user's question accurately. If the documents don't contain enough information to answer fully, say so.

IMPORTANT: Provide complete answers based on the information available. Do not ask follow-up questions. Do not cite the source documents, just provide short and concise answer to the question asked.

Retrieved Documents:
{documents}"""),
        ("human", "{query}")
    ])

    if (llm_model is None):
        raise Exception("llm_model cannot be undefined")
    
    chain = prompt | llm_model | StrOutputParser()
    
    print("🤖 Generating LLM response...")
    response = chain.invoke({
        "query": state["original_query"],
        "documents": context
    })
    
    state["llm_response"] = response
    print("✓ Response generated")
    
    return state

# Node 7: Handle No Results
def handle_no_results(state: DocumentRAGState) -> DocumentRAGState:
    """Generate response when no documents found"""
    state["llm_response"] = f"I couldn't find any relevant information about '{state['original_query']}' in the documentation."
    print("⚠ No results found")
    return state

# Conditional Edge: Check Results
def check_results(state: DocumentRAGState) -> Literal["generate_response", "handle_no_results"]:
    """Decide next step based on search results"""
    if len(state["documents"]) > 0:
        return "generate_response"
    return "handle_no_results"

# Build the Graph
def create_document_rag_graph():
    """Construct the LangGraph workflow"""
    workflow = StateGraph(DocumentRAGState)
    
    # Add nodes
    workflow.add_node("validate_input", validate_input)
    workflow.add_node("initialize_state", initialize_state)
    workflow.add_node("handle_validation_error", handle_validation_error)
    workflow.add_node("embed_query", embed_query)
    workflow.add_node("search_documents", search_documents)
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
    
    # Continue normal flow
    workflow.add_edge("initialize_state", "embed_query")
    workflow.add_edge("embed_query", "search_documents")
    
    # Conditional edge after search
    workflow.add_conditional_edges(
        "search_documents",
        check_results,
        {
            "generate_response": "generate_response",
            "handle_no_results": "handle_no_results"
        }
    )
    
    # Terminal nodes
    workflow.add_edge("generate_response", END)
    workflow.add_edge("handle_no_results", END)
    workflow.add_edge("handle_validation_error", END)
    
    return workflow.compile()

# Main execution function
def run_document_rag(query: str) -> dict:
    """Execute the document RAG pipeline"""
    initialize_models()
    
    graph = create_document_rag_graph()
    
    # Initial state
    initial_state = {
        "original_query": query
    }

    initial_state = DocumentRAGState(
        original_query=query,
        query_embedding=[],
        documents=[],
        llm_response="",
        error=None
    )
    
    # Run the graph
    print(f"\n{'='*60}")
    print(f"Processing query: '{query}'")
    print(f"{'='*60}\n")
    
    final_state = graph.invoke(initial_state)
    
    # Format output
    output = {
        "LlmResponse": final_state.get("llm_response", ""),
        "Documents": final_state.get("documents", [])
    }
    
    return output

# Test it
if __name__ == "__main__":
    # Test with a return policy question
    result = run_document_rag("What is the return policy for shoes?")
    print(f"\n{'='*60}")
    print("FINAL OUTPUT:")
    print(f"{'='*60}")
    print(f"\nLLM Response:\n{result['LlmResponse']}")
    print(f"\nDocuments Found: {len(result['Documents'])}")
    
    # Show which documents were retrieved
    if result['Documents']:
        print(f"\nRetrieved from:")
        for doc in result['Documents']:
            print(f"  - {doc['title']} (chunk {doc['chunk_index']+1}/{doc['total_chunks']}, score: {doc['score']:.3f})")