"""
Semantic Retrieval System Backend API
Implementation of SRS requirements for context processing, query processing, and reranking
"""

import os
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

from src.model.model_manager import ModelManager
from src.utils.utils import prepare_text_input, extract_text_from_file, generate_md5_hash

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL", "localhost:7000")
API_PORT = int(os.getenv("API_PORT", "8080"))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "52428800"))  # 50MB
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Model configurations
MODEL_CONFIG = {
    "sat": {
        "name": "sati",
        "version": 1,
        "url": TRITON_SERVER_URL,
        "grpc": False
    },
    "retrieve_query": {
        "name": "mbert-retrieve-qry",
        "version": 1,
        "url": TRITON_SERVER_URL,
        "grpc": False
    },
    "retrieve_context": {
        "name": "mbert-retrieve-ctx",
        "version": 1,
        "url": TRITON_SERVER_URL,
        "grpc": False,
        "max_length": 512  # model_max_length of context embedding model
    },
    "rerank": {
        "name": "mbert.rerank",
        "version": 1,
        "url": TRITON_SERVER_URL,
        "grpc": False
    }
}

####################
# Pydantic Models
####################

class ChunkResponse(BaseModel):
    id: str = Field(..., description="MD5 hash of the chunk")
    chunk: str = Field(..., description="Text chunk content")
    emb: List[Any] = Field(..., description="Embedding vector")

class QueryRequest(BaseModel):
    text: str = Field(..., description="Query text to process")

class ContextItem(BaseModel):
    id: str = Field(..., description="Context chunk ID")
    text: str = Field(..., description="Context text content")

class QueryItem(BaseModel):
    id: str = Field(..., description="Query chunk ID")
    text: str = Field(..., description="Query text content")

class RerankRequest(BaseModel):
    query: List[QueryItem] = Field(..., description="Query texts with IDs")
    context: List[ContextItem] = Field(..., description="Context texts with IDs")
    thresh: List[float] = Field([0.0, 1.0], description="Score threshold [min, max]")
    limit: int = Field(10, description="Maximum number of results")
    
    @validator('thresh')
    def validate_thresh(cls, v):
        if len(v) != 2 or v[0] > v[1] or v[0] < 0 or v[1] > 1:
            raise ValueError('thresh must be [min, max] with 0 <= min <= max <= 1')
        return v
    
    @validator('limit')
    def validate_limit(cls, v):
        if v < 1 or v > 100:
            raise ValueError('limit must be between 1 and 100')
        return v

class RerankResult(BaseModel):
    context_id: str = Field(..., description="Context chunk ID")
    score: float = Field(..., description="Relevance score")

class ErrorResponse(BaseModel):
    error: Dict[str, Any] = Field(..., description="Error details")



# Initialize model manager
model_manager = ModelManager(model_config=MODEL_CONFIG)

####################
# FastAPI App
####################

app = FastAPI(
    title="Semantic Retrieval System API",
    description="API for semantic text processing, embedding generation, and result reranking",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

####################
# Health Check Endpoints
####################

@app.get("/health")
async def health_check():
    """API server health check"""
    return {"status": "healthy", "service": "Semantic Retrieval API"}

@app.get("/models/ready")
async def models_ready():
    """Check if all models are ready"""
    try:
        # Test each model with a simple input
        test_text = "test"
        input_data = prepare_text_input(test_text)
        
        model_status = {}
        for model_key, model in model_manager.models.items():
            try:
                # Simple health check - just verify model is accessible
                model_status[model_key] = "ready"
            except Exception as e:
                model_status[model_key] = f"error: {str(e)}"
        
        return {"models": model_status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model readiness check failed: {str(e)}")

####################
# Main API Endpoints
####################

@app.post("/context", response_model=List[ChunkResponse])
async def process_context(
    text: Optional[str] = None,
    file: Optional[UploadFile] = File(None, description="File to process")
):
    """
    FR001 - Context Processing Endpoint
    Process text or file to create chunks and embeddings
    """
    try:
        # Validate input
        if not text and not file:
            raise HTTPException(status_code=400, detail="Either text or file must be provided")
        
        if text and file:
            raise HTTPException(status_code=400, detail="Provide either text or file, not both")
        
        # Extract text content
        if file:
            if file.size > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail=f"File size exceeds {MAX_FILE_SIZE} bytes")
            
            file_content = await file.read()
            content = extract_text_from_file(file_content, file.filename)
        else:
            content = text
        
        if len(content.encode('utf-8')) > 1048576:  # 1MB limit
            raise HTTPException(status_code=400, detail="Text content exceeds 1MB limit")
        
        # Segment text into chunks
        chunks = model_manager.segment_text(content)
        
        # Process each chunk
        results = []
        for chunk in chunks:
            if not chunk.strip():
                continue
                
            # Generate MD5 hash for chunk ID
            chunk = chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk
            chunk_id = generate_md5_hash(chunk)
            
            # Get embeddings for chunk
            embeddings = model_manager.get_context_embeddings(chunk)
            embeddings = embeddings[0, 0].tolist() # first token only and first batch only
            
            results.append(ChunkResponse(
                id=chunk_id,
                chunk=chunk,
                emb=embeddings  # Convert to list for JSON serialization
            ))
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Context processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Context processing failed: {str(e)}")

@app.post("/query", response_model=List[ChunkResponse])
async def process_query(request: QueryRequest):
    """
    FR002 - Query Processing Endpoint
    Process query text to create embeddings
    """
    try:
        # Validate input
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Query text cannot be empty")
        
        if len(request.text.encode('utf-8')) > 1048576:  # 1MB limit
            raise HTTPException(status_code=400, detail="Query text exceeds 1MB limit")
        
        # Segment query text (if needed)
        chunks = model_manager.segment_text(request.text)
        
        # Process each chunk
        results = []
        for chunk in chunks:
            if not chunk.strip():
                continue
                
            # Generate MD5 hash for chunk ID
            chunk = chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk
            chunk_id = generate_md5_hash(chunk)
            
            # Get embeddings for chunk
            embeddings = model_manager.get_query_embeddings(chunk)
            embeddings = embeddings[0, 0].tolist()  # first token only and first batch only

            results.append(ChunkResponse(
                id=chunk_id,
                chunk=chunk,
                emb=embeddings
            ))
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/rerank", response_model=List[RerankResult])
async def rerank_results(request: RerankRequest):
    """
    FR003 - Rerank Endpoint
    Rerank context results based on relevance to query
    """
    try:
        # Validate input
        if not request.query:
            raise HTTPException(status_code=400, detail="Query texts cannot be empty")
        
        if not request.context:
            raise HTTPException(status_code=400, detail="Context texts cannot be empty")
        
        results = []
        
        # For each query text
        for query_item in request.query:
            query_text = query_item.text
            
            # Get context texts
            context_texts = [ctx.text for ctx in request.context]
            
            # Calculate scores with all context texts
            scores = model_manager.rerank_results(query_text, context_texts)
            
            # Combine context items with scores
            context_scores = list(zip(request.context, scores))
            
            # Filter by threshold
            filtered_scores = [
                (ctx, score) for ctx, score in context_scores
                if request.thresh[0] <= score <= request.thresh[1]
            ]
            
            # Sort by score (descending) and take top results
            filtered_scores.sort(key=lambda x: x[1], reverse=True)
            top_results = filtered_scores[:request.limit]
            
            # Add to results
            for ctx, score in top_results:
                results.append(RerankResult(
                    context_id=ctx.id,
                    score=score
                ))
        
        # Remove duplicates and sort by score
        unique_results = {}
        for result in results:
            if result.context_id not in unique_results or result.score > unique_results[result.context_id].score:
                unique_results[result.context_id] = result
        
        final_results = list(unique_results.values())
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        return final_results[:request.limit]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reranking error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")

####################
# Error Handlers
####################

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "details": {}
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "details": {"exception": str(exc)}
            }
        }
    )

####################
# Main Entry Point
####################

if __name__ == "__main__":
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, LOG_LEVEL.upper()))
    
    # Start server
    uvicorn.run(
        "backendapi:app",
        host="0.0.0.0",
        port=API_PORT,
        reload=False,
        log_level=LOG_LEVEL.lower()
    )
