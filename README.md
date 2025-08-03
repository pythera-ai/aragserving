# TritonStruct - Semantic Retrieval System

A comprehensive Triton Inference Server deployment with FastAPI backend for serving multiple BERT-based models, providing a complete semantic retrieval system including text processing, embedding generation, and result reranking.

## Overview

This repository provides a production-ready setup for deploying multiple BERT models using NVIDIA Triton Inference Server with a FastAPI backend. The system includes:

- **SAT Model (sati)**: Semantic-Aware Text segmentation for intelligent text chunking
- **Context Encoder (mbert-ctx)**: For encoding document contexts and generating embeddings
- **Query Encoder (mbert-qry)**: For encoding search queries and generating embeddings
- **Reranking Model (mbert-rerank)**: For reranking search results based on relevance
- **FastAPI Backend**: REST API providing endpoints for context processing, query processing, and reranking
- **Ensemble Pipeline**: Coordinated tokenization and inference workflow

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Client        │───▶│ FastAPI      │───▶│ Triton Server   │
│   Application   │    │ Backend      │    │                 │
└─────────────────┘    └──────────────┘    └─────────────────┘
                              │                      │
                              ▼                      ▼
                       ┌──────────────┐    ┌─────────────────┐
                       │ REST APIs    │    │ BERT Models     │
                       │ /context     │    │ • SAT           │
                       │ /query       │    │ • mbert-ctx     │
                       │ /rerank      │    │ • mbert-qry     │
                       └──────────────┘    │ • mbert-rerank  │
                                           └─────────────────┘
```

### API Endpoints

The FastAPI backend provides three main endpoints:

- **POST /context**: Process text/files into semantic chunks with embeddings
- **POST /query**: Process query text into embeddings  
- **POST /rerank**: Rerank context results based on query relevance
- **GET /health**: API health check
- **GET /models/ready**: Check Triton model readiness


## Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with Docker GPU support (optional, CPU inference supported)
- NVIDIA Container Toolkit (for GPU acceleration)
- Python 3.8+ (for backend API)

### 1. Start Triton Server

```bash
# Start the main Triton server with all models
docker compose up -d

# The server will be available at:
# HTTP: http://localhost:7000
# gRPC: localhost:7001  
# Metrics: http://localhost:7002
```

### 2. Install Backend Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### 3. Start FastAPI Backend

```bash
# Start the API server
python backendapi.py

# Or use the startup script
chmod +x start_api.sh
./start_api.sh

# The API will be available at:
# API: http://localhost:8080
# Documentation: http://localhost:8080/docs
# ReDoc: http://localhost:8080/redoc
```

### 4. Test the API

```bash
# Test API health
curl http://localhost:8080/health

# Test model readiness  
curl http://localhost:8080/models/ready

# Process context text
curl -X POST "http://localhost:8080/context" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your context text here"}'

# Process query
curl -X POST "http://localhost:8080/query" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your query here"}'
```


## Development

### Running Tests

```bash
# Run API tests
python test_api.py

# Test with notebook
jupyter notebook test.ipynb
```

### Custom Configuration

Edit `MODEL_CONFIG` in `backendapi.py` to customize model settings:

```python
MODEL_CONFIG = {
    "sat": {
        "name": "sati",
        "version": 1, 
        "url": "localhost:7000",
        "grpc": False
    },
    # ... other models
}
```