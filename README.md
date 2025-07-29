# TritonStruct - Multi-BERT Model Inference Server

A comprehensive Triton Inference Server deployment for serving multiple BERT-based models with different specializations including context encoding, query processing, reranking.

## Overview

This repository provides a production-ready setup for deploying multiple BERT models using NVIDIA Triton Inference Server. The system includes:

- **Context Encoder (mbert-ctx)**: For encoding document contexts
- **Query Encoder (mbert-qry)**: For encoding search queries  
- **Reranking Model (mbert-rerank)**: For reranking search results
- **Ensemble Pipeline**: Coordinated tokenization and inference workflow

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Text Input    │───▶│  Tokenizer   │───▶│ BERT Models     │
└─────────────────┘    └──────────────┘    └─────────────────┘
                              │                      │
                              ▼                      ▼
                       ┌──────────────┐    ┌─────────────────┐
                       │ input_ids    │    │  Embeddings/    │
                       │ attention    │    └─────────────────┘
                       │ token_types  │    
                       └──────────────┘
```


## Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with Docker GPU support
- NVIDIA Container Toolkit

### 1. Clone and Start Main Server

```bash
# Start the main Triton server with ensemble model
docker compose up -d

# The server will be available at:
# HTTP: http://localhost:7000
# gRPC: localhost:7001  
# Metrics: http://localhost:7002
```

### 2. Start Individual Model Services

Use the provided script to start:

```bash
docker compose up -d
```