#!/usr/bin/env python3
"""
Script đơn giản để test API Semantic Retrieval System
"""

import requests
import json
import tempfile
import os

API_BASE_URL = "http://localhost:8080"

# Sample data
SAMPLE_TEXT = """
Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines.
Machine Learning is a subset of AI that enables computers to learn and improve from experience.
Deep Learning uses neural networks with multiple layers to model complex patterns.
Natural Language Processing (NLP) focuses on the interaction between computers and human language.
Computer Vision enables machines to interpret and understand visual information from the world.
"""

SAMPLE_QUERY = "What is machine learning and how does it relate to artificial intelligence?"

TOKENIZER_NAME = "pythera/sat"

def test_health():
    """Test health endpoint"""
    print("=== Testing Health ===")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def test_models_ready():
    """Test models ready endpoint"""
    print("=== Testing Models Ready ===")
    try:
        response = requests.get(f"{API_BASE_URL}/models/ready", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def test_context_processing():
    """Test context processing"""
    print("=== Testing Context Processing ===")
    try:
        data = {
            "text": SAMPLE_TEXT,
            "tokenizer_name": TOKENIZER_NAME
        }
        
        response = requests.post(
            f"{API_BASE_URL}/context",
            data=data,  
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Received {len(result)} chunks")
            for i, chunk in enumerate(result[:2]):
                print(f"Chunk {i+1}: ID={chunk['id'][:8]}..., Length={len(chunk['chunk'])} chars")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def test_query_processing():
    """Test query processing"""
    print("=== Testing Query Processing ===")
    try:
        data = {
            "text": SAMPLE_QUERY,
            "tokenizer_name": TOKENIZER_NAME
        }
        
        response = requests.post(
            f"{API_BASE_URL}/query",
            json=data,  
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Received {len(result)} query chunks")
            for i, chunk in enumerate(result):
                print(f"Query chunk {i+1}: ID={chunk['id'][:8]}..., Text='{chunk['chunk'][:50]}...'")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def test_reranking():
    """Test reranking"""
    print("=== Testing Reranking ===")
    try:
        context_data = {
            "text": SAMPLE_TEXT,
            "tokenizer_name": TOKENIZER_NAME
        }
        
        query_data = {
            "text": SAMPLE_QUERY,
            "tokenizer_name": TOKENIZER_NAME
        }
        
        context_response = requests.post(
            f"{API_BASE_URL}/context",
            data=context_data,
            timeout=30
        )
        
        query_response = requests.post(
            f"{API_BASE_URL}/query",
            json=query_data,
            timeout=30
        )
        
        if context_response.status_code == 200 and query_response.status_code == 200:
            context_chunks = context_response.json()
            query_chunks = query_response.json()
            
            rerank_data = {
                "query": [
                    {
                        "id": chunk["id"],
                        "text": chunk["chunk"]
                    } for chunk in query_chunks
                ],
                "context": [
                    {
                        "id": chunk["id"],
                        "text": chunk["chunk"]
                    } for chunk in context_chunks[:3]  
                ],
                "tokenizer_name": TOKENIZER_NAME,
                "thresh": [0.0, 1.0],
                "limit": 3
            }
            
            response = requests.post(
                f"{API_BASE_URL}/rerank",
                json=rerank_data,
                timeout=30
            )
            
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Received {len(result)} reranked results")
                for i, item in enumerate(result):
                    print(f"Result {i+1}: Context ID={item['context_id'][:8]}..., Score={item['score']:.4f}")
            else:
                print(f"Error: {response.text}")
        else:
            print("Failed to get context or query chunks")
            if context_response.status_code != 200:
                print(f"Context error: {context_response.text}")
            if query_response.status_code != 200:
                print(f"Query error: {query_response.text}")
            
    except Exception as e:
        print(f"Error: {e}")
    print()

def test_file_upload():
    """Test file upload"""
    print("=== Testing File Upload ===")
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""
# Test Document

This is a test document for API testing.

## Section 1
Content for section 1.

## Section 2  
Content for section 2.
            """)
            temp_file_path = f.name
        
        with open(temp_file_path, 'rb') as f:
            files = {"file": ("test.md", f, "text/markdown")}
            data = {"tokenizer_name": TOKENIZER_NAME}
            
            response = requests.post(
                f"{API_BASE_URL}/context",
                files=files,
                data=data,
                timeout=30
            )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Received {len(result)} chunks from file")
        else:
            print(f"Error: {response.text}")
        
        os.unlink(temp_file_path)
        
    except Exception as e:
        print(f"Error: {e}")
    print()

def main():
    """Chạy tất cả tests"""
    print("Starting API Tests...")
    print(f"API URL: {API_BASE_URL}")
    print(f"Tokenizer: {TOKENIZER_NAME}")
    print("=" * 50)
    
    test_health()
    test_models_ready()
    test_context_processing()
    test_query_processing()
    test_reranking()
    test_file_upload()
    
    print("Tests completed!")

if __name__ == "__main__":
    main()