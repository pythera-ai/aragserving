#!/usr/bin/env python3
"""
Test script for Semantic Retrieval System API
"""

import requests
import json
import time
import numpy as np

# API Configuration
API_BASE_URL = "http://localhost:8080"

def test_health_endpoints():
    """Test health check endpoints"""
    print("🔍 Testing health endpoints...")
    
    # Test API health
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"✅ API Health: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ API Health failed: {e}")
    
    # Test model readiness
    try:
        response = requests.get(f"{API_BASE_URL}/models/ready")
        print(f"✅ Models Ready: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Models Ready failed: {e}")

def test_context_processing():
    """Test context processing endpoint"""
    print("\n🔍 Testing context processing...")
    
    # Test with text input
    test_text = "Artificial Intelligence là một lĩnh vực rộng lớn trong khoa học máy tính. Machine Learning là một phần quan trọng của AI. Deep Learning lại là một nhánh của Machine Learning sử dụng mạng neural nhiều lớp."
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/context",
            data={"text": test_text}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Context processing successful! Generated {len(result)} chunks")
            for i, chunk in enumerate(result[:2]):  # Show first 2 chunks
                print(f"   Chunk {i+1}: ID={chunk['id'][:8]}..., Text='{chunk['chunk'][:50]}...', Emb shape={len(chunk['emb'])}")
            return result
        else:
            print(f"❌ Context processing failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Context processing error: {e}")
        return None

def test_query_processing():
    """Test query processing endpoint"""
    print("\n🔍 Testing query processing...")
    
    query_data = {
        "text": "Machine Learning trong AI là gì?"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json=query_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Query processing successful! Generated {len(result)} chunks")
            for i, chunk in enumerate(result):
                print(f"   Query {i+1}: ID={chunk['id'][:8]}..., Text='{chunk['chunk'][:50]}...', Emb shape={len(chunk['emb'])}")
            return result
        else:
            print(f"❌ Query processing failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Query processing error: {e}")
        return None

def test_reranking(query_chunks, context_chunks):
    """Test reranking endpoint"""
    print("\n🔍 Testing reranking...")
    
    if not query_chunks or not context_chunks:
        print("❌ Cannot test reranking: missing query or context chunks")
        return
    
    # Prepare rerank request
    rerank_data = {
        "query": [
            {
                "id": chunk["id"],
                "emb": chunk["emb"]
            } for chunk in query_chunks
        ],
        "context": [
            {
                "id": chunk["id"],
                "emb": chunk["emb"]
            } for chunk in context_chunks
        ],
        "thresh": [0.3, 0.9],
        "limit": 5
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/rerank",
            json=rerank_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Reranking successful! Generated {len(result)} scored results")
            for i, item in enumerate(result):
                print(f"   Result {i+1}: Context ID={item['context_id'][:8]}..., Score={item['score']:.4f}")
            return result
        else:
            print(f"❌ Reranking failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Reranking error: {e}")
        return None

def test_file_upload():
    """Test file upload functionality"""
    print("\n🔍 Testing file upload...")
    
    # Create a test markdown file
    test_content = """# Artificial Intelligence

## Machine Learning
Machine Learning is a subset of AI that focuses on algorithms that can learn from data.

## Deep Learning
Deep Learning uses neural networks with multiple layers to solve complex problems.

## Natural Language Processing
NLP helps computers understand and process human language.
"""
    
    try:
        # Test with file upload
        files = {"file": ("test.md", test_content, "text/markdown")}
        response = requests.post(f"{API_BASE_URL}/context", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ File upload successful! Generated {len(result)} chunks")
            return result
        else:
            print(f"❌ File upload failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ File upload error: {e}")
        return None

def main():
    """Run all tests"""
    print("🚀 Starting Semantic Retrieval System API Tests")
    print("=" * 60)
    
    # Wait for server to be ready
    print("⏳ Waiting for server to be ready...")
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✅ Server is ready!")
                break
        except:
            pass
        time.sleep(1)
        print(f"   Attempt {i+1}/30...")
    else:
        print("❌ Server is not responding. Please start the API server first.")
        return
    
    # Run tests
    test_health_endpoints()
    
    context_chunks = test_context_processing()
    query_chunks = test_query_processing()
    
    if context_chunks and query_chunks:
        test_reranking(query_chunks, context_chunks)
    
    test_file_upload()
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")

if __name__ == "__main__":
    main()
