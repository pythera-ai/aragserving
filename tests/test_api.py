#!/usr/bin/env python3
"""
Pytest test suite for Semantic Retrieval System API
"""

import pytest
import requests
import json
import tempfile
import os
from typing import Dict, Any, List


class TestHealthEndpoints:
    """Test health check and readiness endpoints"""
    
    def test_api_health(self, api_base_url):
        """Test API health endpoint"""
        response = requests.get(f"{api_base_url}/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "service" in data
    
    def test_models_ready(self, api_base_url):
        """Test models readiness endpoint"""
        response = requests.get(f"{api_base_url}/models/ready")
        
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        
        # Check that required models are present
        required_models = ["sat", "retrieve_query", "retrieve_context", "rerank"]
        for model in required_models:
            assert model in data["models"]


class TestContextProcessing:
    """Test context processing endpoint"""
    
    def test_context_processing_with_text(self, api_base_url, sample_text):
        """Test context processing with text input"""
        response = requests.post(
            f"{api_base_url}/context",
            params={"text": sample_text}
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Validate response structure
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Validate each chunk
        for chunk in result:
            assert "id" in chunk
            assert "chunk" in chunk  
            assert "emb" in chunk
            assert isinstance(chunk["id"], str)
            assert isinstance(chunk["chunk"], str)
            assert isinstance(chunk["emb"], list)
            assert len(chunk["id"]) == 32  # MD5 hash length
            assert len(chunk["emb"]) > 0   # Should have embeddings
    
    def test_context_processing_with_file_upload(self, api_base_url, sample_markdown_file):
        """Test context processing with file upload"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(sample_markdown_file)
            temp_file_path = f.name
        
        try:
            # Upload file
            with open(temp_file_path, 'rb') as f:
                files = {"file": ("test.md", f, "text/markdown")}
                response = requests.post(f"{api_base_url}/context", files=files)
            
            assert response.status_code == 200
            result = response.json()
            
            # Validate response
            assert isinstance(result, list)
            assert len(result) > 0
            
            # Should contain content from the markdown file
            all_text = " ".join([chunk["chunk"] for chunk in result])
            assert "Artificial Intelligence" in all_text
            assert "Machine Learning" in all_text
            
        finally:
            # Clean up
            os.unlink(temp_file_path)
    
    def test_context_processing_empty_text(self, api_base_url):
        """Test context processing with empty text"""
        response = requests.post(
            f"{api_base_url}/context",
            params={"text": ""}
        )
        
        assert response.status_code == 400
        error_data = response.json()
        assert "error" in error_data
    
    def test_context_processing_no_input(self, api_base_url):
        """Test context processing with no input"""
        response = requests.post(f"{api_base_url}/context")
        
        assert response.status_code == 400  # FastAPI validation error
    
    def test_context_processing_large_file(self, api_base_url):
        """Test context processing with file exceeding size limit"""
        # Create a large file (over 50MB)
        large_content = "A" * (50 * 1024 * 1024 + 1000)  # 50MB + 1000 bytes
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(large_content)
            temp_file_path = f.name
        
        try:
            with open(temp_file_path, 'rb') as f:
                files = {"file": ("large.txt", f, "text/plain")}
                response = requests.post(f"{api_base_url}/context", files=files)
            
            assert response.status_code == 400
            error_data = response.json()
            assert "error" in error_data
            assert "File size exceeds" in error_data["error"]["message"]
            
        finally:
            os.unlink(temp_file_path)


class TestQueryProcessing:
    """Test query processing endpoint"""
    
    def test_query_processing_valid_text(self, api_base_url, sample_query):
        """Test query processing with valid text"""
        response = requests.post(
            f"{api_base_url}/query",
            json={"text": sample_query}
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Validate response structure
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Validate each chunk
        for chunk in result:
            assert "id" in chunk
            assert "chunk" in chunk
            assert "emb" in chunk
            assert isinstance(chunk["id"], str)
            assert isinstance(chunk["chunk"], str)
            assert isinstance(chunk["emb"], list)
            assert len(chunk["id"]) == 32  # MD5 hash length
            assert len(chunk["emb"]) > 0   # Should have embeddings
    
    def test_query_processing_empty_text(self, api_base_url):
        """Test query processing with empty text"""
        response = requests.post(
            f"{api_base_url}/query",
            json={"text": ""}
        )
        
        assert response.status_code == 400
        error_data = response.json()
        assert "error" in error_data
        assert "cannot be empty" in error_data["error"]["message"]
    
    def test_query_processing_missing_text(self, api_base_url):
        """Test query processing with missing text field"""
        response = requests.post(
            f"{api_base_url}/query",
            json={}
        )
        
        assert response.status_code == 422  # FastAPI validation error
    
    def test_query_processing_invalid_json(self, api_base_url):
        """Test query processing with invalid JSON"""
        response = requests.post(
            f"{api_base_url}/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422


class TestReranking:
    """Test reranking endpoint"""
    
    def test_reranking_valid_input(self, api_base_url, query_chunks, context_chunks):
        """Test reranking with valid query and context"""
        # Use text-based reranking format
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
                } for chunk in context_chunks[:5]  # Limit to 5 for testing
            ],
            "thresh": [0.0, 1.0],
            "limit": 5
        }
        
        response = requests.post(
            f"{api_base_url}/rerank",
            json=rerank_data
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Validate response structure
        assert isinstance(result, list)
        assert len(result) <= 5  # Should respect limit
        
        # Validate each result
        for item in result:
            assert "context_id" in item
            assert "score" in item
            assert isinstance(item["context_id"], str)
            assert isinstance(item["score"], (int, float))
            assert 0.0 <= item["score"] <= 1.0
        
        # Results should be sorted by score (descending)
        if len(result) > 1:
            scores = [item["score"] for item in result]
            assert scores == sorted(scores, reverse=True)
    
    def test_reranking_with_threshold(self, api_base_url, query_chunks, context_chunks):
        """Test reranking with specific threshold"""
        rerank_data = {
            "query": [
                {
                    "id": query_chunks[0]["id"],
                    "text": query_chunks[0]["chunk"]
                }
            ],
            "context": [
                {
                    "id": chunk["id"],
                    "text": chunk["chunk"] 
                } for chunk in context_chunks[:3]
            ],
            "thresh": [0.5, 0.9],  # Only scores between 0.5 and 0.9
            "limit": 10
        }
        
        response = requests.post(
            f"{api_base_url}/rerank",
            json=rerank_data
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # All scores should be within threshold
        for item in result:
            assert 0.5 <= item["score"] <= 0.9
    
    def test_reranking_empty_query(self, api_base_url, context_chunks):
        """Test reranking with empty query"""
        rerank_data = {
            "query": [],
            "context": [
                {
                    "id": context_chunks[0]["id"],
                    "text": context_chunks[0]["chunk"]
                }
            ],
            "thresh": [0.0, 1.0],
            "limit": 5
        }
        
        response = requests.post(
            f"{api_base_url}/rerank",
            json=rerank_data
        )
        
        assert response.status_code == 400
        error_data = response.json()
        assert "error" in error_data
        assert "cannot be empty" in error_data["error"]["message"]
    
    def test_reranking_empty_context(self, api_base_url, query_chunks):
        """Test reranking with empty context"""
        rerank_data = {
            "query": [
                {
                    "id": query_chunks[0]["id"],
                    "text": query_chunks[0]["chunk"]
                }
            ],
            "context": [],
            "thresh": [0.0, 1.0],
            "limit": 5
        }
        
        response = requests.post(
            f"{api_base_url}/rerank",
            json=rerank_data
        )
        
        assert response.status_code == 400
        error_data = response.json()
        assert "error" in error_data
        assert "cannot be empty" in error_data["error"]["message"]
    
    def test_reranking_invalid_threshold(self, api_base_url, query_chunks, context_chunks):
        """Test reranking with invalid threshold"""
        rerank_data = {
            "query": [
                {
                    "id": query_chunks[0]["id"],
                    "text": query_chunks[0]["chunk"]
                }
            ],
            "context": [
                {
                    "id": context_chunks[0]["id"],
                    "text": context_chunks[0]["chunk"]
                }
            ],
            "thresh": [0.8, 0.3],  # Invalid: min > max
            "limit": 5
        }
        
        response = requests.post(
            f"{api_base_url}/rerank",
            json=rerank_data
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_reranking_invalid_limit(self, api_base_url, query_chunks, context_chunks):
        """Test reranking with invalid limit"""
        rerank_data = {
            "query": [
                {
                    "id": query_chunks[0]["id"],
                    "text": query_chunks[0]["chunk"]
                }
            ],
            "context": [
                {
                    "id": context_chunks[0]["id"],
                    "text": context_chunks[0]["chunk"]
                }
            ],
            "thresh": [0.0, 1.0],
            "limit": 150  # Invalid: > 100
        }
        
        response = requests.post(
            f"{api_base_url}/rerank",
            json=rerank_data
        )
        
        assert response.status_code == 422  # Validation error


class TestIntegrationWorkflow:
    """Test complete workflow integration"""
    
    def test_complete_workflow(self, api_base_url, sample_text, sample_query):
        """Test complete workflow: context → query → rerank"""
        
        # Step 1: Process context
        context_response = requests.post(
            f"{api_base_url}/context",
            params={"text": sample_text}
        )
        assert context_response.status_code == 200
        context_chunks = context_response.json()
        assert len(context_chunks) > 0, "Context chunks should not be empty"
        
        # Step 2: Process query
        query_response = requests.post(
            f"{api_base_url}/query",
            json={"text": sample_query}
        )
        assert query_response.status_code == 200
        query_chunks = query_response.json()
        assert len(query_chunks) > 0, "Query chunks should not be empty"
        
        # Step 3: Rerank
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
                } for chunk in context_chunks[:5]
            ],
            "thresh": [0.0, 1.0],
            "limit": 3
        }
        
        rerank_response = requests.post(
            f"{api_base_url}/rerank",
            json=rerank_data
        )
        assert rerank_response.status_code == 200
        rerank_results = rerank_response.json()
        
        # Validate final results
        assert len(rerank_results) <= 3
        assert len(rerank_results) > 0, "Rerank results should not be empty"
        
        # Check that all returned context IDs exist in original context
        context_ids = {chunk["id"] for chunk in context_chunks}
        for result in rerank_results:
            assert result["context_id"] in context_ids
    
    def test_performance_baseline(self, api_base_url, sample_text):
        """Test basic performance expectations"""
        import time
        
        start_time = time.time()
        
        response = requests.post(
            f"{api_base_url}/context",
            params={"text": sample_text}
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        # Should complete within 10 seconds (generous for test environment)
        assert response_time < 10.0, f"Response took {response_time:.2f}s, expected < 10s"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
