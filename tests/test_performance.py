"""
Performance tests for Semantic Retrieval System API
"""

import pytest
import requests
import time
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed


class TestPerformance:
    """Performance and load testing"""
    
    @pytest.mark.performance
    def test_response_time_context(self, api_base_url, sample_text):
        """Test context processing response time"""
        start_time = time.time()
        
        response = requests.post(
            f"{api_base_url}/context",
            params={"text": sample_text}
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 5.0, f"Context processing took {response_time:.2f}s, expected < 5s"
        
        print(f"\n📊 Context processing time: {response_time:.3f}s")
    
    @pytest.mark.performance  
    def test_response_time_query(self, api_base_url, sample_query):
        """Test query processing response time"""
        start_time = time.time()
        
        response = requests.post(
            f"{api_base_url}/query",
            json={"text": sample_query}
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 3.0, f"Query processing took {response_time:.2f}s, expected < 3s"
        
        print(f"\n📊 Query processing time: {response_time:.3f}s")
    
    @pytest.mark.performance
    def test_response_time_rerank(self, api_base_url, query_chunks, context_chunks):
        """Test reranking response time"""
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
                } for chunk in context_chunks[:10]  # Test with 10 contexts
            ],
            "thresh": [0.0, 1.0],
            "limit": 5
        }
        
        start_time = time.time()
        
        response = requests.post(
            f"{api_base_url}/rerank",
            json=rerank_data
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 4.0, f"Reranking took {response_time:.2f}s, expected < 4s"
        
        print(f"\n📊 Reranking time: {response_time:.3f}s")
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_concurrent_requests(self, api_base_url, sample_query):
        """Test concurrent request handling"""
        num_requests = 10
        max_workers = 5
        
        def make_request():
            """Make a single request"""
            start_time = time.time()
            response = requests.post(
                f"{api_base_url}/query",
                json={"text": sample_query}
            )
            end_time = time.time()
            return {
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "success": response.status_code == 200
            }
        
        # Execute concurrent requests
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [future.result() for future in as_completed(futures)]
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        response_times = [r["response_time"] for r in successful_requests]
        
        # Assertions
        assert len(successful_requests) == num_requests, f"Only {len(successful_requests)}/{num_requests} requests succeeded"
        
        avg_time = statistics.mean(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        
        assert avg_time < 6.0, f"Average response time {avg_time:.2f}s too high"
        assert max_time < 10.0, f"Max response time {max_time:.2f}s too high"
        
        print(f"\n📊 Concurrent requests ({num_requests} requests, {max_workers} workers):")
        print(f"   ✅ Success rate: {len(successful_requests)}/{num_requests}")
        print(f"   📈 Avg response time: {avg_time:.3f}s")
        print(f"   ⏱️  Min/Max time: {min_time:.3f}s / {max_time:.3f}s")
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_large_text_processing(self, api_base_url):
        """Test processing of large text input"""
        # Create a large text (around 100KB)
        large_text = "Artificial Intelligence và Machine Learning " * 2000
        
        start_time = time.time()
        
        response = requests.post(
            f"{api_base_url}/context",
            params={"text": large_text}
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        result = response.json()
        assert len(result) > 0
        
        # Should still complete within reasonable time
        assert response_time < 15.0, f"Large text processing took {response_time:.2f}s, expected < 15s"
        
        print(f"\n📊 Large text processing:")
        print(f"   📝 Text size: ~{len(large_text)} characters")
        print(f"   🔢 Generated chunks: {len(result)}")
        print(f"   ⏱️  Processing time: {response_time:.3f}s")
    
    @pytest.mark.performance
    def test_memory_usage_stability(self, api_base_url, sample_text):
        """Test that multiple requests don't cause memory leaks"""
        num_iterations = 20
        response_times = []
        
        for i in range(num_iterations):
            start_time = time.time()
            
            response = requests.post(
                f"{api_base_url}/context",
                params={"text": sample_text}
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            assert response.status_code == 200
            response_times.append(response_time)
            
            # Small delay between requests
            time.sleep(0.1)
        
        # Check that response times don't degrade significantly
        first_half = response_times[:num_iterations//2]
        second_half = response_times[num_iterations//2:]
        
        avg_first = statistics.mean(first_half)
        avg_second = statistics.mean(second_half)
        
        # Second half shouldn't be more than 50% slower than first half
        degradation = (avg_second - avg_first) / avg_first
        assert degradation < 0.5, f"Performance degraded by {degradation*100:.1f}% over {num_iterations} requests"
        
        print(f"\n📊 Memory stability test ({num_iterations} iterations):")
        print(f"   📈 First half avg: {avg_first:.3f}s")
        print(f"   📈 Second half avg: {avg_second:.3f}s") 
        print(f"   📊 Performance change: {degradation*100:+.1f}%")


class TestScalability:
    """Scalability and stress testing"""
    
    @pytest.mark.stress
    @pytest.mark.slow
    def test_high_concurrency(self, api_base_url, sample_query):
        """Test high concurrency handling"""
        num_requests = 50
        max_workers = 20
        
        def make_request():
            try:
                response = requests.post(
                    f"{api_base_url}/query",
                    json={"text": sample_query},
                    timeout=30  # Longer timeout for stress test
                )
                return response.status_code == 200
            except:
                return False
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [future.result() for future in as_completed(futures)]
        
        success_rate = sum(results) / len(results)
        
        # Should handle at least 80% of high-concurrency requests successfully
        assert success_rate >= 0.8, f"Success rate {success_rate*100:.1f}% too low for high concurrency"
        
        print(f"\n🔥 High concurrency test ({num_requests} requests, {max_workers} workers):")
        print(f"   ✅ Success rate: {success_rate*100:.1f}%")
    
    @pytest.mark.stress
    @pytest.mark.slow  
    def test_sustained_load(self, api_base_url, sample_query):
        """Test sustained load over time"""
        duration_seconds = 60  # 1 minute test
        request_interval = 0.5  # Request every 0.5 seconds
        
        start_time = time.time()
        successful_requests = 0
        total_requests = 0
        
        while time.time() - start_time < duration_seconds:
            try:
                response = requests.post(
                    f"{api_base_url}/query",
                    json={"text": sample_query},
                    timeout=10
                )
                if response.status_code == 200:
                    successful_requests += 1
                total_requests += 1
                
                time.sleep(request_interval)
                
            except:
                total_requests += 1
                time.sleep(request_interval)
                continue
        
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        requests_per_second = total_requests / duration_seconds
        
        # Should maintain good performance under sustained load
        assert success_rate >= 0.9, f"Success rate {success_rate*100:.1f}% too low for sustained load"
        assert requests_per_second >= 1.5, f"Request rate {requests_per_second:.1f} req/s too low"
        
        print(f"\n⏰ Sustained load test ({duration_seconds}s):")
        print(f"   ✅ Success rate: {success_rate*100:.1f}%")
        print(f"   📊 Requests/second: {requests_per_second:.1f}")
        print(f"   🔢 Total requests: {successful_requests}/{total_requests}")


if __name__ == "__main__":
    # Run performance tests specifically
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "-m", "performance"  # Only run performance tests by default
    ])
