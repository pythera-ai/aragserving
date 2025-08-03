
## API Documentation

### Environment Variables

The backend API supports the following environment variables:

```bash
TRITON_SERVER_URL=localhost:7000    # Triton server URL
API_PORT=8080                       # API server port
MAX_FILE_SIZE=52428800             # Maximum file size (50MB)
LOG_LEVEL=INFO                     # Logging level
```

### Endpoint Details

#### 1. Context Processing - `POST /context`

Process text or uploaded files into semantic chunks with embeddings.

**Request (Text):**
```bash
curl -X POST "http://localhost:8080/context" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your long document text here..."}'
```

**Request (File Upload):**
```bash
curl -X POST "http://localhost:8080/context" \
  -F "file=@document.txt"
```

**Response:**
```json
[
  {
    "id": "5d41402abc4b2a76b9719d911017c592",
    "chunk": "First semantic chunk of text...", 
    "emb": [0.1, 0.2, 0.3, ...]
  },
  {
    "id": "7d865e959b2466918c9863afca942d0f",
    "chunk": "Second semantic chunk of text...",
    "emb": [0.4, 0.5, 0.6, ...]
  }
]
```

**Supported File Formats:** `.txt`, `.md`, `.docx`

#### 2. Query Processing - `POST /query`

Process query text to generate embeddings for search.

**Request:**
```bash
curl -X POST "http://localhost:8080/query" \
  -H "Content-Type: application/json" \
  -d '{"text": "What is machine learning?"}'
```

**Response:**
```json
[
  {
    "id": "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae",
    "chunk": "What is machine learning?",
    "emb": [0.7, 0.8, 0.9, ...]
  }
]
```

#### 3. Reranking - `POST /rerank`

Rerank context results based on relevance to query using text inputs.

**Request:**
```bash
curl -X POST "http://localhost:8080/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": [
      {
        "id": "query_id_123",
        "text": "machine learning in education"
      }
    ],
    "context": [
      {
        "id": "context_id_456", 
        "text": "AI and ML are transforming education..."
      },
      {
        "id": "context_id_789",
        "text": "Traditional teaching methods..."
      }
    ],
    "thresh": [0.3, 0.9],
    "limit": 10
  }'
```

**Response:**
```json
[
  {
    "context_id": "context_id_456",
    "score": 0.85
  },
  {
    "context_id": "context_id_789", 
    "score": 0.72
  }
]
```

#### 4. Health Checks

**API Health:**
```bash
curl http://localhost:8080/health
```

**Model Readiness:**
```bash
curl http://localhost:8080/models/ready
```

## Model Specifications

| Model | Purpose | Input Format | Output Format |
|-------|---------|--------------|---------------|
| **sati** | Text segmentation | Raw text | Semantic chunks |
| **mbert-retrieve-ctx** | Context encoding | Text chunks | Embeddings (768-dim) |
| **mbert-retrieve-qry** | Query encoding | Query text | Embeddings (768-dim) |  
| **mbert.rerank** | Result reranking | Query + Context text | Relevance scores |

## Usage Examples

### Python Client Example

```python
import requests

# Process a document
response = requests.post(
    "http://localhost:8080/context",
    json={"text": "Your document text here..."}
)
context_chunks = response.json()

# Process a query  
response = requests.post(
    "http://localhost:8080/query",
    json={"text": "Your search query"}
)
query_result = response.json()

# Rerank results
rerank_payload = {
    "query": [{"id": query_result[0]["id"], "text": query_result[0]["chunk"]}],
    "context": [
        {"id": chunk["id"], "text": chunk["chunk"]} 
        for chunk in context_chunks[:5]
    ],
    "thresh": [0.0, 1.0],
    "limit": 5
}

response = requests.post(
    "http://localhost:8080/rerank", 
    json=rerank_payload
)
ranked_results = response.json()
```

### File Upload Example

```python
import requests

# Upload and process a file
with open("document.txt", "rb") as f:
    response = requests.post(
        "http://localhost:8080/context",
        files={"file": f}
    )
chunks = response.json()
```

## Error Handling

The API returns structured error responses:

```json
{
  "error": {
    "code": "HTTP_400",
    "message": "Detailed error message",
    "details": {}
  }
}
```

Common error codes:
- `400`: Bad request (invalid input, file size exceeded)
- `500`: Internal server error (model inference failed)

## Performance Considerations

- **File Size Limit**: 50MB per upload
- **Text Size Limit**: 1MB per request
- **Concurrent Requests**: Supports 100+ concurrent requests
- **Response Time**: < 2s per request (typical)
- **Model Inference**: < 500ms per model call