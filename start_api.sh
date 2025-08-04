#!/bin/bash

# Semantic Retrieval System Startup Script

echo "🚀 Starting Semantic Retrieval System..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed"
    exit 1
fi

# Install requirements if needed
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

# Set environment variables
export TRITON_SERVER_URL=${TRITON_SERVER_URL:-"localhost:7000"}
export API_PORT=${API_PORT:-"8080"}
export MAX_FILE_SIZE=${MAX_FILE_SIZE:-"52428800"}
export LOG_LEVEL=${LOG_LEVEL:-"INFO"}

echo "🔧 Configuration:"
echo "   Triton Server URL: $TRITON_SERVER_URL"
echo "   API Port: $API_PORT"
echo "   Max File Size: $MAX_FILE_SIZE bytes"
echo "   Log Level: $LOG_LEVEL"

# Check if Triton server is accessible
echo "🔍 Checking Triton server connectivity..."
python3 -c "
import requests
import sys
try:
    # Try to connect to Triton server
    url = 'http://$TRITON_SERVER_URL/v2/health/ready'
    response = requests.get(url, timeout=5)
    if response.status_code == 200:
        print('✅ Triton server is accessible')
    else:
        print('⚠️  Triton server responded with status:', response.status_code)
except Exception as e:
    print('⚠️  Cannot connect to Triton server:', str(e))
    print('   Make sure Triton server is running on $TRITON_SERVER_URL')
"

# Start the API server
echo "🌟 Starting FastAPI server..."
python3 -m src.api.v1
