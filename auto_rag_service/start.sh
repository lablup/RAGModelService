#!/bin/bash
# Start script for the RAG Model Service
# This script starts the RAG service with the specified docs path

# Source environment variables
if [ -f /models/RAGModelService/.env ]; then
    source /models/RAGModelService/.env
    echo "Environment variables loaded from .env"
else
    echo "Warning: .env file not found. Make sure OPENAI_API_KEY is set."
fi

# Get the docs path from the first argument
DOCS_PATH="$1"

# Get the GitHub URL from the model definition
GITHUB_URL=$(grep -A 10 "pre_start_actions" /models/RAGModelService/auto_rag_service/deploy/model-definition*.yaml | grep "github" | sed -E 's/.*https:/https:/g' | sed -E 's/'\''.*//g')

# Extract owner and repo from GitHub URL
OWNER=$(echo "$GITHUB_URL" | sed -E 's|https://github.com/([^/]+)/.*|\1|')
REPO=$(echo "$GITHUB_URL" | sed -E 's|https://github.com/[^/]+/([^/]+).*|\1|')

# Set the full path to the docs directory
REPO_DIR="/models/RAGModelService/rag_service/${OWNER}/${REPO}"
FULL_DOCS_PATH="$REPO_DIR"
if [ ! -z "$DOCS_PATH" ]; then
    FULL_DOCS_PATH="${REPO_DIR}/${DOCS_PATH}"
fi

echo "Starting RAG service with docs path: $FULL_DOCS_PATH"

# Check if the repository exists
if [ ! -d "$REPO_DIR" ]; then
    echo "Error: Repository directory not found at $REPO_DIR"
    echo "Please run setup.sh first to clone the repository."
    exit 1
fi

# Launch the RAG service using create_rag_service.py
cd /models/RAGModelService
python /models/RAGModelService/auto_rag_service/create_rag_service.py \
    --github-url "$GITHUB_URL" \
    --docs-path "$FULL_DOCS_PATH" \
    --host "0.0.0.0" \
    --port 8000

# Keep the container running
tail -f /dev/null