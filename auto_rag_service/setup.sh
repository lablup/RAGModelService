#!/bin/bash
# Setup script for the RAG Model Service
# This script installs the necessary dependencies
# and generates the model definition YAML file

# Install the package in development mode
cd /models/RAGModelService
pip install -e .

# Source environment variables
if [ -f .env ]; then
    source .env
    echo "Environment variables loaded from .env"
else
    echo "Warning: .env file not found. Make sure OPENAI_API_KEY is set."
fi

# Get the GitHub URL from the first argument
GITHUB_URL="$1"

if [ -z "$GITHUB_URL" ]; then
    echo "No GitHub URL provided. Skipping model definition generation."
    exit 0
fi

# Extract owner and repo from GitHub URL
OWNER=$(echo "$GITHUB_URL" | sed -E 's|https://github.com/([^/]+)/.*|\1|')
REPO=$(echo "$GITHUB_URL" | sed -E 's|https://github.com/[^/]+/([^/]+).*|\1|')

# Create directory for cloned repositories
DOCS_DIR="/models/RAGModelService/rag_service/${OWNER}"
mkdir -p "$DOCS_DIR"

# Clone repository if it doesn't exist
if [ ! -d "$DOCS_DIR/$REPO" ]; then
    echo "Cloning repository from $GITHUB_URL to $DOCS_DIR/$REPO..."
    git clone "$GITHUB_URL" "$DOCS_DIR/$REPO"
else
    echo "Repository already exists at $DOCS_DIR/$REPO, skipping clone."
fi

# Generate the model definition YAML file
echo "Generating model definition for $GITHUB_URL..."
python /models/RAGModelService/auto_rag_service/generate_model_definition.py \
    --github-url "$GITHUB_URL" \
    --output-dir "/models/RAGModelService/auto_rag_service/deploy" \
    --service-type "gradio"

echo "Setup complete!"