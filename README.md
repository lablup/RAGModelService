# RAG Model Service

A Retrieval-Augmented Generation (RAG) service for document search and generation, providing a simple and efficient way to build a question-answering system powered by your own documentation.

## Features

- Vector-based document search using OpenAI embeddings and FAISS
- LLM-powered document question-answering with context retrieval
- Simple CLI interface for document indexing and testing
- Interactive web UI built with Gradio
- No project-specific code - works with any documentation structure

## Prerequisites

- Python 3.10+
- OpenAI API key

## Installation

### Option 1: Install from source

```bash
# Clone the repository
git clone https://github.com/lablup/RAGModelService.git
cd RAGModelService

# Install the package in development mode
pip install -e .
```

### Option 2: Install dependencies directly

```bash
pip install -r requirements.txt
python-dotenv
gradio
```

## Configuration

1. Create a `.env` file in the root directory based on `.env_example`:

```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o  # or another OpenAI model
TEMPERATURE=0.2
MAX_RESULTS=5
```

## Usage

### 1. Document Indexing

Index your documents to create a vector store:

```bash
# Using the CLI tool with the installed package
python vectordb_manager/vectordb_manager.py --docs-path ./TensorRT-LLM --indices-path ./embedding_indices --create-index

# to Test the search
python vectordb_manager/vectordb_manager.py --search "What is AWQ?" --top-k 5
```

### 2. Terminal Chat Interface

Test the RAG system with a simple command-line interface:

```bash
# run the module directly
python -m app.rag_chatbot
```

### 3. Web Interface

Launch the Gradio web interface:

```bash
# run the module directly
python app/gradio_app.py
```

### 4. Vector DB Testing

Test the vector database functionality:


## Components

- **vectordb_manager**: Handles document collection, vectorization, and storage
- **app/rag_chatbot.py**: Implements the RAG system core functionality
- **app/gradio_app.py**: Provides a web interface using Gradio
- **app/document_filter.py**: Simple document filtering utility


# Auto RAG Service Launch:
## 1. setup_rag.py
This script handles the first stage of creating a RAG service:

Clones a GitHub repository containing documentation
Processes the documentation to create vector embeddings
Tests the RAG system with sample queries
You can use it independently if you just want to prepare vector embeddings for a repository

## 2. launch_gradio.py
This script handles the second stage:

Takes existing vector store and documentation paths
Configures and initializes the RAG system
Launches a Gradio web interface for interacting with the documentation
Use this if you already have processed documentation and want to start a web interface

## 3. create_rag_service.py
This script combines the functionality of the first two scripts:

Clones a GitHub repository
Processes the documentation
Launches a Gradio web interface
One-command solution for creating a complete RAG service from a GitHub URL

## 4. rag_service_portal.py
This script provides a web-based portal:

Offers a Gradio interface where users can enter GitHub URLs
Processes repositories in the background
Provides links to the resulting RAG services
Manages multiple services simultaneously
This is the "front door" to the system for non-technical users

## 5. rag_launcher.py
This is a utility script that:

Sets up example repositories (optionally)
Launches the RAG service portal
Performs environment checks
Provides a unified entry point to start the whole system

## How They Work Together
The system is designed to be modular, where each script can be used independently or as part of a workflow:

Basic Workflow: setup_rag.py → launch_gradio.py

First process the documentation, then launch the interface


Simplified Workflow: create_rag_service.py

All-in-one solution that combines the two steps


Portal Workflow: rag_service_portal.py

Web-based interface that manages multiple services
Uses the other scripts behind the scenes


Complete System: rag_launcher.py

Sets up and launches the entire system
## License

MIT
