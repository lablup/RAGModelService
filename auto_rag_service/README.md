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