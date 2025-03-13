#!/usr/bin/env python3
"""
RAG Service Portal

A Gradio interface that allows users to:
1. Enter a GitHub URL with documentation
2. Process it in the background to create a RAG service
3. Provide a link to the resulting RAG Gradio service
4. Generate a model definition YAML file for Backend.AI model service creation

Usage:
    python rag_service_portal.py
"""

import asyncio
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import gradio as gr
from dotenv import load_dotenv
import structlog

# Ensure project root is in path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Initialize logger
logger = structlog.get_logger()

# Global state to track running services
SERVICES = {}

class ServiceStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


def validate_github_url(url: str) -> bool:
    """
    Validate a GitHub URL.
    
    Args:
        url: URL to validate
        
    Returns:
        True if the URL is valid, False otherwise
    """
    if not url:
        return False
        
    # Basic GitHub URL pattern
    pattern = r"^https?://github\.com/[^/]+/[^/]+(?:/tree/[^/]+(?:/.*)?)?$"
    return bool(re.match(pattern, url))


def get_unique_service_id() -> str:
    """Generate a unique service ID"""
    return str(uuid.uuid4())[:8]


def find_available_port(start_port: int = 8000, end_port: int = 9000) -> int:
    """Find an available port in the specified range"""
    import socket
    
    # Check if port is already in use by a service
    used_ports = [s["port"] for s in SERVICES.values() if "port" in s]
    
    for port in range(start_port, end_port):
        if port in used_ports:
            continue
            
        # Check if port is available
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    
    raise RuntimeError(f"No available ports in range {start_port}-{end_port}")


def create_service_directory(service_id: str) -> Path:
    """
    Create a directory for the service.
    
    Args:
        service_id: Service ID
        
    Returns:
        Path to the service directory
    """
    service_dir = Path("./rag_services") / service_id
    service_dir.mkdir(parents=True, exist_ok=True)
    return service_dir


def generate_model_definition(github_url: str, service_dir: Path) -> Optional[str]:
    """
    Generate a model definition YAML file for the RAG service.
    
    Args:
        github_url: GitHub URL for documentation
        service_dir: Path to the service directory
        
    Returns:
        Path to the generated model definition file, or None if generation failed
    """
    try:
        # Run generate_model_definition.py to create the model definition YAML
        cmd = [
            sys.executable,
            "auto_rag_service/generate_model_definition.py",
            "--github-url", github_url,
            "--output-dir", str(service_dir),
            "--service-type", "gradio"
        ]
        
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        # Extract the path from the output
        output_lines = result.stdout.strip().split('\n')
        for line in output_lines:
            if line.startswith("Model definition written to "):
                model_def_path = line.replace("Model definition written to ", "").strip()
                return model_def_path
        
        # If we couldn't find the path in the output, try to guess it
        # Parse GitHub URL to extract components for the filename
        pattern = r"https?://github\.com/([^/]+)/([^/]+)(?:/tree/([^/]+)(?:/(.+))?)?"
        match = re.match(pattern, github_url)
        if match:
            owner, repo = match.group(1), match.group(2)
            path = match.group(4)  # This will be None if path is not specified
            
            # Generate docs name for the filename
            base_name = f"{owner}-{repo}".lower()
            if path:
                # Replace slashes with hyphens and remove any special characters
                path_part = re.sub(r'[^a-zA-Z0-9-]', '', path.replace('/', '-'))
                docs_name = f"{base_name}-{path_part}"
            else:
                docs_name = base_name
                
            return str(service_dir / f"model-definition-{docs_name}.yaml")
        
        return None
        
    except Exception as e:
        logger.error(f"Error generating model definition: {e}")
        return None


def process_github_url(github_url: str, progress_callback: Optional[callable] = None) -> Dict:
    """
    Process a GitHub URL to create a RAG service.
    
    Args:
        github_url: GitHub URL for documentation
        progress_callback: Callback function to report progress
        
    Returns:
        Service information dictionary
    """
    try:
        # Generate service ID and find available port
        service_id = get_unique_service_id()
        port = find_available_port()
        
        # Create service directory
        service_dir = create_service_directory(service_id)
        docs_dir = service_dir / "docs"
        indices_dir = service_dir / "indices"
        
        # Initialize service information
        service_info = {
            "id": service_id,
            "github_url": github_url,
            "port": port,
            "url": f"http://localhost:{port}",
            "status": ServiceStatus.PENDING,
            "message": "Initializing service...",
            "service_dir": service_dir,
            "docs_dir": docs_dir,
            "indices_dir": indices_dir,
        }
        
        # Add to services dictionary
        SERVICES[service_id] = service_info
        
        # Update progress if callback provided
        if progress_callback:
            progress_callback(0.1, "Cloning repository...")
        
        # Update status
        service_info["status"] = ServiceStatus.PROCESSING
        service_info["message"] = "Cloning repository..."
        
        # Run setup_rag.py to clone repository and create vector indices
        setup_cmd = [
            sys.executable,
            "auto_rag_service/setup_rag.py",
            "--github-url", github_url,
            "--output-dir", str(docs_dir),
            "--indices-path", str(indices_dir),
            "--skip-testing",  # Skip testing to speed up the process
        ]
        
        # Run the command and capture output
        setup_process = subprocess.run(
            setup_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        if progress_callback:
            progress_callback(0.6, "Creating vector embeddings...")
        
        service_info["message"] = "Vector embeddings created successfully."
        
        # Generate model definition YAML
        if progress_callback:
            progress_callback(0.8, "Generating model definition...")
        
        model_def_path = generate_model_definition(github_url, service_dir)
        if model_def_path:
            service_info["model_definition_path"] = model_def_path
            service_info["message"] = "Model definition generated successfully."
        
        # Update status
        service_info["status"] = ServiceStatus.READY
        service_info["message"] = "Service is ready! Launching Gradio interface..."
        
        # Save service info to file for persistence
        with open(service_dir / "service_info.txt", "w") as f:
            for key, value in service_info.items():
                if key not in ["service_dir", "docs_dir", "indices_dir"]:
                    f.write(f"{key}: {value}\n")
        
        # Start the service in a separate thread
        threading.Thread(
            target=start_service,
            args=(service_id,),
            daemon=True
        ).start()
        
        if progress_callback:
            progress_callback(1.0, "Service is ready!")
        
        return service_info
        
    except Exception as e:
        logger.error(f"Error processing GitHub URL: {e}")
        service_info = {
            "id": service_id if 'service_id' in locals() else get_unique_service_id(),
            "github_url": github_url,
            "status": ServiceStatus.ERROR,
            "message": f"Error creating service: {str(e)}",
        }
        SERVICES[service_info["id"]] = service_info
        
        if progress_callback:
            progress_callback(1.0, f"Error: {str(e)}")
        
        return service_info


def start_service(service_id: str) -> None:
    """
    Start a RAG service in a background process.
    
    Args:
        service_id: Service ID
    """
    if service_id not in SERVICES:
        logger.error(f"Service {service_id} not found")
        return
    
    service = SERVICES[service_id]
    
    try:
        # Create command to run the service
        cmd = [
            sys.executable,
            "auto_rag_service/launch_gradio.py",
            "--docs-path", str(service["docs_dir"]),
            "--indices-path", str(service["indices_dir"]),
            "--port", str(service["port"]),
            "--host", "0.0.0.0",  # Allow access from other machines
        ]
        
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        # Store process in service info
        service["process"] = process
        
        # Save the PID for potential later use
        service["pid"] = process.pid
        
        # Wait for service to be ready (checking if port is open)
        for _ in range(30):  # Wait up to 30 seconds
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', service["port"])) == 0:
                    break
            time.sleep(1)
        
        logger.info(f"Service {service_id} started on port {service['port']}")
        
    except Exception as e:
        logger.error(f"Error starting service {service_id}: {e}")
        service["status"] = ServiceStatus.ERROR
        service["message"] = f"Error starting service: {str(e)}"


def create_rag_service(github_url: str, progress=gr.Progress()) -> Tuple[str, str, str, str]:
    """
    Create a RAG service from a GitHub URL (Gradio interface function).
    
    Args:
        github_url: GitHub URL for documentation
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (status, message, url, model_definition_path) for the Gradio interface
    """
    # Validate GitHub URL
    if not validate_github_url(github_url):
        return (
            ServiceStatus.ERROR,
            "Invalid GitHub URL. Please enter a valid GitHub repository URL.",
            "",
            ""
        )
    
    # Process GitHub URL with progress updates
    service_info = process_github_url(
        github_url,
        lambda fraction, message: progress(fraction, desc=message)
    )
    
    # Return relevant information for the interface as separate values
    return (
        service_info["status"],
        service_info["message"],
        service_info.get("url", ""),
        service_info.get("model_definition_path", "")
    )


def create_interface() -> gr.Blocks:
    """Create the Gradio interface"""
    with gr.Blocks(title="RAG Service Creator") as interface:
        gr.Markdown("# RAG Service Creator")
        gr.Markdown(
            """
            Create a Retrieval-Augmented Generation (RAG) service from any GitHub repository 
            containing documentation. Simply enter the GitHub URL, and we'll create a service 
            that allows you to query the documentation using natural language.
            """
        )
        
        with gr.Row():
            github_url = gr.Textbox(
                label="GitHub Repository URL",
                placeholder="https://github.com/owner/repo",
                info="Enter a GitHub repository URL containing documentation (markdown files)",
            )
        
        with gr.Row():
            create_button = gr.Button("Create RAG Service", variant="primary")
        
        with gr.Row():
            with gr.Column():
                status = gr.Textbox(label="Status", interactive=False)
                message = gr.Textbox(label="Message", interactive=False)
                service_url = gr.Textbox(
                    label="Service URL", 
                    interactive=False,
                    info="Click this link to access your RAG service when ready"
                )
                model_definition_path = gr.Textbox(
                    label="Model Definition Path",
                    interactive=False,
                    info="Path to the generated model definition YAML file for Backend.AI"
                )
        
        with gr.Row():
            examples = gr.Examples(
                examples=[
                    "https://github.com/pytorch/pytorch",
                    "https://github.com/microsoft/TypeScript",
                    "https://github.com/pandas-dev/pandas",
                    "https://github.com/fastai/fastai",
                ],
                inputs=github_url,
                label="Example Repositories",
            )
        
        # Handle form submission
        create_button.click(
            fn=create_rag_service,
            inputs=[github_url],
            outputs=[
                status,
                message,
                service_url,
                model_definition_path,
            ],
        )
        
        # Add help information
        with gr.Accordion("Help & Information", open=False):
            gr.Markdown(
                """
                ## How it works
                
                1. Enter a GitHub repository URL containing documentation
                2. We'll clone the repository and create vector embeddings using OpenAI's embeddings API
                3. A Gradio interface will be launched for querying the documentation
                4. A model definition YAML file will be generated for Backend.AI model service creation
                5. You'll receive a link to access the service when it's ready
                
                ## Tips
                
                - The repository should contain markdown (.md) files
                - For better results, specify a path to the documentation directory, e.g., `https://github.com/owner/repo/tree/main/docs`
                - The service will remain active as long as this portal is running
                - Each service runs on a different port to avoid conflicts
                - The generated model definition can be used to create a Backend.AI model service
                """
            )
    
    return interface


def main():
    """Main function"""
    # Setup environment
    load_dotenv()
    
    # Check for OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OpenAI API key is required. Please set the OPENAI_API_KEY environment variable.")
        print("You can create a .env file with the following content:")
        print("OPENAI_API_KEY=your_api_key_here")
        return 1
    
    # Create and launch the interface
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=8000)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())