#!/usr/bin/env python3
"""
Generate Model Definition

This script generates a model definition YAML file for a RAG service based on a GitHub URL.
It extracts the repository name from the URL to use as the documentation name.

Usage:
    python generate_model_definition.py --github-url https://github.com/owner/repo
"""

import argparse
import os
import re
import sys
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate model definition YAML for RAG service"
    )
    
    # GitHub URL
    parser.add_argument(
        "--github-url",
        type=str,
        help="GitHub URL of documentation repository",
        required=True,
    )
    
    # Output file
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for the model definition file",
        default=".",
    )
    
    # Model name prefix
    parser.add_argument(
        "--name-prefix",
        type=str,
        help="Prefix for the model name",
        default="RAG Service for",
    )
    
    # Port
    parser.add_argument(
        "--port",
        type=int,
        help="Port for the service",
        default=8000,
    )
    
    # Service type
    parser.add_argument(
        "--service-type",
        type=str,
        help="Type of service (gradio or fastapi)",
        choices=["gradio", "fastapi"],
        default="gradio",
    )
    
    return parser.parse_args()


def parse_github_url(github_url: str) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Parse a GitHub URL to extract owner, repo, branch, and path.
    
    Args:
        github_url: GitHub URL
        
    Returns:
        Tuple of (owner, repo, branch, path)
    """
    # Remove any trailing slashes
    github_url = github_url.rstrip('/')
    
    # Match GitHub URL with optional branch and path (tree format)
    tree_pattern = r"https?://github\.com/([^/]+)/([^/]+)(?:/tree/([^/]+)(?:/(.+))?)?"
    tree_match = re.match(tree_pattern, github_url)
    
    if tree_match:
        owner = tree_match.group(1)
        repo = tree_match.group(2)
        branch = tree_match.group(3)  # This will be None if branch is not specified
        path = tree_match.group(4)    # This will be None if path is not specified
        return owner, repo, branch, path
    
    # Match GitHub URL with blob format
    blob_pattern = r"https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)"
    blob_match = re.match(blob_pattern, github_url)
    
    if blob_match:
        owner = blob_match.group(1)
        repo = blob_match.group(2)
        branch = blob_match.group(3)
        file_path = blob_match.group(4)
        
        # For blob URLs, extract the directory path
        # First, check if the file path contains a directory
        if '/' in file_path:
            # Get the directory part of the path (everything before the last slash)
            dir_path = '/'.join(file_path.split('/')[:-1])
            return owner, repo, branch, dir_path
        else:
            # It's a file in the root directory
            return owner, repo, branch, None
    
    # Basic GitHub URL (just owner/repo)
    basic_pattern = r"https?://github\.com/([^/]+)/([^/]+)"
    basic_match = re.match(basic_pattern, github_url)
    
    if basic_match:
        owner = basic_match.group(1)
        repo = basic_match.group(2)
        return owner, repo, None, None
    
    # If URL doesn't match any expected patterns
    raise ValueError(f"Invalid GitHub URL: {github_url}")


def generate_model_name(owner: str, repo: str, path: Optional[str], prefix: str) -> str:
    """
    Generate a model name based on the GitHub URL components.
    
    Args:
        owner: GitHub repository owner
        repo: GitHub repository name
        path: Path within the repository (if any)
        prefix: Prefix for the model name
        
    Returns:
        Model name
    """
    # Use the full repository name
    repo_name = repo
    
    # Format model name
    if path:
        if path == "docs":
            return f"{prefix} {repo_name} Documentation"
        else:
            return f"{prefix} {repo_name} Documentation ({path})"
    else:
        return f"{prefix} {repo_name}"


def generate_docs_name(owner: str, repo: str, path: Optional[str]) -> str:
    """
    Generate a documentation name for the YAML filename based on the GitHub URL components.
    
    Args:
        owner: GitHub repository owner
        repo: GitHub repository name
        path: Path within the repository (if any)
        
    Returns:
        Documentation name
    """
    # Use the full repository name (lowercase)
    repo_name = repo.lower()
    
    # Format docs name
    if path:
        # Replace slashes with hyphens and remove any special characters
        path_part = re.sub(r'[^a-zA-Z0-9-]', '', path.replace('/', '-'))
        return f"{repo_name}-{path_part}"
    else:
        return repo_name


def generate_model_definition(github_url: str, model_name: str, port: int, service_type: str) -> Dict:
    """
    Generate a model definition for the RAG service.
    
    Args:
        github_url: GitHub URL
        model_name: Model name
        port: Port number
        service_type: Service type (gradio or fastapi)
        
    Returns:
        Model definition as a dictionary
    """
    # Parse the GitHub URL
    owner, repo, branch, path = parse_github_url(github_url)
    
    # Determine the docs path argument
    docs_path_arg = path if path else ""
    
    # Create the model definition
    model_definition = {
        "models": [
            {
                "name": model_name,
                "model_path": "/models",
                "service": {
                    "pre_start_actions": [
                        "/bin/bash",
                        "/models/RAGModelService/auto_rag_service/setup.sh",
                        github_url
                    ],
                    "start_command": [
                        "/bin/bash",
                        "/models/RAGModelService/auto_rag_service/start.sh",
                        docs_path_arg
                    ],
                    "docs_path_arg": docs_path_arg,
                    "port": port,
                    "repo_owner": owner,
                    "repo_name": repo
                }
            }
        ]
    }
    
    return model_definition


def write_model_definition(model_definition: Dict, output_path: Path) -> None:
    """
    Write model definition to YAML file.
    
    Args:
        model_definition: Model definition dictionary
        output_path: Output file path
    """
    with open(output_path, "w") as f:
        yaml.dump(model_definition, f, default_flow_style=False)
    
    print(f"Model definition written to {output_path}")


def main():
    """Main function."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Parse GitHub URL
        owner, repo, branch, path = parse_github_url(args.github_url)
        
        # Generate model name
        model_name = generate_model_name(owner, repo, path, args.name_prefix)
        
        # Generate docs name for the filename
        docs_name = generate_docs_name(owner, repo, path)
        
        # Generate model definition
        model_definition = generate_model_definition(
            args.github_url,
            model_name,
            args.port,
            args.service_type
        )
        
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output file name
        output_path = output_dir / f"model-definition-{docs_name}.yaml"
        
        # Write model definition to file
        write_model_definition(model_definition, output_path)
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())