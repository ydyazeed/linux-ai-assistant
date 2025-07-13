#!/usr/bin/env python3
import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
from typing import Optional

# Configuration
HUGGINGFACE_DATASET = "ydyazeed/linux-ai-chatbot-embedding"  # The actual dataset repository
MODEL_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

FILES_TO_DOWNLOAD = [
    "linux_manual_embeddings.npy",
    "linux_manual_faiss.index",
    "linux_manual_metadata.pkl",
    "common_issues_embeddings.npy",
    "common_issues_faiss.index",
    "common_issues_metadata.pkl"
]

def download_file(url: str, dest_path: Path, desc: Optional[str] = None) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(dest_path, 'wb') as f, tqdm(
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
        desc=desc or dest_path.name
    ) as pbar:
        for data in response.iter_content(block_size):
            size = f.write(data)
            pbar.update(size)

def main():
    # Create necessary directories
    processed_data_dir = Path("processed_data")
    processed_data_dir.mkdir(exist_ok=True)
    
    print("Starting download of required files...")
    
    # Download embeddings and indices
    for filename in FILES_TO_DOWNLOAD:
        dest_path = processed_data_dir / filename
        if dest_path.exists():
            print(f"Skipping {filename} - already exists")
            continue
            
        url = f"https://huggingface.co/datasets/{HUGGINGFACE_DATASET}/resolve/main/{filename}"
        try:
            print(f"\nDownloading {filename}...")
            download_file(url, dest_path)
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")
            print("You may need to run the RAG pipeline first to generate the embeddings.")
            return
    
    # Download model if it doesn't exist
    model_path = Path("mistral-7b-instruct-v0.1.Q4_K_M.gguf")
    if not model_path.exists():
        print("\nDownloading Mistral-7B model...")
        try:
            download_file(MODEL_URL, model_path, "Downloading Mistral-7B")
        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            return
    else:
        print("\nSkipping model download - already exists")
    
    print("\nAll files downloaded successfully!")

if __name__ == "__main__":
    main() 