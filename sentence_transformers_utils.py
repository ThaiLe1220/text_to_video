import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import torch
import logging

# Setup logging
logger = logging.getLogger(__name__)

DATA_FOLDER = 'data'

def load_user_prompts() -> List[str]:
    prompts = []
    try:
        metadata_path = os.path.join(DATA_FOLDER, 'metadata.txt')
        with open(metadata_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    prompts.append(parts[1].strip())
                else:
                    logger.warning(f"Skipping malformed line: {line}")
        logger.info(f"Loaded {len(prompts)} user prompts.")
        return prompts
    except Exception as e:
        logger.error(f"Error loading user prompts: {e}")
        raise

def save_embeddings_to_file(embeddings: np.ndarray, filename: str):
    """Save embeddings to a file."""
    try:
        np.save(filename, embeddings)
        logger.info(f"Saved embeddings to {filename}")
    except Exception as e:
        logger.error(f"Failed to save embeddings to {filename}: {e}")
        raise

def generate_and_save_embeddings(filename: str):
    """Generate embeddings from user prompts and save them to a file."""
    try:
        # Load prompts
        prompts = load_user_prompts()
        
        # Load model
        device = torch.device('cpu')
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
        logger.info("Loaded SentenceTransformer model for generating embeddings.")
    
        # Generate embeddings
        embeddings = model.encode(prompts, convert_to_numpy=True)
        logger.info(f"Generated embeddings for {len(prompts)} prompts.")
    
        # Save embeddings to a file
        save_embeddings_to_file(embeddings, filename)
    except Exception as e:
        logger.error(f"Error generating and saving embeddings: {e}")
        raise

def load_embeddings() -> np.ndarray:
    """Load embeddings from a file, generating them if necessary."""
    filename = 'embeddings.npy'
    filepath = os.path.join(DATA_FOLDER, filename)

    if not os.path.isfile(filepath):
        logger.warning(f"{filename} not found. Attempting to generate embeddings.")
        metadata_path = os.path.join(DATA_FOLDER, 'metadata.txt')
        if not os.path.isfile(metadata_path):
            logger.critical('metadata.txt not found. Cannot generate embeddings.')
            sys.exit(1)
        
        logger.info('Generating embeddings from metadata.txt')
        generate_and_save_embeddings(filepath)
    
    try:
        embeddings = np.load(filepath)
        logger.info(f'Loaded embeddings from {filename}')
        return embeddings
    except Exception as e:
        logger.error(f"Error loading embeddings from {filename}: {e}")
        raise

def load_data() -> List[List[str]]:
    """Load video data from metadata.txt."""
    filename = 'metadata.txt'
    filepath = os.path.join(DATA_FOLDER, filename)
    
    if not os.path.isfile(filepath):
        logger.critical('metadata.txt not found. Exiting application.')
        sys.exit(1)
        
    videos = []
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    videos.append(parts)
                else:
                    logger.warning(f"Skipping malformed line: {line}")
        logger.info(f"Loaded {len(videos)} videos from {filename}")
        return videos
    except Exception as e:
        logger.error(f"Error loading data from {filename}: {e}")
        raise
