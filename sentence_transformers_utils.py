import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import torch
import logging

logger = logging.getLogger(__name__)


def load_user_prompts(data_folder: str, metadata_filename: str) -> List[str]:
    prompts = []
    metadata_path = os.path.join(data_folder, metadata_filename)
    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    prompts.append(parts[1].strip())
                else:
                    logger.warning(f"Skipping malformed line: {line}")
        logger.info(f"Loaded {len(prompts)} user prompts.")
        return prompts
    except Exception as e:
        logger.error(f"Error loading user prompts: {e}")
        raise


def save_embeddings_to_file(embeddings: np.ndarray, embeddings_path: str):
    """Save embeddings to a file."""
    try:
        np.save(embeddings_path, embeddings)
        logger.info(f"Saved embeddings to {embeddings_path}")
    except Exception as e:
        logger.error(f"Failed to save embeddings to {embeddings_path}: {e}")
        raise


def generate_and_save_embeddings(
    data_folder: str, metadata_filename: str, embeddings_path: str, model_name: str
):
    """Generate embeddings from user prompts and save them to a file."""
    try:
        # Load prompts
        prompts = load_user_prompts(data_folder, metadata_filename)

        # Load model
        device = torch.device("cpu")
        model = SentenceTransformer(model_name, device=device)
        logger.info(
            f"Loaded SentenceTransformer model for generating embeddings: {model_name}"
        )

        # Generate embeddings
        embeddings = model.encode(prompts, convert_to_numpy=True)
        logger.info(f"Generated embeddings for {len(prompts)} prompts.")

        # Save embeddings to a file
        save_embeddings_to_file(embeddings, embeddings_path)
    except Exception as e:
        logger.error(f"Error generating and saving embeddings: {e}")
        raise


def load_embeddings(
    data_folder: str, metadata_filename: str, embeddings_filename: str, model_name: str
) -> np.ndarray:
    """Load embeddings from a file, generating them if necessary."""
    embeddings_path = os.path.join(data_folder, embeddings_filename)
    if not os.path.isfile(embeddings_path):
        logger.warning(
            f"{embeddings_filename} not found. Attempting to generate embeddings."
        )
        metadata_path = os.path.join(data_folder, metadata_filename)
        if not os.path.isfile(metadata_path):
            logger.critical(
                f"{metadata_filename} not found. Cannot generate embeddings."
            )
            sys.exit(1)

        logger.info("Generating embeddings from metadata file")
        generate_and_save_embeddings(
            data_folder, metadata_filename, embeddings_path, model_name
        )

    try:
        embeddings = np.load(embeddings_path)
        logger.info(f"Loaded embeddings from {embeddings_filename}")
        return embeddings
    except Exception as e:
        logger.error(f"Error loading embeddings from {embeddings_filename}: {e}")
        raise


def load_data(data_folder: str, metadata_filename: str) -> List[List[str]]:
    """Load video data from the metadata file."""
    metadata_path = os.path.join(data_folder, metadata_filename)

    if not os.path.isfile(metadata_path):
        logger.critical(f"{metadata_filename} not found. Exiting application.")
        sys.exit(1)

    videos = []
    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    videos.append(parts)
                else:
                    logger.warning(f"Skipping malformed line: {line}")
        logger.info(f"Loaded {len(videos)} videos from {metadata_filename}")
        return videos
    except Exception as e:
        logger.error(f"Error loading data from {metadata_filename}: {e}")
        raise
