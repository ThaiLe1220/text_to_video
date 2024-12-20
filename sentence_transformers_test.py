import time
import numpy as np
import sys
import logging
import torch

# Make sure this import matches your local structure
from sentence_transformers_utils import load_data, load_embeddings
from sentence_transformers import SentenceTransformer

# Configuration
DEVICE = torch.device("cpu")
DATA_FOLDER = "data_ori"
METADATA_FILENAME = "metadata.txt"

# Models and their corresponding embeddings files
AVAILABLE_MODELS = {
    "sentence-transformers/all-mpnet-base-v2": {
    	"embeddings_filename": "embeddings_all-mpnet-base-v2.npy",
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "embeddings_filename": "embeddings_all-MiniLM-L6-v2.npy",
    },
}

# Setup logging (optional, you can skip if not needed)
logging.basicConfig(
    filename="performance_test.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load data
try:
    videos = load_data(DATA_FOLDER, METADATA_FILENAME)
except Exception as e:
    logger.critical(f"Failed to load data: {e}")
    sys.exit(1)

# Load all models and embeddings
models = {}
embeddings_dict = {}

for model_name, config in AVAILABLE_MODELS.items():
    try:
        print(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name, device=DEVICE)
        models[model_name] = model

        # Load or generate embeddings
        embeddings = load_embeddings(
            DATA_FOLDER, METADATA_FILENAME, config["embeddings_filename"], model_name
        )
        embeddings_dict[model_name] = embeddings

        print(f"Model and embeddings ready for: {model_name}\n")
    except Exception as e:
        print(f"Failed to load model or embeddings for {model_name}: {e}")
        logger.critical(f"Failed to load model or embeddings for {model_name}: {e}")
        sys.exit(1)


def load_test_prompts(filename="prompt_gpt_thai.txt"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            # Filter out empty lines and strip leading/trailing spaces
            prompts = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(prompts)} test prompts from {filename}")
            return prompts
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        sys.exit(1)


# Test prompts
test_prompts = load_test_prompts("prompt_gpt_thai.txt")

# Number of top results to retrieve
TOP_K = 10

print("\n=== PERFORMANCE TEST RESULTS ===\n")

for prompt in test_prompts:
    print(f'=== PROMPT: "{prompt}" ===\n')

    for model_name in AVAILABLE_MODELS.keys():
        model = models[model_name]
        sentence_embeddings = embeddings_dict[model_name]
        print(f"--- MODEL: {model_name} ---")

        t1 = time.time()

        # Encode prompt
        prompt_embedding = model.encode([prompt])[0]

        # Compute cosine similarity
        norms = np.linalg.norm(sentence_embeddings, axis=1) * np.linalg.norm(
            prompt_embedding
        )
        similarities = np.divide(
            np.dot(sentence_embeddings, prompt_embedding),
            norms,
            out=np.zeros_like(np.dot(sentence_embeddings, prompt_embedding)),
            where=norms != 0,
        )

        # Get top K indices
        sorted_indices = np.argsort(similarities)[::-1][:TOP_K]

        t2 = time.time()
        elapsed = round(t2 - t1, 4)

        # Print results in a compact txt inline format
        print(f"TIME: {elapsed}s")
        for rank, idx in enumerate(sorted_indices, start=1):
            relevance = round(similarities[idx] * 100, 2)
            video_name = videos[idx][0]
            video_prompt = videos[idx][1]
            print(f"{rank:2d}: {relevance:2.2f}% | {video_name:20s} | {video_prompt}")
        print("")  # Add a newline for better readability
    print("\n" + "=" * 50 + "\n")  # Separator between prompts
