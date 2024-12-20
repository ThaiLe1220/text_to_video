import time
import numpy as np
import sys
import logging
import torch
import pandas as pd
from sentence_transformers_utils import load_data, load_embeddings
from sentence_transformers import SentenceTransformer

# Configuration
DEVICE = torch.device("cpu")
DATA_FOLDER = "data"
METADATA_FILENAME = "metadata.txt"

AVAILABLE_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "embeddings_filename": "embeddings_all-MiniLM-L6-v2.npy",
    },
    # Add more models if needed
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

        embeddings = load_embeddings(
            DATA_FOLDER, METADATA_FILENAME, config["embeddings_filename"], model_name
        )
        embeddings_dict[model_name] = embeddings

        print(f"Model and embeddings ready for: {model_name}\n")
    except Exception as e:
        print(f"Failed to load model or embeddings for {model_name}: {e}")
        logger.critical(f"Failed to load model or embeddings for {model_name}: {e}")
        sys.exit(1)


def load_test_prompts(filename="prompt_gpt_new.txt"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(prompts)} test prompts from {filename}")
            return prompts
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        sys.exit(1)


def export_results_with_summary(results, filename="performance_results_summary.xlsx"):
    if not results:
        print("No results to export.")
        return

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(results)

    # Calculate overall summary metrics
    summary_metrics = {
        "Metric": [
            "Overall Average Top Score (%)",
            "Overall Average Bottom Score (%)",
            "Overall Average of Average Scores (%)",
            "Overall Median of Median Scores (%)",
            "Overall Average Std Deviation (%)",
            "Overall Average Time Taken (sec)",
        ],
        "Value": [
            df["Top_Score_Percentage"].mean(),
            df["Bottom_Score_Percentage"].mean(),
            df["Average_Score_Percentage"].mean(),
            df["Median_Score_Percentage"].median(),
            df["Std_Score_Percentage"].mean(),
            df["Time_Taken_sec"].mean(),
        ],
    }

    summary_df = pd.DataFrame(summary_metrics)

    # Additional summary: Top Performing Model based on Average_Score_Percentage
    top_model = df.groupby("Model_Name")["Average_Score_Percentage"].mean().idxmax()
    top_model_avg_score = (
        df.groupby("Model_Name")["Average_Score_Percentage"].mean().max()
    )

    top_model_metric = {
        "Metric": ["Top Performing Model"],
        "Value": [f"{top_model} with an average score of {top_model_avg_score:.2f}%"],
    }
    top_model_df = pd.DataFrame(top_model_metric)

    # Create an Excel writer object using openpyxl as the engine
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        # Write the main results table
        df.to_excel(writer, sheet_name="Performance Results", index=False, startrow=0)

        # Write the summary metrics below the main table
        summary_start_row = len(df) + 3  # Adding some space
        summary_df.to_excel(
            writer,
            sheet_name="Performance Results",
            index=False,
            startrow=summary_start_row,
        )

        # Write the top model information below the summary
        top_model_start_row = (
            summary_start_row + len(summary_df) + 2
        )  # Adding some space
        top_model_df.to_excel(
            writer,
            sheet_name="Performance Results",
            index=False,
            startrow=top_model_start_row,
        )

    print(f"Results and summary successfully exported to {filename}")


# Test prompts
test_prompts = load_test_prompts("prompt_gpt.txt")
TOP_K = 10

retrieval_results = []

print("\n=== PERFORMANCE TEST RESULTS ===\n")

for idx, prompt in enumerate(test_prompts):
    print(f'=== PROMPT {idx+1}: "{prompt}" ===\n')

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

        # Extract scores for analysis
        top_scores = similarities[sorted_indices] * 100  # convert to percentage

        # Focus metrics:
        # Top score (rank 1)
        top_rank_idx, top_rank_score = sorted_indices[0], top_scores[0]
        top_video_name = videos[top_rank_idx][0]
        top_video_prompt = videos[top_rank_idx][1]

        # Bottom score (rank 10)
        bottom_rank_idx, bottom_rank_score = sorted_indices[-1], top_scores[-1]
        bottom_video_name = videos[bottom_rank_idx][0]
        bottom_video_prompt = videos[bottom_rank_idx][1]

        # Average score
        avg_score = np.mean(top_scores)

        # Median score
        median_score = np.median(top_scores)

        # Std deviation
        std_score = np.std(top_scores)

        # Print them out for logging
        print(f"TIME: {elapsed}s")
        print(
            f"Top Score (Rank 1): {top_rank_score:.2f}% | {top_video_name} | {top_video_prompt}"
        )
        print(
            f"Bottom Score (Rank {TOP_K}): {bottom_rank_score:.2f}% | {bottom_video_name} | {bottom_video_prompt}"
        )
        print(f"Average Score: {avg_score:.2f}%")
        print(f"Median Score: {median_score:.2f}%")
        print(f"Std Deviation: {std_score:.2f}%")
        print("")

        # Store the summarized result
        retrieval_results.append(
            {
                "Prompt_ID": idx + 1,
                "Prompt_Text": prompt,
                "Model_Name": model_name,
                "Time_Taken_sec": elapsed,
                "Top_Score_Percentage": top_rank_score,
                "Top_Score_Video_Name": top_video_name,
                "Top_Score_Video_Prompt": top_video_prompt,
                "Bottom_Score_Percentage": bottom_rank_score,
                "Bottom_Score_Video_Name": bottom_video_name,
                "Bottom_Video_Prompt": bottom_video_prompt,
                "Average_Score_Percentage": avg_score,
                "Median_Score_Percentage": median_score,
                "Std_Score_Percentage": std_score,
            }
        )
    print("\n" + "=" * 50 + "\n")

# After all prompts are processed, export the results with summary
export_results_with_summary(retrieval_results, filename="results_summary_gpt.xlsx")
