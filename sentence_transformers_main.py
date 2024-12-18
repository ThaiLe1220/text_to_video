from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
import time
from sentence_transformers_utils import load_data, load_embeddings
import uvicorn
import logging
import torch
from typing import Optional

# Configuration (Consider moving to environment variables or a config file)
DEVICE = torch.device('cpu')
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
HOST = '127.0.0.1'
PORT = 5000

# Setup logging
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load data and embeddings
try:
    videos = load_data()
    embeddings = load_embeddings()
except Exception as e:
    logger.critical(f"Failed to load data or embeddings: {e}")
    sys.exit(1)

# Load the SentenceTransformer model
try:
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    logger.info(f'Loaded model: {MODEL_NAME}')
except Exception as e:
    logger.critical(f"Failed to load model {MODEL_NAME}: {e}")
    sys.exit(1)

app = FastAPI()

class SearchRequest(BaseModel):
    prompt: str

@app.post('/search')
def compare_route(
    request: SearchRequest, 
    items: int = Query(5, gt=0, description="Maximum number of prompts to return")
):
    try:
        prompt = request.prompt
        t1 = time.time()
        
        # Generate embedding for the given prompt
        prompt_embedding = model.encode([prompt])[0]
        sentence_embeddings = embeddings
        
        # Calculate cosine similarities with error handling
        norms = np.linalg.norm(sentence_embeddings, axis=1) * np.linalg.norm(prompt_embedding)
        similarities = np.divide(
            np.dot(sentence_embeddings, prompt_embedding), 
            norms, 
            out=np.zeros_like(np.dot(sentence_embeddings, prompt_embedding)), 
            where=norms!=0
        )
        
        # Sort based on similarity scores and get the top `items` closest sentences
        sorted_indices = np.argsort(similarities)[::-1][:items]
        
        # Collect the top closest prompts and their related data
        top_closest = [{
            'relevance': round(similarities[i] * 100, 2),
            'video_name': videos[i][0],
            'prompt': videos[i][1],
        } for i in sorted_indices]
        
        t2 = time.time()
        
        # Log the result
        if sorted_indices.size > 0:
            top_video = videos[sorted_indices[0]][0]
        else:
            top_video = 'No results found'
        logger.info(f'{round(t2 - t1, 2)}s - Prompt: {prompt} - Found: {top_video}')
        
        return {'result': top_closest}
    except Exception as e:
        logger.error(f"Error in /search endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == '__main__':
    uvicorn.run(app, host=HOST, port=PORT)
