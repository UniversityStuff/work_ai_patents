import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from tqdm import tqdm
import numpy as np
import time
import random

# Optional imports for BERT
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

load_dotenv()

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Embedding models
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
BERT_MODEL_NAME = "anferico/bert-for-patents"

# Cache directory
CACHE_DIR = Path(__file__).parent / "cache"

# Global config
CURRENT_EMBEDDER = "gemini"  # or "bert"
_bert_tokenizer = None
_bert_model = None


def _get_bert_model():
    """Lazy load BERT model and tokenizer."""
    global _bert_tokenizer, _bert_model
    if not BERT_AVAILABLE:
        raise ImportError("transformers and torch are required for BERT embeddings.")
    
    if _bert_model is None:
        print(f"Loading BERT model: {BERT_MODEL_NAME}...")
        _bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        _bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME)
        # Move to GPU if available
        if torch.cuda.is_available():
            _bert_model = _bert_model.to("cuda")
        _bert_model.eval()
        print("BERT model loaded.")
    
    return _bert_tokenizer, _bert_model


def get_gemini_embedding(text: str, max_retries: int = 3) -> list[float]:
    """Get embedding for a single text using Gemini API."""
    retries = 0
    while retries < max_retries:
        try:
            response = client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=text
            )
            return response.embeddings[0].values
        except Exception as e:
            retries += 1
            print(f"Error getting embedding (Attempt {retries}/{max_retries}): {e}")
            if retries >= max_retries:
                raise e
            time.sleep(1 + random.random())  # Simple backoff


def get_bert_embedding(text: str) -> list[float]:
    """Get embedding for a single text using BERT."""
    tokenizer, model = _get_bert_model()
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    # Move inputs to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use mean pooling of last hidden states
    # Attention mask is needed to ignore padding tokens in the mean calculation
    token_embeddings = outputs.last_hidden_state # [batch_size, seq_len, hidden_size]
    attention_mask = inputs['attention_mask'] # [batch_size, seq_len]
    
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    mean_pooled = sum_embeddings / sum_mask
    return mean_pooled[0].cpu().numpy().tolist()


def get_embedding(text: str) -> list[float]:
    """Wrapper to get embedding based on CURRENT_EMBEDDER."""
    if CURRENT_EMBEDDER == "bert":
        return get_bert_embedding(text)
    else:
        return get_gemini_embedding(text)


def get_embeddings_batch(texts: list[str], labels: list = None, ids: list = None, cache_name: str = None) -> np.ndarray:
    """
    Get embeddings for a batch of texts.
    Optionally caches results to avoid re-computation.
    
    Args:
        texts: List of text strings to embed
        labels: Optional list of labels/categories corresponding to texts
        ids: Optional list of IDs corresponding to texts
        cache_name: Optional name for cache file
        
    Returns:
        numpy array of embeddings with shape (len(texts), embedding_dim)
    """
    import hashlib
    
    # Create hash of texts AND embedder type for cache validation
    # If we switch embedder, we must invalidate cache
    content_hash = hashlib.md5("".join(texts).encode()).hexdigest()[:16]
    unique_id = f"{content_hash}_{CURRENT_EMBEDDER}"
    
    # Check cache first
    if cache_name:
        CACHE_DIR.mkdir(exist_ok=True)
        # Append embedder name to cache file to avoid conflicts
        real_cache_name = f"{cache_name}_{CURRENT_EMBEDDER}"
        cache_path = CACHE_DIR / f"{real_cache_name}.pkl"
        
        if cache_path.exists():
            print(f"Loading embeddings from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                # Validate cache: check hash and length
                cached_embeddings = cached_data.get('embeddings', cached_data) if isinstance(cached_data, dict) else cached_data
                cached_hash = cached_data.get('hash', None) if isinstance(cached_data, dict) else None
                
                # Check if we need to invalidate because we want to save labels/ids but they aren't there
                cache_has_metadata = isinstance(cached_data, dict) and 'labels' in cached_data and 'ids' in cached_data
                need_metadata = labels is not None or ids is not None
                
                if cached_hash == unique_id and len(cached_embeddings) == len(texts):
                    if not need_metadata or cache_has_metadata:
                         print(f"  Cache valid (hash match)")
                         return np.array(cached_embeddings)
                    else:
                        print(f"  Cache valid but missing requested metadata (labels/ids). Regenerating to include them.")
                else:
                    print(f"  Cache invalid (hash mismatch or size mismatch), regenerating...")
    
    # Generate embeddings
    embeddings = []
    
    # Batch processing for BERT could be faster, but keeping it simple for now to match Gemini structure
    # For Gemini strict rate limits apply, for BERT we can go faster but loop is fine for now
    
    desc = f"Generating embeddings ({CURRENT_EMBEDDER})"
    for text in tqdm(texts, desc=desc):
        embedding = get_embedding(text)
        embeddings.append(embedding)
    
    embeddings_array = np.array(embeddings)
    
    # Save to cache with hash and metadata
    if cache_name:
        real_cache_name = f"{cache_name}_{CURRENT_EMBEDDER}"
        cache_path = CACHE_DIR / f"{real_cache_name}.pkl"
        data_to_save = {
            'embeddings': embeddings, 
            'hash': unique_id
        }
        if labels is not None:
            data_to_save['labels'] = labels
        if ids is not None:
            data_to_save['ids'] = ids
            
        with open(cache_path, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"Saved embeddings to cache: {cache_path}")
    
    return embeddings_array


def create_text_from_row(row: dict) -> str:
    """Create embedding text from a data row (Title + Abstract)."""
    import pandas as pd
    
    title = row.get('Title', '') or ''
    abstract = row.get('Abstract', '') or ''
    
    # Handle NaN values
    title = str(title) if not pd.isna(title) else ''
    abstract = str(abstract) if not pd.isna(abstract) else ''
    
    return f"Title: {title}\nAbstract: {abstract}"
