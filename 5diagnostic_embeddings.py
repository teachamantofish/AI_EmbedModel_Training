# Diagnostic script to check fine-tuned embedding model
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

from scripts.config_embed_training import BASE_CWD, DOCDIR, LOG_FILES
from scripts.custom_logger import setup_global_logger

# Load the fine-tuned model
model_path = OUTPUT_MODEL_PATH
print(f"Loading model from: {model_path}")

try:
    model = SentenceTransformer(str(model_path))
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    exit(1)

# Test with simple text
test_texts = [
    "Function Summary",
    "Object Reference", 
    "CMS Connector"
]

print(f"\nTesting with {len(test_texts)} simple texts...")

# Generate embeddings
embeddings = model.encode(test_texts)
print(f"Embedding shape: {embeddings.shape}")
print(f"Embedding dtype: {embeddings.dtype}")

# Check for problematic values
print(f"Contains NaN: {np.isnan(embeddings).any()}")
print(f"Contains Inf: {np.isinf(embeddings).any()}")
print(f"All zeros: {np.allclose(embeddings, 0)}")

# Check embedding statistics
print(f"Min value: {np.min(embeddings)}")
print(f"Max value: {np.max(embeddings)}")
print(f"Mean: {np.mean(embeddings)}")
print(f"Std: {np.std(embeddings)}")

# Check individual embedding norms
norms = np.linalg.norm(embeddings, axis=1)
print(f"Embedding norms: {norms}")

# Try manual cosine similarity
if not np.isnan(embeddings).any() and not np.allclose(norms, 0):
    # Normalize embeddings
    embeddings_normalized = embeddings / norms[:, np.newaxis]
    
    # Manual cosine similarity
    similarity_manual = np.dot(embeddings_normalized, embeddings_normalized.T)
    print(f"Manual similarity matrix:\n{similarity_manual}")
else:
    print("Cannot compute similarity - embeddings contain NaN or are zero-magnitude")

# Test with original Qwen model for comparison
print("\n" + "="*50)
print("Testing original Qwen model for comparison...")

try:
    original_model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")
    original_embeddings = original_model.encode(test_texts)
    print(f"Original embedding shape: {original_embeddings.shape}")
    print(f"Original contains NaN: {np.isnan(original_embeddings).any()}")
    
    original_norms = np.linalg.norm(original_embeddings, axis=1)
    print(f"Original embedding norms: {original_norms}")
    
    if not np.isnan(original_embeddings).any():
        from sentence_transformers.util import cos_sim
        original_sim = cos_sim(original_embeddings, original_embeddings)
        print(f"Original similarity matrix:\n{original_sim.numpy()}")
        
except Exception as e:
    print(f"Could not load original model: {e}")