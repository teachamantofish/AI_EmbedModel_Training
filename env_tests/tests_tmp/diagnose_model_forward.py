"""
Quick diagnostic: load the local SentenceTransformer model and run a tiny forward
on sample texts to verify whether embeddings contain NaNs/Inf on CPU and GPU.

Run from project root:
.\.venv\Scripts\Activate.ps1
python diagnose_model_forward.py
"""
import torch
from sentence_transformers import SentenceTransformer
from config_embed_training import MODEL_NAME

print(f"MODEL_NAME: {MODEL_NAME}")

texts = [
    "Function test",
    "Object test",
    "Parameter test"
]

for dev_name in ['cpu', 'cuda']:
    try:
        device = torch.device(dev_name) if torch.cuda.is_available() or dev_name == 'cpu' else None
    except Exception:
        device = None
    if device is None:
        print(f"Skipping device {dev_name}")
        continue
    print(f"\nTesting device: {device}")
    try:
        model = SentenceTransformer(str(MODEL_NAME))
        print("Model loaded")
        try:
            model = model.to(device)
            print(f"Moved model to {device}")
        except Exception as e:
            print(f"Could not move model to device: {e}")
        # forward
        with torch.no_grad():
            emb = model.encode(texts, convert_to_tensor=True)
            try:
                emb = emb.to(device)
            except Exception:
                pass
            print(f"Embeddings shape: {emb.shape}")
            try:
                import torch
                print(f"isfinite: {torch.isfinite(emb).all().item()}, any_nan: {torch.isnan(emb).any().item()}")
                print(f" mean={emb.mean().item()}, std={emb.std().item()}, min={emb.min().item()}, max={emb.max().item()}")
            except Exception as e:
                print(f"Diagnostics failed: {e}")
    except Exception as e:
        print(f"Failed on device {device}: {e}")

print("Done")