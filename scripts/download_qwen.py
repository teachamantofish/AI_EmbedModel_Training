from sentence_transformers import SentenceTransformer
from pathlib import Path
# For example, get the model ID from https://github.com/QwenLM/Qwen3-Embedding
MODEL_ID = 'cross-encoder/ms-marco-MiniLM-L12-v2'
OUT = Path(r'C:/GIT/AI_DataSource/framemaker/ms-marco-MiniLM-L-12-v2')
print('Downloading model:', MODEL_ID)
model = SentenceTransformer(MODEL_ID)
print('Saving to:', OUT)
model.save(str(OUT))
print('Done')
