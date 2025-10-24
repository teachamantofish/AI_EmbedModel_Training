"""
Tokenize triplets and save a tokenized dataset for fast training.

Config constants at top of file.
Produces:
 - tokenized dataset saved with datasets.save_to_disk(TOKENIZED_OUT)
 - metadata.json with tokenizer, max_length, padding, and token-length stats

Usage:
  .venv\Scripts\Activate.ps1
  python scripts/tokenize_triplets.py

"""
import os
from pathlib import Path
import json
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import sys
import functools
import shutil

from scripts.config_embed_training import *
from scripts.custom_logger import setup_global_logger

# Set up custom logger with CSV output to LOG_FILES directory
script_base = os.path.splitext(os.path.basename(__file__))[0]
LOG_HEADER = ["Date", "Level", "Message", "Test Step", "Result"]
logger = setup_global_logger(script_name=script_base, cwd=LOG_FILES, log_level='INFO', headers=LOG_HEADER)

# Use central config for paths and defaults so the tokenizer tokenizes the same data the trainer expects.
# Ensure repo root is on sys.path so we can import the top-level config module when running this script directly
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# ---------------------- CONFIG ----------------------
TRAIN_JSON = "triplets_train.json"  # file name expected inside TRAINING_DATA_DIR
# Use the TOKENIZED_DATA_DIR from config for output
TOKENIZED_OUT = Path(TOKENIZED_DATA_DIR)
MAX_LENGTH = 256
PADDING = "max_length"  # use 'max_length' for simplicity; consider 'longest' + DataCollator for dynamic padding
# NUM_PROC: use PREFETCH_WORKERS if present (safe fallback to 1 on Windows)
NUM_PROC = int(TRAINING_CONFIG.get("PREFETCH_WORKERS", 1) or 1)
BATCH_SIZE = 1000  # tokenization batching for speed
# ----------------------------------------------------

if __name__ == '__main__':
    # Accept optional environment override or CLI args for a fixed max length and output suffix
    import argparse
    parser = argparse.ArgumentParser(description="Tokenize triplets and optionally pad to a fixed length.")
    parser.add_argument('--fixed-length', type=int, default=None, help='If set, pad/truncate to this fixed length (overrides MAX_LENGTH).')
    parser.add_argument('--no-bucket', action='store_true', help='Disable bucketing and pad all examples to the fixed length.')
    parser.add_argument('--out-suffix', type=str, default=None, help='If set, append this suffix to the tokenized output folder name.')
    args = parser.parse_args()
    if args.fixed_length is not None:
        FIXED_MAX_LENGTH = int(args.fixed_length)
    else:
        FIXED_MAX_LENGTH = MAX_LENGTH
    USE_BUCKETING = not bool(args.no_bucket)
    # The canonical training JSON is expected under TRAINING_DATA_DIR (from config)
    data_path = Path(TRAINING_DATA_DIR) / TRAIN_JSON
    if not data_path.exists():
        # fallback: check repo-local embedding_training_data for convenience
        repo_root = Path(__file__).resolve().parents[1]
        repo_candidate = repo_root / "embedding_training_data" / TRAIN_JSON
        if repo_candidate.exists():
            data_path = repo_candidate
    if not data_path.exists():
        raise SystemExit(f"Could not find {TRAIN_JSON} at {data_path}. Place your train JSON there or update TRAIN_JSON or TRAINING_DATA_DIR in the config.")

    # tokenizer will be lazily created per-process by get_tokenizer(); create a small print here
    print(f"Tokenizer model (from config): {CONFIG_MODEL_NAME}")

# Lazily initialize tokenizer per process so datasets.map with multiprocessing works on Windows
_tokenizer = None
def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(CONFIG_MODEL_NAME)
    return _tokenizer


# Top-level preprocess function that is picklable by multiprocessing workers.
# EFFECTIVE_MAX_LENGTH will be set in __main__ before mapping so worker processes can read it
EFFECTIVE_MAX_LENGTH = MAX_LENGTH
USE_BUCKETING = True

# Bucket definitions (inclusive upper bounds). Anything > the last bound is "overflow".
BUCKET_BOUNDS = [16, 32, 48, 64, 84]


def _nearest_bucket_size(length: int):
    """Return bucket size for a given token length using BUCKET_BOUNDS.
    If length <= bound returns that bound. If length > last bound, round up to nearest 8."""
    for b in BUCKET_BOUNDS:
        if length <= b:
            return b
    # overflow: round up to nearest 8
    rem = length % 8
    return length if rem == 0 else length + (8 - rem)


def preprocess_examples_with_len(examples, **kwargs):
    anchors = examples.get('anchor') or examples.get('a') or examples.get('text')
    positives = examples.get('positive') or examples.get('p')
    negatives = examples.get('negative') or examples.get('n')
    out = {}
    tok = get_tokenizer()

    batch_bucket_sizes = []
    if USE_BUCKETING:
        # First compute unpadded token lengths for each sequence so we can bucket by the max
        a_tok_unpadded = tok(anchors, truncation=False, padding=False)
        p_tok_unpadded = tok(positives, truncation=False, padding=False)
        n_tok_unpadded = tok(negatives, truncation=False, padding=False)
        for i in range(len(anchors)):
            a_len = len(a_tok_unpadded['input_ids'][i]) if 'input_ids' in a_tok_unpadded else 0
            p_len = len(p_tok_unpadded['input_ids'][i]) if 'input_ids' in p_tok_unpadded else 0
            n_len = len(n_tok_unpadded['input_ids'][i]) if 'input_ids' in n_tok_unpadded else 0
            max_len = max(a_len, p_len, n_len)
            bucket_size = _nearest_bucket_size(max_len)
            batch_bucket_sizes.append(bucket_size)
    else:
        # No-bucket mode: use the global EFFECTIVE_MAX_LENGTH for all examples
        for i in range(len(anchors)):
            batch_bucket_sizes.append(int(EFFECTIVE_MAX_LENGTH))

    # Tokenize each example individually with its bucket size padding
    # This keeps batches small but respects per-example bucket padding
    anchor_input_ids = []
    anchor_attention_mask = []
    positive_input_ids = []
    positive_attention_mask = []
    negative_input_ids = []
    negative_attention_mask = []

    for i in range(len(anchors)):
        bs = batch_bucket_sizes[i]
        a_tok = tok(anchors[i], truncation=True, padding='max_length', max_length=bs)
        p_tok = tok(positives[i], truncation=True, padding='max_length', max_length=bs)
        n_tok = tok(negatives[i], truncation=True, padding='max_length', max_length=bs)

        anchor_input_ids.append(a_tok['input_ids'])
        anchor_attention_mask.append(a_tok.get('attention_mask'))
        positive_input_ids.append(p_tok['input_ids'])
        positive_attention_mask.append(p_tok.get('attention_mask'))
        negative_input_ids.append(n_tok['input_ids'])
        negative_attention_mask.append(n_tok.get('attention_mask'))

    out['anchor_input_ids'] = anchor_input_ids
    out['anchor_attention_mask'] = anchor_attention_mask
    out['positive_input_ids'] = positive_input_ids
    out['positive_attention_mask'] = positive_attention_mask
    out['negative_input_ids'] = negative_input_ids
    out['negative_attention_mask'] = negative_attention_mask
    # also return the bucket size per example so main flow can split per-bucket
    out['bucket_size'] = batch_bucket_sizes
    return out

if __name__ == '__main__':
    # load the JSON into datasets
    ds = load_dataset("json", data_files=str(data_path))
    # Ensure the module-level EFFECTIVE_MAX_LENGTH is set before mapping so worker processes use it
    print(f"Tokenizing dataset (this may take a while...), fixed_length={FIXED_MAX_LENGTH}")
    EFFECTIVE_MAX_LENGTH = FIXED_MAX_LENGTH
    tokenized = ds['train'].map(preprocess_examples_with_len, batched=True, batch_size=BATCH_SIZE, num_proc=NUM_PROC)

    # Split tokenized dataset into per-bucket datasets and save each padded to its bucket
    # Collect all unique bucket sizes present (tokenized['bucket_size'] is a per-example int)
    all_buckets = sorted(set(int(x) for x in tokenized['bucket_size']))
    print(f"Detected bucket sizes in dataset: {all_buckets}")

    # If bucketing is enabled, split tokenized dataset into per-bucket datasets and save each padded to its bucket
    if USE_BUCKETING:
        for bucket in all_buckets:
            mask = [int(bs) == bucket for bs in tokenized['bucket_size']]
            # Create a subset by selecting indices where mask is True
            subset_indices = [i for i, m in enumerate(mask) if m]
            subset = tokenized.select(subset_indices)
            out_dir = TOKENIZED_OUT / f"bucket_{bucket}"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving bucket {bucket} with {len(subset)} examples to {out_dir}")
            subset.save_to_disk(str(out_dir))
    else:
        print("Bucketing disabled: saving a single fixed-length padded tokenized dataset.")


    # compute length stats on input_ids per field (anchor only for simplicity)
    # Compute true (unpadded) token lengths. If attention masks are present use their sums
    # which count non-padded tokens; fallback to counting non-None tokens in input_ids.
    if 'anchor_attention_mask' in tokenized.column_names:
        anchor_lens = [int(sum(x)) for x in tokenized['anchor_attention_mask']]
    else:
        anchor_lens = [len([t for t in x if t is not None]) for x in tokenized['anchor_input_ids']]
    avg_len = float(np.mean(anchor_lens))
    p90 = int(np.percentile(anchor_lens, 90))
    p95 = int(np.percentile(anchor_lens, 95))

    # Ensure output directory from central config
    TOKENIZED_OUT = Path(TOKENIZED_OUT)
    if args.out_suffix:
        TOKENIZED_OUT = TOKENIZED_OUT.parent / (TOKENIZED_OUT.name + '_' + args.out_suffix)
    TOKENIZED_OUT.mkdir(parents=True, exist_ok=True)
    print(f"Saving tokenized dataset to {TOKENIZED_OUT}")
    tokenized.save_to_disk(str(TOKENIZED_OUT))

    # If we saved to a suffixed directory and the main TOKENIZED_DATA_DIR doesn't point there,
    # copy the suffixed tokenized folder into the canonical tokenized dir so finetuner finds it.
    canonical = Path(TOKENIZED_DATA_DIR)
    if args.out_suffix:
        if canonical.exists():
            print(f"Canonical tokenized dir {canonical} already exists; not overwriting.")
        else:
            print(f"Copying {TOKENIZED_OUT} to canonical tokenized dir {canonical}")
            shutil.copytree(str(TOKENIZED_OUT), str(canonical))

    metadata = {
        "tokenizer": CONFIG_MODEL_NAME,
        "max_length": FIXED_MAX_LENGTH,
        "padding": PADDING,
        "num_proc": NUM_PROC,
        "batch_size": BATCH_SIZE,
        "avg_anchor_length": avg_len,
        "p90_anchor_length": p90,
        "p95_anchor_length": p95,
        "anchor_lengths_are_unpadded_counts": True,
        "num_examples": len(tokenized),
    }
    with open(TOKENIZED_OUT / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Tokenization complete. Metadata:")
    print(json.dumps(metadata, indent=2))
