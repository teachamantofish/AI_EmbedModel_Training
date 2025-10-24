

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from itertools import combinations

from scripts.config_embed_training import BASE_CWD, DOCDIR, LOG_FILES
from scripts.custom_logger import setup_global_logger

# High-level flow (newbie friendly):
# 1. Load the chunk JSON file that was generated earlier in the pipeline.
# 2. Generate training triplets (anchor, positive, negative) from the chunks:
#    - Use hierarchical relationships (parent-child) for positive pairs
#    - Use semantic similarity (same heading paths, summaries) for positive pairs
#    - Generate THREE difficulty levels of negatives for each positive pair:
#      * EASY: Random chunks from completely different domains/topics
#      * MEDIUM: Chunks from similar topics but different contexts
#      * HARD: Chunks from same topic that look similar but aren't the answer
# 3. Export training data separated by difficulty into subdirectories:
#    - embedding_training_data/easy/  (for epoch 1 training)
#    - embedding_training_data/medium/  (for epoch 2 training)
#    - embedding_training_data/hard/  (for epoch 3 training)
# 4. Each difficulty level gets train/test splits for evaluation
# 5. Data is ready for curriculum learning with run_multi_epoch_training.py
#
# WHY DIFFICULTY LEVELS:
# - Curriculum learning: Train on easy examples first, then progressively harder
# - Easy negatives teach basic domain separation
# - Medium negatives teach topic-level distinctions
# - Hard negatives teach fine-grained semantic differences
# - This progressive approach often produces better final models than random negatives

chunkfile = BASE_CWD / DOCDIR / "a_chunks.json"

# Set up global logger with script-specific CSV header; overwrite existing log
script_base = os.path.splitext(os.path.basename(__file__))[0]
LOG_HEADER = ["Date", "Level", "Message", "Triplets Generated", "Training Data Type"]
logger = setup_global_logger(script_name=script_base, cwd=LOG_FILES, log_level='INFO', headers=LOG_HEADER)


# Configuration for training data generation
MIN_CONTENT_LENGTH = 50  # Minimum characters for content to be useful for training
MAX_CONTENT_LENGTH = 2000  # Maximum characters to avoid overwhelming the model
MIN_TRIPLETS_PER_CATEGORY = 100  # Minimum triplets per category/domain
MAX_TRIPLETS_PER_CATEGORY = 1000  # Maximum triplets per category to avoid bias
TRAIN_TEST_SPLIT = 0.8  # 80% training, 20% evaluation

# Difficulty levels for curriculum learning
# EASY: Random chunks from completely different domains/topics
# MEDIUM: Chunks from similar topics but different contexts
# HARD: Chunks from same topic that look similar but aren't the answer
GENERATE_DIFFICULTY_LEVELS = True  # Set to False to skip easy/medium/hard separation


def load_chunks() -> Tuple[List[Dict], Dict]:
    """Load chunks from JSON file and return chunks list plus metadata."""
    with open(chunkfile, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    
    if isinstance(loaded, dict) and "chunks" in loaded:
        chunks = loaded["chunks"]
        metadata = {k: v for k, v in loaded.items() if k != "chunks"}
    else:
        chunks = loaded
        metadata = {}
    
    logger.info(f"Loaded {len(chunks)} chunks from {chunkfile}")
    return chunks, metadata


def filter_chunks_for_training(chunks: List[Dict]) -> List[Dict]:
    """Filter chunks suitable for training based on content length and quality."""
    filtered = []
    
    for chunk in chunks:
        content = chunk.get("content", "").strip()
        
        # Skip chunks that are too short or too long
        if len(content) < MIN_CONTENT_LENGTH or len(content) > MAX_CONTENT_LENGTH:
            continue
            
        # Skip chunks without meaningful content
        if not content or content.lower() in ["", "n/a", "none", "null"]:
            continue
            
        # Prefer chunks with summaries as they tend to be higher quality
        chunk_summary = chunk.get("chunk_summary", "")
        if chunk_summary and chunk_summary.lower() != "false":
            chunk["has_summary"] = True
        else:
            chunk["has_summary"] = False
            
        filtered.append(chunk)
    
    logger.info(f"Filtered to {len(filtered)} chunks suitable for training")
    return filtered


def group_chunks_by_domain(chunks: List[Dict]) -> Dict[str, List[Dict]]:
    """Group chunks by domain/category for systematic positive/negative sampling."""
    groups = defaultdict(list)
    
    for chunk in chunks:
        # Primary grouping by category/domain
        category = chunk.get("category", "unknown").lower()
        title = chunk.get("title", "").lower()
        filename = chunk.get("filename", "unknown").lower()
        
        # Create multiple grouping strategies for better negative sampling
        # Strategy 1: Category + Title (primary domain)
        domain_key = f"{category}_{title}" if title else category
        groups[domain_key].append(chunk)
        
        # Strategy 2: Filename-based grouping for sub-domains
        if filename and filename != "unknown":
            file_key = f"file_{filename}"
            groups[file_key].append(chunk)
        
        # Strategy 3: Header-based grouping for topic separation
        header_path = chunk.get("concat_header_path", "")
        if header_path and "/" in header_path:
            # Use first two levels of header path as topic grouping
            header_parts = header_path.split("/")[:2]
            topic_key = f"topic_{'_'.join(header_parts)}"
            groups[topic_key].append(chunk)
        
        # Strategy 4: Content-length based grouping
        content_len = len(chunk.get("content", ""))
        if content_len < 200:
            groups["short_content"].append(chunk)
        elif content_len < 800:
            groups["medium_content"].append(chunk)
        else:
            groups["long_content"].append(chunk)
    
    # Remove groups that are too small to be useful
    filtered_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    
    logger.info(f"Grouped chunks into {len(filtered_groups)} domains: {list(filtered_groups.keys())[:10]}{'...' if len(filtered_groups) > 10 else ''}")
    return filtered_groups


def generate_positive_pairs(chunks: List[Dict]) -> List[Tuple[Dict, Dict, str]]:
    """Generate positive pairs from chunks using various similarity strategies."""
    positive_pairs = []
    
    # Strategy 1: Parent-child relationships
    parent_child_pairs = _generate_parent_child_pairs(chunks)
    positive_pairs.extend([(p, c, "parent_child") for p, c in parent_child_pairs])
    
    # Strategy 2: Same heading path (siblings)
    sibling_pairs = _generate_sibling_pairs(chunks)
    positive_pairs.extend([(s1, s2, "siblings") for s1, s2 in sibling_pairs])
    
    # Strategy 3: Similar summaries
    summary_pairs = _generate_summary_similarity_pairs(chunks)
    positive_pairs.extend([(s1, s2, "similar_summary") for s1, s2 in summary_pairs])
    
    # Strategy 4: Same document sections
    section_pairs = _generate_same_section_pairs(chunks)
    positive_pairs.extend([(s1, s2, "same_section") for s1, s2 in section_pairs])
    
    logger.info(f"Generated {len(positive_pairs)} positive pairs", 
                extra={"Triplets Generated": str(len(positive_pairs)), "Training Data Type": "positive_pairs"})
    return positive_pairs


def _generate_parent_child_pairs(chunks: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """Generate positive pairs from parent-child relationships."""
    id_to_chunk = {chunk["id"]: chunk for chunk in chunks if chunk.get("id")}
    pairs = []
    
    for chunk in chunks:
        parent_id = chunk.get("parent_id")
        if parent_id and parent_id in id_to_chunk:
            parent = id_to_chunk[parent_id]
            pairs.append((parent, chunk))
    
    return pairs


def _generate_sibling_pairs(chunks: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """Generate positive pairs from chunks with same header path (siblings)."""
    by_header_path = defaultdict(list)
    
    for chunk in chunks:
        header_path = chunk.get("concat_header_path", "")
        if header_path and len(header_path) > 10:  # Avoid very short/generic paths
            by_header_path[header_path].append(chunk)
    
    pairs = []
    for path, path_chunks in by_header_path.items():
        if len(path_chunks) >= 2:
            # Generate pairs within the same header path
            for chunk1, chunk2 in combinations(path_chunks[:5], 2):  # Limit to avoid explosion
                pairs.append((chunk1, chunk2))
    
    return pairs


def _generate_summary_similarity_pairs(chunks: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """Generate positive pairs from chunks with similar summaries."""
    summary_groups = defaultdict(list)
    
    for chunk in chunks:
        summary = chunk.get("chunk_summary", "").strip()
        if summary and summary.lower() != "false" and len(summary) > 20:
            # Group by first few words of summary to find similar concepts
            summary_key = " ".join(summary.lower().split()[:5])
            summary_groups[summary_key].append(chunk)
    
    pairs = []
    for key, group_chunks in summary_groups.items():
        if len(group_chunks) >= 2:
            for chunk1, chunk2 in combinations(group_chunks[:3], 2):  # Limit combinations
                pairs.append((chunk1, chunk2))
    
    return pairs


def _generate_same_section_pairs(chunks: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """Generate positive pairs from chunks in the same document section."""
    by_section = defaultdict(list)
    
    for chunk in chunks:
        filename = chunk.get("filename", "")
        # Use first part of header path as section identifier
        header_path = chunk.get("concat_header_path", "")
        if header_path and "/" in header_path:
            section = header_path.split("/")[0]
            section_key = f"{filename}:{section}"
            by_section[section_key].append(chunk)
    
    pairs = []
    for section_key, section_chunks in by_section.items():
        if len(section_chunks) >= 2:
            for chunk1, chunk2 in combinations(section_chunks[:4], 2):
                pairs.append((chunk1, chunk2))
    
    return pairs


def generate_negatives_by_difficulty(anchor: Dict, positive: Dict, 
                                   all_chunks: List[Dict], 
                                   domain_groups: Dict[str, List[Dict]]) -> Tuple[Dict, Dict, Dict]:
    """
    Generate easy, medium, and hard negatives for the same anchor/positive pair.
    
    EASY NEGATIVES: Random chunks from completely different domains
    - Different category entirely
    - Different file/document
    - Minimal topic overlap
    
    MEDIUM NEGATIVES: Chunks from similar topics but different contexts
    - Same general category but different subtopics
    - Different sections of documentation
    - Moderate semantic distance
    
    HARD NEGATIVES: Confusing chunks that look similar but aren't the answer
    - Same topic/category
    - Similar header paths or keywords
    - High semantic similarity but wrong context
    
    Returns: (easy_negative, medium_negative, hard_negative)
    """
    
    anchor_category = anchor.get("category", "unknown").lower()
    anchor_title = anchor.get("title", "").lower()
    anchor_domain = f"{anchor_category}_{anchor_title}" if anchor_title else anchor_category
    anchor_path = anchor.get("concat_header_path", "").lower()
    anchor_filename = anchor.get("filename", "")
    anchor_content_words = set(anchor.get("content", "").lower().split()[:30])
    
    # EASY NEGATIVE: Maximum distance - completely different domain
    easy_negative = _select_easy_negative(anchor, anchor_domain, domain_groups, all_chunks)
    
    # MEDIUM NEGATIVE: Moderate distance - similar domain, different context
    medium_negative = _select_medium_negative(anchor, anchor_path, anchor_filename, 
                                              anchor_content_words, all_chunks, positive)
    
    # HARD NEGATIVE: Minimum distance - same topic, confusingly similar
    hard_negative = _select_hard_negative(anchor, anchor_path, anchor_content_words, 
                                         all_chunks, positive, easy_negative, medium_negative)
    
    return easy_negative, medium_negative, hard_negative


def _select_easy_negative(anchor: Dict, anchor_domain: str, 
                         domain_groups: Dict[str, List[Dict]], 
                         all_chunks: List[Dict]) -> Dict:
    """Select an easy negative: completely different domain/topic."""
    # Try to find chunks from a different domain
    if len(domain_groups) > 1:
        different_domains = [d for d in domain_groups.keys() 
                           if d != anchor_domain and not d.startswith("file_") 
                           and not d.startswith("topic_") and not d.startswith("short_")
                           and not d.startswith("medium_") and not d.startswith("long_")]
        
        if different_domains:
            neg_domain = random.choice(different_domains)
            if domain_groups[neg_domain]:
                return random.choice(domain_groups[neg_domain])
    
    # Fallback: different filename
    different_file_chunks = [c for c in all_chunks 
                           if c.get("filename") != anchor.get("filename")]
    if different_file_chunks:
        return random.choice(different_file_chunks)
    
    # Ultimate fallback: any random chunk
    other_chunks = [c for c in all_chunks if c.get("id") != anchor.get("id")]
    return random.choice(other_chunks) if other_chunks else all_chunks[0]


def _select_medium_negative(anchor: Dict, anchor_path: str, anchor_filename: str,
                           anchor_content_words: Set[str], all_chunks: List[Dict],
                           positive: Dict) -> Dict:
    """Select a medium negative: similar topic but different context."""
    candidates = []
    
    for chunk in all_chunks:
        chunk_id = chunk.get("id")
        # Skip anchor and positive
        if chunk_id == anchor.get("id") or chunk_id == positive.get("id"):
            continue
        
        chunk_path = chunk.get("concat_header_path", "").lower()
        chunk_filename = chunk.get("filename", "")
        
        # Look for chunks with some path similarity but not too much
        if anchor_path and chunk_path:
            anchor_parts = set(anchor_path.split("/"))
            chunk_parts = set(chunk_path.split("/"))
            overlap = len(anchor_parts.intersection(chunk_parts))
            total = len(anchor_parts.union(chunk_parts))
            path_similarity = overlap / total if total > 0 else 0
            
            # Medium difficulty: 20-50% path overlap
            if 0.2 <= path_similarity <= 0.5:
                candidates.append(chunk)
            # Also accept same file but different section
            elif chunk_filename == anchor_filename and path_similarity < 0.3:
                candidates.append(chunk)
    
    if candidates:
        return random.choice(candidates)
    
    # Fallback: chunks with moderate content similarity
    moderate_similarity_chunks = []
    for chunk in all_chunks:
        if chunk.get("id") not in [anchor.get("id"), positive.get("id")]:
            chunk_words = set(chunk.get("content", "").lower().split()[:30])
            if anchor_content_words and chunk_words:
                overlap = len(anchor_content_words.intersection(chunk_words))
                similarity = overlap / len(anchor_content_words.union(chunk_words))
                # 15-40% content similarity
                if 0.15 <= similarity <= 0.4:
                    moderate_similarity_chunks.append(chunk)
    
    if moderate_similarity_chunks:
        return random.choice(moderate_similarity_chunks)
    
    # Ultimate fallback: any different chunk
    other_chunks = [c for c in all_chunks 
                   if c.get("id") not in [anchor.get("id"), positive.get("id")]]
    return random.choice(other_chunks) if other_chunks else all_chunks[0]


def _select_hard_negative(anchor: Dict, anchor_path: str, 
                         anchor_content_words: Set[str], all_chunks: List[Dict],
                         positive: Dict, easy_neg: Dict, medium_neg: Dict) -> Dict:
    """Select a hard negative: same topic, confusingly similar."""
    candidates = []
    
    for chunk in all_chunks:
        chunk_id = chunk.get("id")
        # Skip anchor, positive, and already selected negatives
        if chunk_id in [anchor.get("id"), positive.get("id"), 
                       easy_neg.get("id"), medium_neg.get("id")]:
            continue
        
        chunk_path = chunk.get("concat_header_path", "").lower()
        chunk_words = set(chunk.get("content", "").lower().split()[:30])
        
        # Calculate similarities
        path_similarity = 0
        if anchor_path and chunk_path:
            anchor_parts = set(anchor_path.split("/"))
            chunk_parts = set(chunk_path.split("/"))
            overlap = len(anchor_parts.intersection(chunk_parts))
            total = len(anchor_parts.union(chunk_parts))
            path_similarity = overlap / total if total > 0 else 0
        
        content_similarity = 0
        if anchor_content_words and chunk_words:
            overlap = len(anchor_content_words.intersection(chunk_words))
            content_similarity = overlap / len(anchor_content_words.union(chunk_words))
        
        # Hard negative: high path similarity (>50%) OR high content similarity (>40%)
        if path_similarity > 0.5 or content_similarity > 0.4:
            # Score by combined similarity (higher is harder)
            score = path_similarity * 0.6 + content_similarity * 0.4
            candidates.append((chunk, score))
    
    if candidates:
        # Sort by score and pick from top candidates (most confusing)
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [c[0] for c in candidates[:min(5, len(candidates))]]
        return random.choice(top_candidates)
    
    # Fallback: same file/category
    same_category_chunks = [c for c in all_chunks 
                           if c.get("category") == anchor.get("category")
                           and c.get("id") not in [anchor.get("id"), positive.get("id"),
                                                   easy_neg.get("id"), medium_neg.get("id")]]
    if same_category_chunks:
        return random.choice(same_category_chunks)
    
    # Ultimate fallback: any different chunk
    other_chunks = [c for c in all_chunks 
                   if c.get("id") not in [anchor.get("id"), positive.get("id"),
                                         easy_neg.get("id"), medium_neg.get("id")]]
    return random.choice(other_chunks) if other_chunks else all_chunks[0]


def generate_negative_pairs(positive_pairs: List[Tuple[Dict, Dict, str]], 
                          all_chunks: List[Dict], 
                          domain_groups: Dict[str, List[Dict]]) -> List[Tuple[Dict, Dict, str]]:
    """Generate negative pairs using multiple strategies for robust training."""
    negative_pairs = []
    
    for anchor, positive, pair_type in positive_pairs:
        negatives_found = 0
        
        # Strategy 1: Different domain (if multiple domains exist)
        if len(domain_groups) > 1:
            anchor_category = anchor.get("category", "unknown").lower()
            anchor_title = anchor.get("title", "").lower()
            anchor_domain = f"{anchor_category}_{anchor_title}" if anchor_title else anchor_category
            
            candidate_domains = [d for d in domain_groups.keys() if d != anchor_domain]
            if candidate_domains:
                for _ in range(min(NEGATIVE_SAMPLING_RATIO, len(candidate_domains))):
                    neg_domain = random.choice(candidate_domains)
                    if domain_groups[neg_domain]:
                        negative = random.choice(domain_groups[neg_domain])
                        negative_pairs.append((anchor, negative, f"negative_cross_domain_{pair_type}"))
                        negatives_found += 1
        
        # Strategy 2: Different filename/document  
        if negatives_found < NEGATIVE_SAMPLING_RATIO:
            different_file_chunks = [c for c in all_chunks 
                                   if c.get("filename") != anchor.get("filename")]
            if different_file_chunks:
                remaining_needed = NEGATIVE_SAMPLING_RATIO - negatives_found
                for _ in range(remaining_needed):
                    negative = random.choice(different_file_chunks)
                    negative_pairs.append((anchor, negative, f"negative_diff_file_{pair_type}"))
                    negatives_found += 1
        
        # Strategy 3: Different header path (semantic distance)
        if negatives_found < NEGATIVE_SAMPLING_RATIO:
            anchor_path = anchor.get("concat_header_path", "").lower()
            different_path_chunks = []
            
            for chunk in all_chunks:
                chunk_path = chunk.get("concat_header_path", "").lower()
                # Skip if same path or very similar path
                if chunk_path != anchor_path and chunk.get("id") != anchor.get("id"):
                    # Calculate simple path dissimilarity
                    if anchor_path and chunk_path:
                        anchor_parts = set(anchor_path.split("/"))
                        chunk_parts = set(chunk_path.split("/"))
                        overlap = len(anchor_parts.intersection(chunk_parts))
                        total = len(anchor_parts.union(chunk_parts))
                        similarity = overlap / total if total > 0 else 0
                        
                        # Use chunks with low path similarity as negatives
                        if similarity < 0.3:  # Less than 30% path overlap
                            different_path_chunks.append(chunk)
            
            if different_path_chunks:
                remaining_needed = NEGATIVE_SAMPLING_RATIO - negatives_found
                for _ in range(min(remaining_needed, len(different_path_chunks))):
                    negative = random.choice(different_path_chunks)
                    negative_pairs.append((anchor, negative, f"negative_diff_path_{pair_type}"))
                    negatives_found += 1
        
        # Strategy 4: Fallback - random sampling with content dissimilarity
        if negatives_found < NEGATIVE_SAMPLING_RATIO:
            anchor_words = set(anchor.get("content", "").lower().split()[:20])  # First 20 words
            dissimilar_chunks = []
            
            for chunk in all_chunks:
                if chunk.get("id") != anchor.get("id") and chunk.get("id") != positive.get("id"):
                    chunk_words = set(chunk.get("content", "").lower().split()[:20])
                    if anchor_words and chunk_words:
                        overlap = len(anchor_words.intersection(chunk_words))
                        similarity = overlap / len(anchor_words.union(chunk_words))
                        
                        # Use chunks with low content similarity
                        if similarity < 0.2:  # Less than 20% word overlap
                            dissimilar_chunks.append(chunk)
            
            if dissimilar_chunks:
                remaining_needed = NEGATIVE_SAMPLING_RATIO - negatives_found
                for _ in range(min(remaining_needed, len(dissimilar_chunks))):
                    negative = random.choice(dissimilar_chunks)
                    negative_pairs.append((anchor, negative, f"negative_dissimilar_{pair_type}"))
                    negatives_found += 1
        
        # Strategy 5: Ultimate fallback - completely random (avoid identical chunks)
        if negatives_found < NEGATIVE_SAMPLING_RATIO:
            available_chunks = [c for c in all_chunks 
                              if c.get("id") not in [anchor.get("id"), positive.get("id")]]
            if available_chunks:
                remaining_needed = NEGATIVE_SAMPLING_RATIO - negatives_found
                for _ in range(min(remaining_needed, len(available_chunks))):
                    negative = random.choice(available_chunks)
                    negative_pairs.append((anchor, negative, f"negative_random_{pair_type}"))
                    negatives_found += 1
    
    logger.info(f"Generated {len(negative_pairs)} negative pairs using multiple strategies",
                extra={"Triplets Generated": str(len(negative_pairs)), "Training Data Type": "negative_pairs"})
    return negative_pairs


def create_training_triplets_by_difficulty(positive_pairs: List[Tuple[Dict, Dict, str]], 
                                          all_chunks: List[Dict],
                                          domain_groups: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """
    Create training triplets with easy, medium, and hard negatives for curriculum learning.
    
    Returns dictionary with keys: 'easy', 'medium', 'hard'
    Each contains list of triplets with appropriate difficulty negatives.
    """
    triplets_by_difficulty = {
        'easy': [],
        'medium': [],
        'hard': []
    }
    
    logger.info(f"Generating triplets with difficulty levels for {len(positive_pairs)} positive pairs")
    
    for anchor, positive, pos_type in positive_pairs:
        # Generate all three difficulty levels for this pair
        easy_neg, medium_neg, hard_neg = generate_negatives_by_difficulty(
            anchor, positive, all_chunks, domain_groups
        )
        
        # Create easy triplet
        easy_triplet = {
            "anchor": _extract_text_for_training(anchor),
            "positive": _extract_text_for_training(positive),
            "negative": _extract_text_for_training(easy_neg),
            "anchor_id": anchor.get("id"),
            "positive_id": positive.get("id"),
            "negative_id": easy_neg.get("id"),
            "pair_type": pos_type,
            "difficulty": "easy",
            "negative_type": "cross_domain",
            "anchor_domain": f"{anchor.get('category', '')}_{anchor.get('title', '')}",
            "positive_domain": f"{positive.get('category', '')}_{positive.get('title', '')}",
            "negative_domain": f"{easy_neg.get('category', '')}_{easy_neg.get('title', '')}"
        }
        triplets_by_difficulty['easy'].append(easy_triplet)
        
        # Create medium triplet
        medium_triplet = {
            "anchor": _extract_text_for_training(anchor),
            "positive": _extract_text_for_training(positive),
            "negative": _extract_text_for_training(medium_neg),
            "anchor_id": anchor.get("id"),
            "positive_id": positive.get("id"),
            "negative_id": medium_neg.get("id"),
            "pair_type": pos_type,
            "difficulty": "medium",
            "negative_type": "similar_topic",
            "anchor_domain": f"{anchor.get('category', '')}_{anchor.get('title', '')}",
            "positive_domain": f"{positive.get('category', '')}_{positive.get('title', '')}",
            "negative_domain": f"{medium_neg.get('category', '')}_{medium_neg.get('title', '')}"
        }
        triplets_by_difficulty['medium'].append(medium_triplet)
        
        # Create hard triplet
        hard_triplet = {
            "anchor": _extract_text_for_training(anchor),
            "positive": _extract_text_for_training(positive),
            "negative": _extract_text_for_training(hard_neg),
            "anchor_id": anchor.get("id"),
            "positive_id": positive.get("id"),
            "negative_id": hard_neg.get("id"),
            "pair_type": pos_type,
            "difficulty": "hard",
            "negative_type": "confusingly_similar",
            "anchor_domain": f"{anchor.get('category', '')}_{anchor.get('title', '')}",
            "positive_domain": f"{positive.get('category', '')}_{positive.get('title', '')}",
            "negative_domain": f"{hard_neg.get('category', '')}_{hard_neg.get('title', '')}"
        }
        triplets_by_difficulty['hard'].append(hard_triplet)
    
    for difficulty, triplets in triplets_by_difficulty.items():
        logger.info(f"Created {len(triplets)} {difficulty} triplets",
                   extra={"Triplets Generated": str(len(triplets)), 
                         "Training Data Type": f"{difficulty}_triplets"})
    
    return triplets_by_difficulty


def _extract_text_for_training(chunk: Dict) -> str:
    """Extract the best text representation for training from a chunk."""
    content = chunk.get("content", "").strip()
    summary = chunk.get("chunk_summary", "").strip()
    heading = chunk.get("concat_header_path", "").strip()
    
    # Prefer summary if available and not a placeholder
    if summary and summary.lower() != "false" and len(summary) > 20:
        if heading:
            return f"{heading}: {summary}"
        return summary
    
    # Fallback to content with optional heading
    if heading and len(content) > 100:
        return f"{heading}: {content}"
    
    return content


def export_training_data_by_difficulty(triplets_by_difficulty: Dict[str, List[Dict]], 
                                      base_output_dir: Path) -> None:
    """
    Export training data separated by difficulty level for curriculum learning.
    
    Creates subdirectories: easy/, medium/, hard/
    Each contains train/test splits in JSON format for the orchestrator to use.
    """
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    for difficulty, triplets in triplets_by_difficulty.items():
        if not triplets:
            logger.warning(f"No {difficulty} triplets to export")
            continue
        
        # Create difficulty-specific subdirectory
        difficulty_dir = base_output_dir / difficulty
        difficulty_dir.mkdir(parents=True, exist_ok=True)
        
        # Split into train/test
        random.shuffle(triplets)
        split_idx = int(len(triplets) * TRAIN_TEST_SPLIT)
        train_triplets = triplets[:split_idx]
        test_triplets = triplets[split_idx:]
        
        # Export in JSON format (for tokenizer compatibility)
        _export_json_format(train_triplets, difficulty_dir / "triplets_train.json")
        _export_json_format(test_triplets, difficulty_dir / "triplets_test.json")
        
        # Export statistics
        _export_statistics(triplets, difficulty_dir / "statistics.json")
        
        logger.info(f"Exported {difficulty} data to {difficulty_dir}: "
                   f"{len(train_triplets)} train, {len(test_triplets)} test",
                   extra={"Triplets Generated": str(len(triplets)), 
                         "Training Data Type": f"{difficulty}_export"})


def _export_json_format(triplets: List[Dict], output_path: Path) -> None:
    """Export triplets in JSON format for the training pipeline."""
    # Simplify triplets to just the essential fields for training
    simplified = []
    for t in triplets:
        simplified.append({
            "anchor": t["anchor"],
            "positive": t["positive"],
            "negative": t["negative"],
            "difficulty": t.get("difficulty", "unknown"),
            "pair_type": t.get("pair_type", "unknown")
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(simplified, f, indent=2, ensure_ascii=False)


def _export_statistics(triplets: List[Dict], output_path: Path) -> None:
    """Export training data statistics."""
    stats = {
        "total_triplets": len(triplets),
        "difficulty": triplets[0].get("difficulty", "unknown") if triplets else "unknown",
        "pair_types": {},
        "domains": {},
        "avg_text_lengths": {}
    }
    
    # Analyze pair types
    for triplet in triplets:
        pair_type = triplet.get("pair_type", "unknown")
        stats["pair_types"][pair_type] = stats["pair_types"].get(pair_type, 0) + 1
    
    # Analyze domains
    for triplet in triplets:
        domain = triplet.get("anchor_domain", "unknown")
        stats["domains"][domain] = stats["domains"].get(domain, 0) + 1
    
    # Analyze text lengths
    anchor_lengths = [len(t["anchor"]) for t in triplets]
    positive_lengths = [len(t["positive"]) for t in triplets]
    negative_lengths = [len(t["negative"]) for t in triplets]
    
    stats["avg_text_lengths"] = {
        "anchor": sum(anchor_lengths) / len(anchor_lengths) if anchor_lengths else 0,
        "positive": sum(positive_lengths) / len(positive_lengths) if positive_lengths else 0,
        "negative": sum(negative_lengths) / len(negative_lengths) if negative_lengths else 0
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def main():
    """Main function to generate training data for embedding model fine-tuning."""
    logger.info("Starting embedding model training data generation with difficulty levels")
    
    # Load and filter chunks
    chunks, metadata = load_chunks()
    filtered_chunks = filter_chunks_for_training(chunks)
    
    if len(filtered_chunks) < 100:
        logger.warning(f"Only {len(filtered_chunks)} chunks available - may not be sufficient for training")
    
    # Group by domain for systematic sampling
    domain_groups = group_chunks_by_domain(filtered_chunks)
    
    # Generate positive pairs
    positive_pairs = generate_positive_pairs(filtered_chunks)
    
    if not positive_pairs:
        logger.error("No positive pairs generated - cannot create training data")
        return
    
    logger.info(f"Generated {len(positive_pairs)} positive pairs")
    
    # Create triplets with easy, medium, and hard negatives for curriculum learning
    triplets_by_difficulty = create_training_triplets_by_difficulty(
        positive_pairs, filtered_chunks, domain_groups
    )
    
    total_triplets = sum(len(t) for t in triplets_by_difficulty.values())
    if total_triplets == 0:
        logger.error("No triplets created - check positive/negative pair generation")
        return
    
    # Export training data by difficulty level
    output_dir = BASE_CWD / DOCDIR / "embedding_training_data"
    export_training_data_by_difficulty(triplets_by_difficulty, output_dir)
    
    logger.info(f"Training data generation complete:")
    logger.info(f"  Easy: {len(triplets_by_difficulty['easy'])} triplets")
    logger.info(f"  Medium: {len(triplets_by_difficulty['medium'])} triplets")
    logger.info(f"  Hard: {len(triplets_by_difficulty['hard'])} triplets")
    logger.info(f"  Total: {total_triplets} triplets exported to {output_dir}")
    logger.info(f"Data ready for curriculum learning with run_multi_epoch_training.py")


if __name__ == "__main__":
    main()
