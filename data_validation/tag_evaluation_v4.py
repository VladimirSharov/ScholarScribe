import json
import numpy as np
from datetime import datetime
import os
from collections import defaultdict

# For character-level distance metrics
import Levenshtein
from jellyfish import jaro_winkler_similarity

# For multilingual embeddings
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

# Optional - for lightweight translation
# pip install transformers
# For CPU-only usage:
# pip install --no-deps easynmt

# Import translation module conditionally
try:
    from easynmt import EasyNMT
    HAS_TRANSLATION = True
except ImportError:
    HAS_TRANSLATION = False

# Initialize multilingual model (only once)
# model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Initialize lightweight translator if available
translator = None
if HAS_TRANSLATION:
    try:
        # This uses a smaller model suitable for CPU
        translator = EasyNMT('opus-mt', device='cpu')
        print("Translation model loaded successfully")
    except Exception as e:
        print(f"Could not load translation model: {e}")
        translator = None

def exact_match_ratio(original_tags, generated_tags):
    """Calculate exact match ratio between tag sets"""
    if not original_tags or not generated_tags:
        return 0.0
    
    original_set = set(tag.lower() for tag in original_tags)
    generated_set = set(tag.lower() for tag in generated_tags)
    
    matches = original_set.intersection(generated_set)
    return len(matches) / max(len(original_set), len(generated_set))

def fuzzy_match_ratio(original_tags, generated_tags, threshold=0.85):
    """Use Levenshtein distance for fuzzy matching with a threshold"""
    if not original_tags or not generated_tags:
        return 0.0
    
    matches = 0
    original_tags_lower = [tag.lower() for tag in original_tags]
    generated_tags_lower = [tag.lower() for tag in generated_tags]
    
    # For each original tag, find its best match in generated tags
    for orig_tag in original_tags_lower:
        best_ratio = 0
        for gen_tag in generated_tags_lower:
            # Calculate normalized Levenshtein similarity (1 - distance/max_length)
            max_len = max(len(orig_tag), len(gen_tag))
            if max_len == 0:  # Handle empty strings
                continue
            distance = Levenshtein.distance(orig_tag, gen_tag)
            ratio = 1 - (distance / max_len)
            best_ratio = max(best_ratio, ratio)
        
        # If best match exceeds threshold, count as match
        if best_ratio >= threshold:
            matches += 1
    
    return matches / max(len(original_tags), len(generated_tags))

def jaro_winkler_match_ratio(original_tags, generated_tags, threshold=0.85):
    """Use Jaro-Winkler similarity for fuzzy matching"""
    if not original_tags or not generated_tags:
        return 0.0
    
    matches = 0
    original_tags_lower = [tag.lower() for tag in original_tags]
    generated_tags_lower = [tag.lower() for tag in generated_tags]
    
    # For each original tag, find its best match in generated tags
    for orig_tag in original_tags_lower:
        best_ratio = 0
        for gen_tag in generated_tags_lower:
            ratio = jaro_winkler_similarity(orig_tag, gen_tag)
            best_ratio = max(best_ratio, ratio)
        
        # If best match exceeds threshold, count as match
        if best_ratio >= threshold:
            matches += 1
    
    return matches / max(len(original_tags), len(generated_tags))

def embedding_similarity(original_tags, generated_tags, model):
    """Calculate semantic similarity using multilingual embeddings"""
    if not original_tags or not generated_tags:
        return 0.0
    
    # Get embeddings for all tags
    original_embeddings = model.encode(original_tags)
    generated_embeddings = model.encode(generated_tags)
    
    # Calculate pairwise cosine similarities
    matches = 0
    match_threshold = 0.75  # Threshold for considering tags as matching
    
    for i, orig_embedding in enumerate(original_embeddings):
        # Find best matching generated tag for this original tag
        best_score = 0
        for gen_embedding in generated_embeddings:
            # Cosine similarity between normalized vectors
            similarity = np.dot(orig_embedding, gen_embedding) / (
                np.linalg.norm(orig_embedding) * np.linalg.norm(gen_embedding))
            best_score = max(best_score, similarity)
        
        # If best match exceeds threshold, count as match
        if best_score >= match_threshold:
            matches += 1
    
    return matches / max(len(original_tags), len(generated_tags))

def translate_tags(tags, target_lang='en', source_lang=None):
    """Translate tags to target language using lightweight translation"""
    if not translator or not tags:
        return tags
    
    try:
        # Attempt translation with timeout
        translated_tags = []
        for tag in tags:
            try:
                # Only translate if the tag appears to be non-English
                # This is a simple heuristic - you might want a more sophisticated check
                if any(ord(c) > 127 for c in tag) and len(tag) > 1:
                    result = translator.translate(tag, target_lang=target_lang, source_lang=source_lang)
                    translated_tags.append(result.lower())
                else:
                    translated_tags.append(tag.lower())
            except Exception:
                # If translation fails, keep original
                translated_tags.append(tag.lower())
                
        return translated_tags
    except Exception as e:
        print(f"Translation failed: {e}")
        return tags

def evaluate_tags_enhanced(results_file, use_translation=False, use_embeddings=False):
    """Evaluate tag generation with multiple metrics"""
    # Load data
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data.get('metadata', {})
    results = data.get('results', [])
    
    # Initialize models conditionally
    model = None
    if use_embeddings:
        try:
            model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            print("Embedding model loaded successfully")
        except Exception as e:
            print(f"Could not load embedding model: {e}")
            use_embeddings = False
    
    # Metrics storage
    metrics = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "source_file": results_file,
        "original_metadata": metadata,
        "summary_metrics": {
            "total_samples": len(results),
            "combined_tags": {
                "exact_match_ratio_avg": 0.0,
                "fuzzy_match_ratio_avg": 0.0,
                "jaro_winkler_match_ratio_avg": 0.0,
                "embedding_similarity_avg": 0.0 if use_embeddings else None,
                "perfect_exact_matches": 0,
                "zero_exact_matches": 0
            },
            "translated_tags": {
                "enabled": use_translation and translator is not None,
                "exact_match_ratio_avg": 0.0,
                "fuzzy_match_ratio_avg": 0.0,
                "jaro_winkler_match_ratio_avg": 0.0,
            }
        },
        "detailed_metrics": [],
        # Add language distribution analysis
        "language_distribution": defaultdict(int)
    }
    
    # Lists to store metrics for each item
    exact_ratios = []
    fuzzy_ratios = []
    jaro_ratios = []
    
    # Lists for translated metrics if translation is enabled
    exact_translated_ratios = []
    fuzzy_translated_ratios = []
    jaro_translated_ratios = []
    
    # List for embedding similarities if using embeddings
    embedding_similarities = []
    
    # Process each result item
    for item in results:
        # Get language if available
        language = item.get('language', 'unknown')
        metrics["language_distribution"][language] += 1
        
        # Combine original subject and additional tags
        original_subject = item.get('original_subject_tags', [])
        original_additional = item.get('original_additional_tags', [])
        original_combined = list(set(original_subject + original_additional))
        
        # Get generated tags
        generated_tags = item.get('generated_tags', [])
        
        # Translate tags if requested
        translated_original = original_combined
        translated_generated = generated_tags
        if use_translation and translator:
            # Detect language or use provided language
            lang = item.get('language', 'unknown')
            source_lang = None if lang == 'unknown' else lang
            
            # Translate both sets to English for comparison
            translated_original = translate_tags(original_combined, target_lang='en', source_lang=source_lang)
            translated_generated = translate_tags(generated_tags, target_lang='en', source_lang=source_lang)
        
        # Calculate metrics on original tags
        exact = exact_match_ratio(original_combined, generated_tags)
        fuzzy = fuzzy_match_ratio(original_combined, generated_tags)
        jaro = jaro_winkler_match_ratio(original_combined, generated_tags)
        
        # Calculate metrics on translated tags if translation was used
        exact_translated = exact
        fuzzy_translated = fuzzy
        jaro_translated = jaro
        if use_translation and translator:
            exact_translated = exact_match_ratio(translated_original, translated_generated)
            fuzzy_translated = fuzzy_match_ratio(translated_original, translated_generated)
            jaro_translated = jaro_winkler_match_ratio(translated_original, translated_generated)
        
        # Calculate embedding similarity if requested
        embedding_sim = 0.0
        if use_embeddings and model:
            embedding_sim = embedding_similarity(original_combined, generated_tags, model)
        
        # Store metrics
        exact_ratios.append(exact)
        fuzzy_ratios.append(fuzzy)
        jaro_ratios.append(jaro)
        
        # Use translated metrics lists if translation is enabled
        if use_translation and translator:
            exact_translated_ratios.append(exact_translated)
            fuzzy_translated_ratios.append(fuzzy_translated)
            jaro_translated_ratios.append(jaro_translated)
        
        # Store embedding similarities if using embeddings
        if use_embeddings and model:
            embedding_similarities.append(embedding_sim)
        
        # Track perfect and zero matches
        if exact == 1.0:
            metrics["summary_metrics"]["combined_tags"]["perfect_exact_matches"] += 1
        if exact == 0.0:
            metrics["summary_metrics"]["combined_tags"]["zero_exact_matches"] += 1
        
        # Store detailed metrics for this item
        item_metrics = {
            "identifier": item.get("identifier", ""),
            "language": language,
            "combined_tags_metrics": {
                "exact_match_ratio": exact,
                "fuzzy_match_ratio": fuzzy,
                "jaro_winkler_match_ratio": jaro,
                "original_count": len(original_combined),
                "generated_count": len(generated_tags),
                "original_tags": original_combined,
                "generated_tags": generated_tags
            }
        }
        
        # Add embedding similarity if enabled
        if use_embeddings and model:
            item_metrics["combined_tags_metrics"]["embedding_similarity"] = embedding_sim
            
        # Add translated metrics if translation is enabled
        if use_translation and translator:
            item_metrics["translated_tags_metrics"] = {
                "exact_match_ratio": exact_translated,
                "fuzzy_match_ratio": fuzzy_translated,
                "jaro_winkler_match_ratio": jaro_translated,
                "translated_original": translated_original,
                "translated_generated": translated_generated
            }
        metrics["detailed_metrics"].append(item_metrics)
    
    # Calculate averages for original metrics
    metrics["summary_metrics"]["combined_tags"]["exact_match_ratio_avg"] = np.mean(exact_ratios) if exact_ratios else 0
    metrics["summary_metrics"]["combined_tags"]["fuzzy_match_ratio_avg"] = np.mean(fuzzy_ratios) if fuzzy_ratios else 0
    metrics["summary_metrics"]["combined_tags"]["jaro_winkler_match_ratio_avg"] = np.mean(jaro_ratios) if jaro_ratios else 0
    
    # Calculate averages for translated metrics if translation is enabled
    if use_translation and translator:
        metrics["summary_metrics"]["translated_tags"]["exact_match_ratio_avg"] = np.mean(exact_translated_ratios) if exact_translated_ratios else 0
        metrics["summary_metrics"]["translated_tags"]["fuzzy_match_ratio_avg"] = np.mean(fuzzy_translated_ratios) if fuzzy_translated_ratios else 0
        metrics["summary_metrics"]["translated_tags"]["jaro_winkler_match_ratio_avg"] = np.mean(jaro_translated_ratios) if jaro_translated_ratios else 0
    
    # Calculate average for embedding similarity if embeddings are used
    if use_embeddings and model:
        metrics["summary_metrics"]["combined_tags"]["embedding_similarity_avg"] = np.mean(embedding_similarities) if embedding_similarities else 0
    
    # Add percentages
    total = len(results)
    metrics["summary_metrics"]["combined_tags"]["perfect_match_percentage"] = (metrics["summary_metrics"]["combined_tags"]["perfect_exact_matches"] / total) * 100 if total else 0
    metrics["summary_metrics"]["combined_tags"]["zero_match_percentage"] = (metrics["summary_metrics"]["combined_tags"]["zero_exact_matches"] / total) * 100 if total else 0
    
    # Convert language distribution counter to regular dict for JSON serialization
    metrics["language_distribution"] = dict(metrics["language_distribution"])
    
    # Add language-specific performance analysis
    language_metrics = {}
    for language in metrics["language_distribution"]:
        language_items = [item for item in metrics["detailed_metrics"] if item["language"] == language]
        if language_items:
            language_metrics[language] = {
                "count": len(language_items),
                "exact_match_avg": np.mean([item["combined_tags_metrics"]["exact_match_ratio"] for item in language_items]),
                "fuzzy_match_avg": np.mean([item["combined_tags_metrics"]["fuzzy_match_ratio"] for item in language_items]),
                "jaro_winkler_avg": np.mean([item["combined_tags_metrics"]["jaro_winkler_match_ratio"] for item in language_items]),
                # "embedding_avg": np.mean([item["combined_tags_metrics"]["embedding_similarity"] for item in language_items])  # Uncomment if using embeddings
            }
    
    metrics["language_specific_metrics"] = language_metrics
    
    return metrics

def get_distribution(detailed_metrics, value_func):
    """Get distribution of values in ranges from 0 to 1"""
    ranges = {
        "0.0": 0,
        "0.1-0.2": 0,
        "0.2-0.3": 0,
        "0.3-0.4": 0,
        "0.4-0.5": 0,
        "0.5-0.6": 0,
        "0.6-0.7": 0,
        "0.7-0.8": 0,
        "0.8-0.9": 0,
        "0.9-1.0": 0,
        "1.0": 0
    }
    
    for item in detailed_metrics:
        value = value_func(item)
        if value == 0.0:
            ranges["0.0"] += 1
        elif value == 1.0:
            ranges["1.0"] += 1
        else:
            # Find the appropriate range
            for i in range(1, 10):
                lower = i / 10
                upper = (i + 1) / 10
                if lower <= value < upper:
                    ranges[f"{lower:.1f}-{upper:.1f}"] += 1
                    break
    
    return ranges

def create_visualization_data(metrics, output_path):
    """Create a simplified JSON file for visualization"""
    viz_data = {
        "evaluation_date": metrics["evaluation_timestamp"],
        "model_name": metrics["original_metadata"].get("model", "Unknown"),
        "total_samples": metrics["summary_metrics"]["total_samples"],
        "language_distribution": metrics["language_distribution"],
        "metrics_summary": {
            "combined_tags": {
                "exact_match_average": metrics["summary_metrics"]["combined_tags"]["exact_match_ratio_avg"],
                "fuzzy_match_average": metrics["summary_metrics"]["combined_tags"]["fuzzy_match_ratio_avg"],
                "jaro_winkler_average": metrics["summary_metrics"]["combined_tags"]["jaro_winkler_match_ratio_avg"],
                # "embedding_similarity_average": metrics["summary_metrics"]["combined_tags"]["embedding_similarity_avg"],  # Uncomment if using embeddings
                "perfect_matches_percent": metrics["summary_metrics"]["combined_tags"]["perfect_match_percentage"],
                "zero_matches_percent": metrics["summary_metrics"]["combined_tags"]["zero_match_percentage"]
            }
        },
        "language_specific_performance": metrics["language_specific_metrics"],
        "distribution": {
            "exact_match": get_distribution(metrics["detailed_metrics"], 
                                          lambda x: x["combined_tags_metrics"]["exact_match_ratio"]),
            "fuzzy_match": get_distribution(metrics["detailed_metrics"], 
                                          lambda x: x["combined_tags_metrics"]["fuzzy_match_ratio"]),
            "jaro_winkler": get_distribution(metrics["detailed_metrics"], 
                                          lambda x: x["combined_tags_metrics"]["jaro_winkler_match_ratio"]),
            # "embedding_similarity": get_distribution(metrics["detailed_metrics"], 
            #                               lambda x: x["combined_tags_metrics"]["embedding_similarity"])  # Uncomment if using embeddings
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(viz_data, f, ensure_ascii=False, indent=2)
    print(f"Visualization data saved to {output_path}")

def main():
    # File paths
    input_file = "generated_tags_results.json"  # Your input file
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"tag_evaluation_{timestamp}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    metrics_output = os.path.join(output_dir, "tag_evaluation_metrics_enhanced.json")
    viz_output = os.path.join(output_dir, "tag_visualization_data_enhanced.json")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Please update the script with the correct path to your results file.")
        return
    
    # Check for available features
    use_translation = False
    use_embeddings = False
    
    # Check for translation capability
    if HAS_TRANSLATION and translator:
        print("Translation capability is available.")
        use_translation_input = input("Use translation for evaluation? (y/n): ").lower().strip()
        use_translation = use_translation_input == 'y'
    else:
        print("Translation is not available. To enable it, install EasyNMT with 'pip install --no-deps easynmt'")
    
    # Check for embedding capability
    try:
        import sentence_transformers
        print("Embedding capability is available.")
        use_embeddings_input = input("Use embeddings for semantic similarity? (y/n): ").lower().strip()
        use_embeddings = use_embeddings_input == 'y'
        if use_embeddings:
            print("Note: This might be slow on CPU-only systems")
    except ImportError:
        print("Embeddings are not available. To enable them, install SentenceTransformers with 'pip install sentence-transformers'")
    
    # Process data
    print(f"Evaluating tag generation in '{input_file}'...")
    metrics = evaluate_tags_enhanced(input_file, use_translation=use_translation, use_embeddings=use_embeddings)
    
    # Save metrics
    with open(metrics_output, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to {metrics_output}")
    
    # Create visualization data
    create_visualization_data(metrics, viz_output)
    
    # Print summary for quick reference
    print("\nEnhanced Evaluation Summary:")
    print("-" * 50)
    print(f"Total samples: {metrics['summary_metrics']['total_samples']}")
    
    # Language distribution
    print("\nLanguage Distribution:")
    for lang, count in metrics['language_distribution'].items():
        percentage = (count / metrics['summary_metrics']['total_samples']) * 100
        print(f"  {lang}: {count} samples ({percentage:.1f}%)")
    
    print("\nPerformance Metrics (Original Tags):")
    print(f"  Exact match ratio:      {metrics['summary_metrics']['combined_tags']['exact_match_ratio_avg']:.4f}")
    print(f"  Fuzzy match ratio:      {metrics['summary_metrics']['combined_tags']['fuzzy_match_ratio_avg']:.4f}")
    print(f"  Jaro-Winkler ratio:     {metrics['summary_metrics']['combined_tags']['jaro_winkler_match_ratio_avg']:.4f}")
    
    # Print embedding similarity if enabled
    if metrics['summary_metrics']['combined_tags']['embedding_similarity_avg'] is not None:
        print(f"  Embedding similarity:   {metrics['summary_metrics']['combined_tags']['embedding_similarity_avg']:.4f}")
        
    # Print translated metrics if translation was enabled
    if metrics['summary_metrics']['translated_tags']['enabled']:
        print("\nPerformance Metrics (After Translation to English):")
        print(f"  Exact match ratio:      {metrics['summary_metrics']['translated_tags']['exact_match_ratio_avg']:.4f}")
        print(f"  Fuzzy match ratio:      {metrics['summary_metrics']['translated_tags']['fuzzy_match_ratio_avg']:.4f}")
        print(f"  Jaro-Winkler ratio:     {metrics['summary_metrics']['translated_tags']['jaro_winkler_match_ratio_avg']:.4f}")
    
    print(f"\nPerfect matches: {metrics['summary_metrics']['combined_tags']['perfect_exact_matches']} " +
          f"({metrics['summary_metrics']['combined_tags']['perfect_match_percentage']:.1f}%)")
    print(f"Zero matches: {metrics['summary_metrics']['combined_tags']['zero_exact_matches']} " +
          f"({metrics['summary_metrics']['combined_tags']['zero_match_percentage']:.1f}%)")
    
    print("\nLanguage-Specific Performance:")
    for lang, lang_metrics in metrics["language_specific_metrics"].items():
        print(f"  {lang} ({lang_metrics['count']} samples):")
        print(f"    Exact match: {lang_metrics['exact_match_avg']:.4f}")
        print(f"    Fuzzy match: {lang_metrics['fuzzy_match_avg']:.4f}")
        print(f"    Jaro-Winkler: {lang_metrics['jaro_winkler_avg']:.4f}")
        # print(f"    Embedding: {lang_metrics['embedding_avg']:.4f}")  # Uncomment if using embeddings
    
    print("\nMetric Explanations:")
    print("  - Exact match: Direct string comparison (current approach)")
    print("  - Fuzzy match: Uses Levenshtein distance to find similar strings")
    print("  - Jaro-Winkler: Better for detecting typos and small variations")
    
    if metrics['summary_metrics']['combined_tags']['embedding_similarity_avg'] is not None:
        print("  - Embedding similarity: Semantic similarity using multilingual embeddings")
        
    if metrics['summary_metrics']['translated_tags']['enabled']:
        print("  - Translation-based metrics: Compares tags after translating to a common language (English)")
        
    print("\nPerformance Comparison:")
    print("  Higher values in translated metrics than original metrics indicate")
    print("  that your tags are semantically correct across languages even if")
    print("  they don't match exactly in their original form.")
    
    print(f"\nResults saved in: {output_dir}/")

if __name__ == "__main__":
    main()