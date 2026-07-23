import json
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wasserstein_distance
from collections import Counter
import os

def load_results(filepath):
    """Load the generated tags results from JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: File {filepath} is not valid JSON.")
        return None

def detect_language(text_list):
    """Simple language detection based on common words."""
    if not text_list or len(text_list) == 0:
        return "unknown"
    
    # Simple language detection based on common words
    finnish_words = {"ja", "on", "ei", "se", "että", "koulun", "oppilaat", "opetus"}
    english_words = {"and", "the", "of", "to", "in", "is", "school", "learning"}
    
    text = " ".join(text_list).lower()
    
    finnish_count = sum(1 for word in finnish_words if word in text)
    english_count = sum(1 for word in english_words if word in text)
    
    if finnish_count > english_count:
        return "finnish"
    elif english_count > finnish_count:
        return "english"
    else:
        return "unknown"

def calculate_emd(source_embeddings, target_embeddings):
    """
    Calculate Earth Mover's Distance between two sets of tag embeddings.
    Using Wasserstein distance as an implementation of EMD.
    """
    if len(source_embeddings) == 0 or len(target_embeddings) == 0:
        return np.nan
    
    # Calculate pairwise distances
    distances = pdist(np.vstack([source_embeddings, target_embeddings]), metric='cosine')
    dist_matrix = squareform(distances)
    
    # Extract the cross-distances between the two sets
    cross_distances = dist_matrix[:len(source_embeddings), len(source_embeddings):]
    
    # For each source embedding, find the minimum distance to any target embedding
    min_distances = np.min(cross_distances, axis=1)
    
    # The EMD is the average of these minimum distances
    return np.mean(min_distances)

def tag_similarity_analysis(results_data):
    """
    Analyze the similarity between original and generated tags using EMD.
    """
    if not results_data or 'results' not in results_data:
        print("No results data found.")
        return None
    
    # Load sentence transformer model for embeddings
    print("Loading sentence transformer model...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # Multilingual model
    
    analysis_results = []
    
    print(f"Processing {len(results_data['results'])} items...")
    
    for i, item in enumerate(results_data['results']):
        if i % 100 == 0:
            print(f"Processing item {i+1}/{len(results_data['results'])}...")
        
        # Combine original subject and additional tags
        original_tags = item.get('original_subject_tags', []) + item.get('original_additional_tags', [])
        generated_tags = item.get('generated_tags', [])
        
        # Skip items with no tags
        if not original_tags or not generated_tags:
            continue
            
        # Detect language
        language = detect_language(original_tags + generated_tags)
        
        # Get embeddings for all tags
        original_embeddings = model.encode(original_tags)
        generated_embeddings = model.encode(generated_tags)
        
        # Calculate Earth Mover's Distance
        emd = calculate_emd(original_embeddings, generated_embeddings)
        
        # Calculate exact match metrics
        exact_matches = set(original_tags).intersection(set(generated_tags))
        precision = len(exact_matches) / len(generated_tags) if generated_tags else 0
        recall = len(exact_matches) / len(original_tags) if original_tags else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        
        analysis_results.append({
            'identifier': item.get('identifier', ''),
            'original_tags': original_tags,
            'generated_tags': generated_tags,
            'language': language,
            'emd_score': emd,
            'exact_match_count': len(exact_matches),
            'exact_match_precision': precision,
            'exact_match_recall': recall,
            'exact_match_f1': f1
        })
    
    return analysis_results

def visualize_results(analysis_results):
    """Create visualizations of the analysis results."""
    if not analysis_results:
        print("No analysis results to visualize.")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(analysis_results)
    
    # Create output directory
    os.makedirs('evaluation_results', exist_ok=True)
    
    # EMD score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['emd_score'].dropna(), bins=30, alpha=0.7)
    plt.title('Distribution of Earth Mover\'s Distance Scores')
    plt.xlabel('EMD Score (lower is better)')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)
    plt.savefig('evaluation_results/emd_distribution.png')
    
    # EMD by language
    plt.figure(figsize=(10, 6))
    language_data = df.groupby('language')['emd_score'].mean().sort_values()
    language_data.plot(kind='bar')
    plt.title('Average EMD Score by Language')
    plt.xlabel('Language')
    plt.ylabel('Average EMD Score (lower is better)')
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('evaluation_results/emd_by_language.png')
    
    # Exact match vs EMD correlation
    plt.figure(figsize=(10, 6))
    plt.scatter(df['exact_match_f1'], df['emd_score'], alpha=0.5)
    plt.title('Correlation between Exact Match F1 and EMD Score')
    plt.xlabel('Exact Match F1 Score (higher is better)')
    plt.ylabel('EMD Score (lower is better)')
    plt.grid(alpha=0.3)
    plt.savefig('evaluation_results/f1_vs_emd.png')
    
    # Save results to CSV
    df.to_csv('evaluation_results/tag_similarity_analysis.csv', index=False)
    
    # Generate summary statistics
    with open('evaluation_results/summary_statistics.txt', 'w') as f:
        f.write("SUMMARY STATISTICS\n")
        f.write("=================\n\n")
        
        f.write(f"Total items analyzed: {len(df)}\n")
        f.write(f"Language distribution: {df['language'].value_counts().to_dict()}\n\n")
        
        f.write("EMD Statistics:\n")
        f.write(f"  Mean: {df['emd_score'].mean():.4f}\n")
        f.write(f"  Median: {df['emd_score'].median():.4f}\n")
        f.write(f"  Std Dev: {df['emd_score'].std():.4f}\n")
        f.write(f"  Min: {df['emd_score'].min():.4f}\n")
        f.write(f"  Max: {df['emd_score'].max():.4f}\n\n")
        
        f.write("Exact Match Statistics:\n")
        f.write(f"  Mean F1: {df['exact_match_f1'].mean():.4f}\n")
        f.write(f"  Mean Precision: {df['exact_match_precision'].mean():.4f}\n")
        f.write(f"  Mean Recall: {df['exact_match_recall'].mean():.4f}\n")
    
    print(f"Results saved to evaluation_results/ directory")
    
    return df

def analyze_tag_tokens(analysis_results):
    """Analyze token-level patterns in tags."""
    all_original_tags = []
    all_generated_tags = []
    
    for item in analysis_results:
        all_original_tags.extend(item['original_tags'])
        all_generated_tags.extend(item['generated_tags'])
    
    # Get token counts
    original_counter = Counter(all_original_tags)
    generated_counter = Counter(all_generated_tags)
    
    # Find most common original tags not in generated
    missing_tags = {tag: count for tag, count in original_counter.items() 
                   if tag not in generated_counter}
    
    # Find most common generated tags not in original
    new_tags = {tag: count for tag, count in generated_counter.items() 
               if tag not in original_counter}
    
    # Save token analysis
    with open('evaluation_results/tag_token_analysis.txt', 'w') as f:
        f.write("TAG TOKEN ANALYSIS\n")
        f.write("=================\n\n")
        
        f.write(f"Total unique original tags: {len(original_counter)}\n")
        f.write(f"Total unique generated tags: {len(generated_counter)}\n\n")
        
        f.write("Top 20 original tags:\n")
        for tag, count in original_counter.most_common(20):
            f.write(f"  {tag}: {count}\n")
        
        f.write("\nTop 20 generated tags:\n")
        for tag, count in generated_counter.most_common(20):
            f.write(f"  {tag}: {count}\n")
        
        f.write("\nTop 20 original tags not in generated:\n")
        for tag, count in sorted(missing_tags.items(), key=lambda x: x[1], reverse=True)[:20]:
            f.write(f"  {tag}: {count}\n")
        
        f.write("\nTop 20 generated tags not in original:\n")
        for tag, count in sorted(new_tags.items(), key=lambda x: x[1], reverse=True)[:20]:
            f.write(f"  {tag}: {count}\n")

def main():
    # Load results data
    filepath = "generated_tags_results.json"
    results_data = load_results(filepath)
    
    if not results_data:
        print("Failed to load results data.")
        return
    
    # Analyze tag similarity
    print("Analyzing tag similarity...")
    analysis_results = tag_similarity_analysis(results_data)
    
    if not analysis_results:
        print("Analysis failed.")
        return
    
    # Visualize results
    print("Visualizing results...")
    df = visualize_results(analysis_results)
    
    # Analyze tag tokens
    print("Analyzing tag tokens...")
    analyze_tag_tokens(analysis_results)
    
    print("Analysis complete!")
    
    # Display sample results
    print("\nSample results (5 items):")
    sample_df = df.sample(min(5, len(df))).sort_values('emd_score')
    pd.set_option('display.max_columns', None)
    print(sample_df[['language', 'emd_score', 'exact_match_f1', 'original_tags', 'generated_tags']])

if __name__ == "__main__":
    main()