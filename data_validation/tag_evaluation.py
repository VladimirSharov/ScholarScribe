import json
import numpy as np
from datetime import datetime
import os

def exact_match_ratio(original_tags, generated_tags):
    """Calculate exact match ratio between tag sets"""
    if not original_tags or not generated_tags:
        return 0.0
    
    original_set = set(tag.lower() for tag in original_tags)
    generated_set = set(tag.lower() for tag in generated_tags)
    
    matches = original_set.intersection(generated_set)
    return len(matches) / max(len(original_set), len(generated_set))

def evaluate_tags(results_file):
    """Evaluate tag generation performance using only exact match"""
    # Load data
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data.get('metadata', {})
    results = data.get('results', [])
    
    # Metrics storage
    metrics = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "source_file": results_file,
        "original_metadata": metadata,
        "summary_metrics": {
            "total_samples": len(results),
            "subject_tags": {
                "exact_match_ratio_avg": 0.0,
                "perfect_matches": 0,
                "zero_matches": 0
            },
            "additional_tags": {
                "exact_match_ratio_avg": 0.0,
                "perfect_matches": 0,
                "zero_matches": 0
            }
        },
        "detailed_metrics": []
    }
    
    # Lists to store metrics for each item
    subject_exact_ratios = []
    additional_exact_ratios = []
    
    # Process each result item
    for item in results:
        original_subject = item.get('original_subject_tags', [])
        generated_subject = item.get('generated_subject_tags', [])
        original_additional = item.get('original_additional_tags', [])
        generated_additional = item.get('generated_additional_tags', [])
        
        # Calculate exact match metrics
        subject_exact = exact_match_ratio(original_subject, generated_subject)
        additional_exact = exact_match_ratio(original_additional, generated_additional)
        
        # Store metrics
        subject_exact_ratios.append(subject_exact)
        additional_exact_ratios.append(additional_exact)
        
        # Track perfect and zero matches
        if subject_exact == 1.0:
            metrics["summary_metrics"]["subject_tags"]["perfect_matches"] += 1
        if subject_exact == 0.0:
            metrics["summary_metrics"]["subject_tags"]["zero_matches"] += 1
        
        if additional_exact == 1.0:
            metrics["summary_metrics"]["additional_tags"]["perfect_matches"] += 1
        if additional_exact == 0.0:
            metrics["summary_metrics"]["additional_tags"]["zero_matches"] += 1
        
        # Store detailed metrics for this item
        item_metrics = {
            "identifier": item.get("identifier", ""),
            "subject_tags_metrics": {
                "exact_match_ratio": subject_exact,
                "original_count": len(original_subject),
                "generated_count": len(generated_subject),
                "original_tags": original_subject,
                "generated_tags": generated_subject
            },
            "additional_tags_metrics": {
                "exact_match_ratio": additional_exact,
                "original_count": len(original_additional),
                "generated_count": len(generated_additional),
                "original_tags": original_additional,
                "generated_tags": generated_additional
            }
        }
        metrics["detailed_metrics"].append(item_metrics)
    
    # Calculate averages
    metrics["summary_metrics"]["subject_tags"]["exact_match_ratio_avg"] = np.mean(subject_exact_ratios) if subject_exact_ratios else 0
    metrics["summary_metrics"]["additional_tags"]["exact_match_ratio_avg"] = np.mean(additional_exact_ratios) if additional_exact_ratios else 0
    
    # Add percentages
    total = len(results)
    metrics["summary_metrics"]["subject_tags"]["perfect_match_percentage"] = (metrics["summary_metrics"]["subject_tags"]["perfect_matches"] / total) * 100 if total else 0
    metrics["summary_metrics"]["subject_tags"]["zero_match_percentage"] = (metrics["summary_metrics"]["subject_tags"]["zero_matches"] / total) * 100 if total else 0
    metrics["summary_metrics"]["additional_tags"]["perfect_match_percentage"] = (metrics["summary_metrics"]["additional_tags"]["perfect_matches"] / total) * 100 if total else 0
    metrics["summary_metrics"]["additional_tags"]["zero_match_percentage"] = (metrics["summary_metrics"]["additional_tags"]["zero_matches"] / total) * 100 if total else 0
    
    return metrics

def save_metrics(metrics, output_file):
    """Save metrics to a JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to {output_file}")

def create_visualization_data(metrics, output_file):
    """Create a simplified JSON file for visualization"""
    viz_data = {
        "evaluation_date": metrics["evaluation_timestamp"],
        "model_name": metrics["original_metadata"].get("model", "Unknown"),
        "total_samples": metrics["summary_metrics"]["total_samples"],
        "metrics_summary": {
            "subject_tags": {
                "exact_match_average": metrics["summary_metrics"]["subject_tags"]["exact_match_ratio_avg"],
                "perfect_matches_percent": metrics["summary_metrics"]["subject_tags"]["perfect_match_percentage"],
                "zero_matches_percent": metrics["summary_metrics"]["subject_tags"]["zero_match_percentage"]
            },
            "additional_tags": {
                "exact_match_average": metrics["summary_metrics"]["additional_tags"]["exact_match_ratio_avg"],
                "perfect_matches_percent": metrics["summary_metrics"]["additional_tags"]["perfect_match_percentage"],
                "zero_matches_percent": metrics["summary_metrics"]["additional_tags"]["zero_match_percentage"]
            }
        },
        "distribution": {
            "subject_exact_match": get_distribution(metrics["detailed_metrics"], 
                                                  lambda x: x["subject_tags_metrics"]["exact_match_ratio"]),
            "additional_exact_match": get_distribution(metrics["detailed_metrics"], 
                                                     lambda x: x["additional_tags_metrics"]["exact_match_ratio"])
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(viz_data, f, ensure_ascii=False, indent=2)
    print(f"Visualization data saved to {output_file}")

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

def main():
    # File paths
    input_file = "generated_tags_results.json"  # Your input file
    metrics_output = "tag_evaluation_metrics.json"
    viz_output = "tag_visualization_data.json"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Please update the script with the correct path to your results file.")
        return
    
    # Process data
    print(f"Evaluating tag generation in '{input_file}'...")
    metrics = evaluate_tags(input_file)
    
    # Save metrics
    save_metrics(metrics, metrics_output)
    
    # Create visualization data
    create_visualization_data(metrics, viz_output)
    
    # Print summary for quick reference
    print("\nEvaluation Summary:")
    print("-" * 40)
    print(f"Total samples: {metrics['summary_metrics']['total_samples']}")
    print("\nSubject Tags:")
    print(f"  Exact match ratio: {metrics['summary_metrics']['subject_tags']['exact_match_ratio_avg']:.2f}")
    print(f"  Perfect matches: {metrics['summary_metrics']['subject_tags']['perfect_matches']} " +
          f"({metrics['summary_metrics']['subject_tags']['perfect_match_percentage']:.1f}%)")
    print(f"  Zero matches: {metrics['summary_metrics']['subject_tags']['zero_matches']} " +
          f"({metrics['summary_metrics']['subject_tags']['zero_match_percentage']:.1f}%)")
    
    print("\nAdditional Tags:")
    print(f"  Exact match ratio: {metrics['summary_metrics']['additional_tags']['exact_match_ratio_avg']:.2f}")
    print(f"  Perfect matches: {metrics['summary_metrics']['additional_tags']['perfect_matches']} " +
          f"({metrics['summary_metrics']['additional_tags']['perfect_match_percentage']:.1f}%)")
    print(f"  Zero matches: {metrics['summary_metrics']['additional_tags']['zero_matches']} " +
          f"({metrics['summary_metrics']['additional_tags']['zero_match_percentage']:.1f}%)")

if __name__ == "__main__":
    main()