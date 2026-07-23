import json
from typing import Dict, List
from collections import defaultdict
import statistics
import os
from pathlib import Path

# Default paths
DEFAULT_INPUT_PATH = Path('data_preparation/dataset/full_dataset.json')
DEFAULT_OUTPUT_PATH = Path('data_validation/output_json_dataset_field_metrics.json')

def generate_entity_id(thesis_title: List[str]) -> str:
    """Generate a truncated ID from thesis title."""
    return thesis_title[0][:20] if thesis_title else "NO_TITLE"

def analyze_dataset(data: List[Dict]) -> Dict:
    """Analyze dataset for various metrics and anomalies."""
    stats = {
        'counts': defaultdict(int),
        'anomalies': [],
        'text_stats': {
            'thesis_title': {'min': float('inf'), 'max': 0, 'total': 0, 'min_id': '', 'max_id': ''},
            'abstract': {'min': float('inf'), 'max': 0, 'total': 0, 'min_id': '', 'max_id': ''}
        },
        'tag_stats': {
            'additional_tags': {'min': float('inf'), 'max': 0, 'total': 0, 'min_id': '', 'max_id': ''},
            'subject_tags': {'min': float('inf'), 'max': 0, 'total': 0, 'min_id': '', 'max_id': ''}
        },
        'faculty_distribution': defaultdict(int)
    }
    
    for item in data:
        entity_id = generate_entity_id(item.get('thesis_title', []))
        
        # Validate single-value fields
        for field in ['date_issued', 'language', 'thesis_title']:
            if len(item.get(field, [])) != 1:
                stats['anomalies'].append({
                    'id': entity_id,
                    'issue': f"Unexpected number of {field}: {len(item.get(field, []))}"
                })
        
        # Text length statistics
        for field in ['thesis_title', 'abstract']:
            if item.get(field):
                total_length = sum(len(text) for text in item[field])
                stats['text_stats'][field]['total'] += total_length
                stats['counts'][field] += 1
                
                if total_length < stats['text_stats'][field]['min']:
                    stats['text_stats'][field]['min'] = total_length
                    stats['text_stats'][field]['min_id'] = entity_id
                if total_length > stats['text_stats'][field]['max']:
                    stats['text_stats'][field]['max'] = total_length
                    stats['text_stats'][field]['max_id'] = entity_id
        
        # Tag count statistics
        for field in ['additional_tags', 'subject_tags']:
            if item.get(field):
                tag_count = len(item[field])
                stats['tag_stats'][field]['total'] += tag_count
                stats['counts'][field] += 1
                
                if tag_count < stats['tag_stats'][field]['min']:
                    stats['tag_stats'][field]['min'] = tag_count
                    stats['tag_stats'][field]['min_id'] = entity_id
                if tag_count > stats['tag_stats'][field]['max']:
                    stats['tag_stats'][field]['max'] = tag_count
                    stats['tag_stats'][field]['max_id'] = entity_id
        
        # Faculty analysis
        if item.get('faculty'):
            faculty_count = len(item['faculty'])
            stats['faculty_distribution'][faculty_count] += 1
            if faculty_count != 2:
                stats['anomalies'].append({
                    'id': entity_id,
                    'issue': f"Unusual number of faculties: {faculty_count}"
                })
        
        # Multiple abstracts check
        if len(item.get('abstract', [])) > 1:
            stats['anomalies'].append({
                'id': entity_id,
                'issue': f"Multiple abstracts found: {len(item['abstract'])}"
            })
    
    # Calculate averages
    for field in ['thesis_title', 'abstract']:
        if stats['counts'][field] > 0:
            stats['text_stats'][field]['average'] = stats['text_stats'][field]['total'] / stats['counts'][field]
    
    for field in ['additional_tags', 'subject_tags']:
        if stats['counts'][field] > 0:
            stats['tag_stats'][field]['average'] = stats['tag_stats'][field]['total'] / stats['counts'][field]
    
    return stats

def save_analysis(stats: Dict, output_path: Path):
    """Save analysis results to a JSON file."""
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)

def print_analysis(stats: Dict):
    """Print formatted analysis results."""
    print("\n=== Text Statistics ===")
    for field in ['thesis_title', 'abstract']:
        print(f"\n{field.upper()}:")
        print(f"Min length: {stats['text_stats'][field]['min']} (ID: {stats['text_stats'][field]['min_id']})")
        print(f"Max length: {stats['text_stats'][field]['max']} (ID: {stats['text_stats'][field]['max_id']})")
        print(f"Average length: {stats['text_stats'][field]['average']:.2f}")

    print("\n=== Tag Statistics ===")
    for field in ['additional_tags', 'subject_tags']:
        print(f"\n{field.upper()}:")
        print(f"Min tags: {stats['tag_stats'][field]['min']} (ID: {stats['tag_stats'][field]['min_id']})")
        print(f"Max tags: {stats['tag_stats'][field]['max']} (ID: {stats['tag_stats'][field]['max_id']})")
        print(f"Average tags: {stats['tag_stats'][field]['average']:.2f}")

    print("\n=== Faculty Distribution ===")
    for count, num in sorted(stats['faculty_distribution'].items()):
        print(f"Entries with {count} faculties: {num}")

    print("\n=== Anomalies ===")
    for anomaly in stats['anomalies']:
        print(f"ID: {anomaly['id']}")
        print(f"Issue: {anomaly['issue']}\n")

def main():
    """Main function to run the analysis with default paths."""
    try:
        # Read input data
        with open(DEFAULT_INPUT_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Analyze data
        stats = analyze_dataset(data)
        
        # Save results
        save_analysis(stats, DEFAULT_OUTPUT_PATH)
        
        # Print analysis to console
        print_analysis(stats)
        
        print(f"\nAnalysis results have been saved to: {DEFAULT_OUTPUT_PATH}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find the input file at {DEFAULT_INPUT_PATH}")
        print("Please make sure the file exists and the path is correct.")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in the input file")
        print(f"Details: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()