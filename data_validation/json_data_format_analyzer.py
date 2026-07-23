import json
from collections import defaultdict, Counter

def analyze_json_structure(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        field_summary = defaultdict(Counter)
        
        for sublist in data:
            for item in sublist:
                key = item.get('key')
                value = item.get('value')
                if key and value:
                    field_summary[key]['count'] += 1
                    field_summary[key][type(value).__name__] += 1

        total_items = sum(len(sublist) for sublist in data)
        summary = {}
        
        for field, details in field_summary.items():
            summary[field] = {
                'total_occurrences': details['count'],
                'data_types': {k: v for k, v in details.items() if k != 'count'},
                'flags': []
            }
            if details['count'] < total_items:
                summary[field]['flags'].append('L')  # Less frequent
            if len(summary[field]['data_types']) > 1:
                summary[field]['flags'].append('D')  # Diverse data types
            if 'NoneType' in details:
                summary[field]['flags'].append('N')  # Contains null values
        
        return summary
    except Exception as e:
        print(f"Error processing the file: {str(e)}")
        return {}

def save_summary_to_file(summary, output_file):
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(summary, file, indent=4, ensure_ascii=False)
        print(f"Summary saved to {output_file}")
    except Exception as e:
        print(f"Failed to save summary: {str(e)}")

if __name__ == "__main__":
    input_file_path = 'data_collection/output.json'
    output_file_path = 'data_validation/output_DataFormats.json'
    summary = analyze_json_structure(input_file_path)
    save_summary_to_file(summary, output_file_path)
