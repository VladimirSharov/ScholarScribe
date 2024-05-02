import json
from collections import defaultdict

def analyze_json_structure(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # Assuming data is a list of lists of dictionaries

        field_summary = defaultdict(lambda: defaultdict(int))

        # Analyze data for each field and count the occurrences of each value
        for sublist in data:
            for metadata in sublist:
                key = metadata['key']
                value = metadata['value']
                field_summary[key][value] += 1

        # Prepare the summary output with detailed mapping of each value
        summary = {key: dict(value_counts) for key, value_counts in field_summary.items()}
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
    input_file_path = 'output.json'
    output_file_path = 'output_ValuesSortedByField.json'
    summary = analyze_json_structure(input_file_path)
    save_summary_to_file(summary, output_file_path)
