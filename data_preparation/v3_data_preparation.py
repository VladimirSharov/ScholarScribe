import json
import os
from collections import Counter, defaultdict

def standardize_language_code(code):
    """
    Standardize language codes to ISO 639-2/B format
    """
    language_mapping = {
        'fi': 'fin', 'fin': 'fin',
        'en': 'eng', 'eng': 'eng',
        'ru': 'rus', 'rus': 'rus',
        'fr': 'fre', 'fre': 'fre',
        'sv': 'swe', 'swe': 'swe',
        'de': 'ger', 'ger': 'ger',
        'ita': 'ita',
        'lat': 'lat'
    }
    # If code is None, return None
    if not code:
        return None
    
    # Convert to string and lowercase
    code_str = str(code).lower()
    
    return language_mapping.get(code_str, code_str)

def filter_and_prepare_data(input_file, datasets_info, field_name_mapping, output_dir):
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a directory for metadata and logs
        metadata_dir = os.path.join(output_dir, 'metadata')
        os.makedirs(metadata_dir, exist_ok=True)

        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)  # Load the original complex nested JSON data

        results = {output_file: [] for output_file in datasets_info.keys()}
        
        # Tracking statistics
        total_entries = len(data)
        deprecated_entries = 0
        language_counter = Counter()
        missing_lang_entries = 0

        # Traverse through the nested lists and dictionaries
        for entry_list in data:
            simplified_entry = {}
            
            # Process each item in the entry
            for item in entry_list:
                # Special handling for abstract with language
                if item['key'] == 'dc.description.abstract':
                    # If language is present in the abstract item
                    if item.get('language'):
                        # Use the language from the field if available
                        simplified_entry['abstract_language'] = standardize_language_code(item['language'])
                
                # Mapping keys and processing
                if item['key'] in {key for keys in datasets_info.values() for key in keys}:
                    new_key = field_name_mapping.get(item['key'], item['key'])
                    
                    # Special handling for specific keys
                    if new_key not in simplified_entry:
                        simplified_entry[new_key] = []
                    
                    # Add value
                    simplified_entry[new_key].append(item['value'])
                
                # Add identifier
                if item['key'] == 'dc.identifier.uri':
                    simplified_entry['identifier'] = item['value']

            # Check if entry has all necessary keys for each dataset
            for output_file, fields_to_include in datasets_info.items():
                entry_copy = simplified_entry.copy()
                
                # Check if all required fields are present
                if all(field_name_mapping.get(key, key) in entry_copy for key in fields_to_include):
                    results[output_file].append(entry_copy)
                else:
                    deprecated_entries += 1

        # Save the simplified data to new JSON files
        for output_file, simplified_data in results.items():
            full_path = os.path.join(output_dir, output_file)
            with open(full_path, 'w', encoding='utf-8') as file:
                json.dump(simplified_data, file, indent=4, ensure_ascii=False)
            print(f"Data prepared and saved to {full_path}")

        # Create metadata log
        metadata_log = {
            'total_entries': total_entries,
            'deprecated_entries': deprecated_entries,
            'language_distribution': dict(language_counter)
        }

        # Save metadata log
        metadata_path = os.path.join(metadata_dir, 'processing_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as file:
            json.dump(metadata_log, file, indent=4, ensure_ascii=False)
        
        print("\n--- Processing Metadata ---")
        for key, value in metadata_log.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"Error processing the file: {str(e)}")

if __name__ == "__main__":
    input_file_path = 'data_collection/output.json'
    output_directory = 'data_preparation/dataset'  # Directory to store all output files
    
    datasets_info = {
        'full_dataset_v3.json': {"dc.title", "dc.contributor.tiedekunta", "dc.description.abstract", "dc.subject.other", "dc.subject.yso", "dc.language.iso", "dc.date.issued", "dc.identifier.uri"},
        # 'tags_dataset.json': {"dc.title", "dc.contributor.tiedekunta", "dc.subject.other", "dc.subject.yso", "dc.date.issued", "dc.identifier.uri"}
    }
    
    field_name_mapping = {
        "dc.contributor.tiedekunta": "faculty",
        "dc.subject.yso": "subject_tags",
        "dc.subject.other": "additional_tags",
        "dc.title": "thesis_title",
        "dc.description.abstract": "abstract",
        "dc.language.iso": "language",
        "dc.date.issued": "date_issued",
        "dc.identifier.uri": "identifier"
    }

    filter_and_prepare_data(input_file_path, datasets_info, field_name_mapping, output_directory)