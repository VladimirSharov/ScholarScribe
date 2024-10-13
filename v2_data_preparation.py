import json
import os

# ToDo
## Create script which will output label file (data_file_name+label, in folder labels, so later I could auto based on file name)

def filter_and_prepare_data(input_file, datasets_info, field_name_mapping, output_dir):
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)  # Load the original complex nested JSON data

        results = {output_file: [] for output_file in datasets_info.keys()}

        # Traverse through the nested lists and dictionaries
        for entry_list in data:
            for output_file, fields_to_include in datasets_info.items():
                simplified_entry = {}
                for item in entry_list:
                    if item['key'] in fields_to_include:
                        new_key = field_name_mapping.get(item['key'], item['key'])
                        # Create a list for each key if it doesn't exist
                        if new_key not in simplified_entry:
                            simplified_entry[new_key] = []
                        # Append the value to the appropriate key
                        simplified_entry[new_key].append(item['value'])

                # Only add entries that have all necessary keys
                if all(field_name_mapping.get(key, key) in simplified_entry for key in fields_to_include):
                    results[output_file].append(simplified_entry)

        # Save the simplified data to new JSON files in the specified directory
        for output_file, simplified_data in results.items():
            full_path = os.path.join(output_dir, output_file)
            with open(full_path, 'w', encoding='utf-8') as file:
                json.dump(simplified_data, file, indent=4, ensure_ascii=False)
            print(f"Data prepared and saved to {full_path}")

    except Exception as e:
        print(f"Error processing the file: {str(e)}")

if __name__ == "__main__":
    input_file_path = 'output.json'
    output_directory = 'prepared_datasets'  # Directory to store all output files
    datasets_info = {
        'title_abstract.json': {"dc.title", "dc.description.abstract"},
        'title_subjects.json': {"dc.title", "dc.description.abstract", "dc.subject.other", "dc.subject.yso"},
        'full_details.json': {"dc.title", "dc.description.abstract", "dc.subject.other", "dc.subject.yso", "dc.language.iso", "dc.date.issued"},
        'faculty_related_v2.json': {"dc.title", "dc.contributor.tiedekunta", "dc.subject.other", "dc.subject.yso","dc.date.issued"}
    }
    field_name_mapping = {
        "dc.contributor.tiedekunta": "faculty",
        "dc.subject.yso": "subject_tags",
        "dc.subject.other": "additional_tags",
        "dc.title": "thesis_title",
        "dc.description.abstract": "abstract",
        "dc.language.iso": "language",
        "dc.date.issued": "date_issued"
    }

    filter_and_prepare_data(input_file_path, datasets_info, field_name_mapping, output_directory)
