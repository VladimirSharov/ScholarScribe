import json
import os

def remove_fields_from_dataset(input_path, output_path, fields_to_remove):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Loop through each entity and remove specified fields
    for entity in data:
        for field in fields_to_remove:
            if field in entity:
                del entity[field]  # Safely delete the field

    # Write the modified data to a temporary file first
    temp_file = f'{output_path}.temp'
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    # Replace the original file with the temp file (atomic operation)
    os.replace(temp_file, output_path)
    print(f"Fields {fields_to_remove} successfully removed and saved to {output_path}")

input_path = 'data_preparation/tempFieldMod/full_dataset_e.json'
output_path = 'data_preparation/tempFieldMod/full_dataset_er.json'
fields_to_remove = ['thesis_title', 'abstract']

remove_fields_from_dataset(input_path, output_path, fields_to_remove)
