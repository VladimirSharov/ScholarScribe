import json
import os


# Function to split entities by abstract
def split_by_abstract(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Prepare new list with each abstract in its own entity
    new_data = []
    for entity in data:
        abstracts = entity.get("abstract", [])
        for abstract in abstracts:
            new_entity = entity.copy()  # Copy the entire entity
            new_entity["abstract"] = [abstract]  # Replace with a single abstract
            new_data.append(new_entity)  # Add to the new dataset

    # Save the new dataset to the output path
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

    print(f"Abstracts split and saved to {output_path}")


# File paths from the existing split
files_to_process = [
    "data_split/full_dataset_train.json",
    "data_split/full_dataset_val.json",
    "data_split/full_dataset_test.json"
]

# Process each file and write the outputs to the `data_split_v2` folder
for file_path in files_to_process:
    base_name = os.path.basename(file_path)  # Extract the file name
    output_path = f"data_split_v2/{base_name}"  # Write to data_split_v2 with the same name
    split_by_abstract(file_path, output_path)
