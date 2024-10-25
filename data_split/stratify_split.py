import json
from sklearn.model_selection import train_test_split
import os
import random

# Function to load data, split it, and save the parts into separate files
def stratify_and_split(input_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Load your data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Shuffle the data to ensure randomness
    random.shuffle(data)

    # Create a dictionary to group entities by faculty (use the first item in the faculty list)
    faculties = {}
    for entity in data:
        if 'faculty' in entity and len(entity['faculty']) > 0:
            faculty_name = entity['faculty'][0]  # Assuming first value (Finnish) represents the faculty
            if faculty_name not in faculties:
                faculties[faculty_name] = []
            faculties[faculty_name].append(entity)

    # Lists to store the split data
    train_data = []
    val_data = []
    test_data = []

    # Stratify data by faculty
    for faculty, entities in faculties.items():
        n = len(entities)
        if n > 2:
            # Normal splitting for faculties with more than 2 entities
            train, temp = train_test_split(entities, test_size=(1 - train_ratio), random_state=42)
            val, test = train_test_split(temp, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)
        elif n == 2:
            # For faculties with only 2 entities, split into train and test
            train, test = train_test_split(entities, test_size=0.5, random_state=42)
            val = []  # No validation set
        else:
            # For faculties with 1 entity, put it in the training set
            train = entities
            val = []
            test = []

        # Append the split data to the appropriate list
        train_data.extend(train)
        val_data.extend(val)
        test_data.extend(test)

    # Create output filenames based on the input filename
    base_name = os.path.splitext(os.path.basename(input_path))[0]  # Get the base filename without extension
    output_train_path = f'data_split/{base_name}_train.json'
    output_val_path = f'data_split/{base_name}_val.json'
    output_test_path = f'data_split/{base_name}_test.json'

    # Save the split data into separate files
    with open(output_train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    with open(output_val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=4)

    with open(output_test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)

    print(f"Data split and saved into:\n - {output_train_path}\n - {output_val_path}\n - {output_test_path}")


# Call the function with your file path and desired split ratio
input_path = 'data_preparation/dataset/full_dataset.json'
stratify_and_split(input_path, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
