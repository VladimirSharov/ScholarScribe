import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

# Dictionary to standardize faculty names (cross-language consistency)
FACULTY_DICT = {
    "Humanistis-yhteiskuntatieteellinen tiedekunta": "Faculty of Humanities and Social Sciences",
    "Humanistinen tiedekunta": "Faculty of Humanities",
    "Yhteiskuntatieteellinen tiedekunta": "Faculty of Social Sciences",
    "Kasvatustieteiden ja psykologian tiedekunta": "Faculty of Education and Psychology",
    "Kasvatustieteiden tiedekunta": "Faculty of Education",
    "Liikuntatieteellinen tiedekunta": "Faculty of Sport and Health Sciences",
    "Matemaattis-luonnontieteellinen tiedekunta": "Faculty of Sciences",
    "Kauppakorkeakoulu": "School of Business and Economics",
    "Informaatioteknologian tiedekunta": "Faculty of Information Technology"
}

# Step 1: Process subject_tags and additional_tags into binary vectors
def process_tags(data, tag_fields=['subject_tags', 'additional_tags']):
    unique_tags = set()

    # Collect unique tags across both fields
    for entry in data:
        for field in tag_fields:
            unique_tags.update(entry.get(field, []))

    # Create tag-to-ID mappings
    class2id = {tag: idx for idx, tag in enumerate(sorted(unique_tags))}
    id2class = {idx: tag for tag, idx in class2id.items()}

    # Transform each entry into a binary vector
    def tags_to_vector(entry):
        tags = []
        for field in tag_fields:
            tags += entry.get(field, [])
        
        tag_vector = [0] * len(class2id)
        for tag in tags:
            if tag in class2id:
                tag_vector[class2id[tag]] = 1
        return tag_vector

    # Apply to dataset
    for entry in data:
        entry['tags_vector'] = tags_to_vector(entry)

    return data, class2id, id2class

# Step 2: Process faculty using one-hot encoding and normalization
def process_faculty(data):
    # Normalize faculties
    def normalize_faculty(faculty_list):
        normalized = []
        for faculty in faculty_list:
            normalized.append(FACULTY_DICT.get(faculty, faculty))  # Use normalized version or original
        return list(set(normalized))  # Remove duplicates

    # Apply normalization
    for entry in data:
        entry['faculty'] = normalize_faculty(entry.get('faculty', []))

    # Perform one-hot encoding
    faculty_data = [entry['faculty'] for entry in data]
    faculty_df = pd.DataFrame(faculty_data, columns=['faculty'])
    encoder = OneHotEncoder(sparse=False)
    encoded_faculty = encoder.fit_transform(faculty_df)

    # Add one-hot encoding back into the dataset
    for i, entry in enumerate(data):
        entry['faculty_vector'] = list(encoded_faculty[i])

    return data

# Step 3: Process language with one-hot encoding
def process_language(data):
    # Collect unique languages and apply one-hot encoding
    languages = [entry.get('language', ['']) for entry in data]
    lang_df = pd.DataFrame(languages, columns=['language'])
    encoder = OneHotEncoder(sparse=False)
    encoded_language = encoder.fit_transform(lang_df)

    # Add one-hot encoding back into the dataset
    for i, entry in enumerate(data):
        entry['language_vector'] = list(encoded_language[i])

    return data

# Step 4: Process year (either as is or bucketize)
def process_year(data):
    # Convert year to integer and store
    for entry in data:
        year = entry.get('date_issued', [''])[0]
        try:
            entry['year'] = int(year)
        except ValueError:
            entry['year'] = None  # Handle invalid years gracefully

    return data

# Step 5: Combine all processing functions
def process_all_fields(input_file, output_file):
    # Load the dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Apply all processing steps
    data, class2id, id2class = process_tags(data)
    data = process_faculty(data)
    data = process_language(data)
    data = process_year(data)

    # Save the processed dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return class2id, id2class  # Return for reference if needed

if __name__ == "__main__":
    input_file = 'data_preparation/tempFieldMod/full_dataset_er.json'
    output_file = 'data_preparation/tempFieldMod/full_dataset_erp.json'
    
    class2id, id2class = process_all_fields(input_file, output_file)
    print(f"Processing complete. Output saved to {output_file}.")
