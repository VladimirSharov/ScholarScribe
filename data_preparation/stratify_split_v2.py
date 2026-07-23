import json
import os
import random
import logging
from sklearn.model_selection import train_test_split

def stratify_and_split(input_path, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Load your data
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {input_path}")
        return

    # Shuffle the data to ensure randomness
    random.shuffle(data)

    # Create a dictionary to group entities by faculty
    faculties = {}
    unclassified_entities = []

    for entity in data:
        # More robust faculty extraction
        if 'faculty' in entity and entity['faculty']:
            # Use the first faculty, prioritizing non-empty strings
            faculty_names = [f for f in entity['faculty'] if f and isinstance(f, str)]
            
            if faculty_names:
                faculty_name = faculty_names[0]
                if faculty_name not in faculties:
                    faculties[faculty_name] = []
                faculties[faculty_name].append(entity)
            else:
                unclassified_entities.append(entity)
        else:
            unclassified_entities.append(entity)

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

    # Handle unclassified entities
    if unclassified_entities:
        logger.warning(f"Number of unclassified entities: {len(unclassified_entities)}")
        # You can choose to add these to train data, or handle differently
        train_data.extend(unclassified_entities)

    # Create output filenames based on the input filename
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_folder = 'data_split_v3'
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Define output paths
    output_train_path = os.path.join(output_folder, f'{base_name}_train.json')
    output_val_path = os.path.join(output_folder, f'{base_name}_val.json')
    output_test_path = os.path.join(output_folder, f'{base_name}_test.json')

    # Save the split data into separate files
    def save_json(path, data):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    save_json(output_train_path, train_data)
    save_json(output_val_path, val_data)
    save_json(output_test_path, test_data)

    # Print split statistics
    logger.info(f"Data split completed:")
    logger.info(f"Total entities: {len(data)}")
    logger.info(f"Train set: {len(train_data)} entities")
    logger.info(f"Validation set: {len(val_data)} entities")
    logger.info(f"Test set: {len(test_data)} entities")
    logger.info(f"Files saved:\n - {output_train_path}\n - {output_val_path}\n - {output_test_path}")

# Call the function with your file path and desired split ratio
input_path = 'data_preparation/dataset/full_dataset_v3.json'
stratify_and_split(input_path, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)