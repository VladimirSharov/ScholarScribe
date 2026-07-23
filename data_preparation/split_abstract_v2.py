import json
import os
import uuid
import logging

def split_by_abstract(input_path, output_path):
    """
    Split entities by their abstracts, creating a new unique identifier for each split entity.
    
    Args:
        input_path (str): Path to the input JSON file
        output_path (str): Path to save the output JSON file
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # Load input data
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Prepare new list with each abstract in its own entity
        new_data = []
        original_count = len(data)
        split_count = 0

        for entity in data:
            # Ensure abstracts is a list and not empty
            abstracts = entity.get("abstract", [])
            
            # If no abstracts, skip or keep original entity
            if not abstracts:
                logger.warning(f"Entity {entity.get('identifier', 'Unknown')} has no abstracts.")
                continue

            # Process each abstract
            for idx, abstract in enumerate(abstracts):
                # Create a new entity for each abstract
                new_entity = entity.copy()
                
                # Modify abstract to contain only one abstract
                new_entity["abstract"] = [abstract]
                
                # Create a new unique identifier
                original_id = entity.get('identifier', '')
                # Combine original ID with abstract index and a unique suffix
                new_identifier = f"{original_id}_abstract_{idx}_{uuid.uuid4().hex[:8]}"
                new_entity["identifier"] = new_identifier
                
                # Optionally, track the original identifier
                new_entity["original_identifier"] = original_id
                
                # Optional: track abstract language if available
                if "abstract_language" in entity and isinstance(entity["abstract_language"], list):
                    new_entity["abstract_language"] = [entity["abstract_language"][idx]] if idx < len(entity.get("abstract_language", [])) else []
                
                new_data.append(new_entity)
                split_count += 1

        # Save the new dataset to the output path
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)

        # Log statistics
        logger.info(f"Processing file: {input_path}")
        logger.info(f"Original entities: {original_count}")
        logger.info(f"Split entities: {split_count}")
        logger.info(f"Abstracts split and saved to {output_path}")

    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {input_path}")
    except Exception as e:
        logger.error(f"Unexpected error processing {input_path}: {e}")

def main():
    # Define input and output directories
    input_dir = 'data_split_v3'  # Update this to match your current folder structure
    output_dir = 'data_split_v4'  # New output directory for split files
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Files to process
    files_to_process = [
        "full_dataset_v3_train.json",
        "full_dataset_v3_val.json",
        "full_dataset_v3_test.json"
    ]

    # Process each file
    for filename in files_to_process:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Call the split function
        split_by_abstract(input_path, output_path)

if __name__ == "__main__":
    main()