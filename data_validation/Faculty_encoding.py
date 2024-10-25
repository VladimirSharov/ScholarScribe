import json

def validate_faculty(input_path, output_pairs_path, output_singles_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    faculty_pairs = set()
    single_language_faculties = 0
    total_faculty_entities = 0
    single_language_entities = []

    for entity in data:
        if 'faculty' in entity:
            total_faculty_entities += 1
            faculty = entity['faculty']
            if len(faculty) == 2:
                faculty_pairs.add(tuple(faculty))  # Add as a tuple (Finnish, English)
            else:
                single_language_faculties += 1
                single_language_entities.append(entity)

    print(f"Total entities: {len(data)}")
    print(f"Entities with faculty field: {total_faculty_entities}")
    print(f"Entities with single-language faculty: {single_language_faculties}")
    print(f"Unique faculty pairs (Finnish, English): {len(faculty_pairs)}")

    # Save unique faculty pairs to a file
    with open(output_pairs_path, 'w', encoding='utf-8') as f:
        json.dump(list(faculty_pairs), f, ensure_ascii=False, indent=4)

    # Save single-language faculty entities to a file
    with open(output_singles_path, 'w', encoding='utf-8') as f:
        json.dump(single_language_entities, f, ensure_ascii=False, indent=4)

# Call the function to validate the faculty field
input_path = 'prepared_datasets/faculty_related_v2.json'
output_pairs_path = 'prepared_datasets/unique_faculty_pairs.json'
output_singles_path = 'prepared_datasets/single_language_faculties.json'

validate_faculty(input_path, output_pairs_path, output_singles_path)
