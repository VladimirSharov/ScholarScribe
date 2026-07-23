import json

# Sample of your data structure
schema = {
    "description": "Academic thesis abstracts database",
    "fields": {
        "date_issued": {
            "type": "array of strings",
            "description": "Year when the thesis was published"
        },
        "identifier": {
            "type": "string",
            "description": "URL identifier for the abstract"
        },
        "abstract_language": {
            "type": "string",
            "description": "Language code of the abstract (e.g., 'fin' for Finnish)"
        },
        "abstract": {
            "type": "array of strings",
            "description": "The thesis abstract text"
        },
        "language": {
            "type": "array of strings",
            "description": '''Language code(s) of the thesis. 
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
            '''
        },
        "additional_tags": {
            "type": "array of strings",
            "description": "Additional keywords/tags related to the thesis"
        },
        "thesis_title": {
            "type": "array of strings",
            "description": "Title of the thesis"
        },
        "faculty": {
            "type": "array of strings",
            "description": "Faculty name(s) associated with the thesis"
        },
        "subject_tags": {
            "type": "array of strings",
            "description": "Subject keywords categorizing the thesis"
        },
        "original_identifier": {
            "type": "string",
            "description": "Original URL identifier for the thesis"
        }
    },
    "example": {
        # Include a simplified example here
    }
}

# Save the schema to a file
with open("db_schema.json", 'w', encoding='utf-8') as f:
    json.dump(schema, f, ensure_ascii=False, indent=4)

print("Schema saved to db_schema.json")