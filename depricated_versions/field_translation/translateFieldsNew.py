import json
from deep_translator import GoogleTranslator

# Set up the translator (auto-detects source language and translates to English)
translator = GoogleTranslator(source='auto', target='en')

def translate_tags(input_path, output_path):
    # Load the dataset
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for entity in data:
        if 'subject_tags' in entity:
            translated_tags = []
            for tag in entity['subject_tags']:
                try:
                    # Translate each subject tag (no need for lang_tgt in deep_translator)
                    translated = translator.translate(tag)
                    translated_tags.append(translated)
                except Exception as e:
                    print(f"Error translating tag: {tag}, error: {e}")
                    translated_tags.append(tag)  # Keep the original in case of error
            # Add translations to the new field
            entity['subject_tags_eng'] = translated_tags

    # Save the new dataset with translations
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Use your file paths
input_path = 'prepared_datasets/faculty_related_v2.json'
output_path = 'prepared_datasets/faculty_related_v2_translation.json'

translate_tags(input_path, output_path)
