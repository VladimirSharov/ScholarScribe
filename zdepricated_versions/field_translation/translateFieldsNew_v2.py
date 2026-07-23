import json
import time
from deep_translator import GoogleTranslator
from requests.exceptions import RequestException

# Set up the translator
translator = GoogleTranslator(source='auto', target='en')

def translate_tags(input_path, output_path, retry_attempts=3, save_interval=10):
    # Load the dataset
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Prepare a new data list to hold the translated entities
    translated_data = []
    failed_translations = []

    for idx, entity in enumerate(data):
        if 'subject_tags' in entity:
            translated_tags = []
            for tag in entity['subject_tags']:
                for attempt in range(retry_attempts):
                    try:
                        # Translate each subject tag
                        translated = translator.translate(tag)
                        translated_tags.append(translated)
                        break  # Translation successful, break out of retry loop
                    except (RequestException, TimeoutError) as e:
                        print(f"Error translating tag: {tag}, attempt {attempt + 1}/{retry_attempts}, error: {e}")
                        time.sleep(2)  # Wait before retrying
                        if attempt == retry_attempts - 1:
                            # If we've reached the maximum retry attempts, log the failure
                            failed_translations.append(tag)
                            translated_tags.append(tag)  # Keep the original tag in case of repeated failure

            # Add translations to the new field
            entity['subject_tags_eng'] = translated_tags

        # Append the processed entity to the translated data
        translated_data.append(entity)

        # Save progress every `save_interval` entities
        if (idx + 1) % save_interval == 0:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(translated_data, f, ensure_ascii=False, indent=4)
            print(f"Saved progress at entity {idx + 1}")

    # Final save after the loop finishes
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=4)

    print(f"Translation complete. Failed to translate {len(failed_translations)} tags.")
    if failed_translations:
        print("Failed tags:", failed_translations)

# Use your file paths
input_path = 'prepared_datasets/faculty_related_v2.json'
output_path = 'prepared_datasets/faculty_related_v2_translation.json'

translate_tags(input_path, output_path)
