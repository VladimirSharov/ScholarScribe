import json
from deep_translator import GoogleTranslator

# Initialize the translator
translator = GoogleTranslator(source='auto', target='en')

# Example JSON data
data = [
    {
        "dc.subject.other": [
            "Plinius Secundus, Gaius",
            "Naturalis historia"
        ],
        "dc.contributor.tiedekunta": [
            "Humanistinen tiedekunta",
            "Faculty of Humanities"
        ],
        "dc.subject.yso": [
            "historia",
            "latinankielinen kirjallisuus",
            "maailmankuva",
            "maailmankaikkeus",
            "planeetat",
            "tähdet",
            "antiikki"
        ]
    }
]

# Function to translate a single field
def translate_field(field_values, target_language='en'):
    translated_values = []
    for value in field_values:
        # Translate the value
        translated = translator.translate(value)
        translated_values.append(translated)
    return translated_values

# Translate and modify the JSON structure
for entity in data:
    # Create a new dictionary to store the translations
    new_translations = {}

    for key, values in entity.items():
        # Translate the values and store both original and translated
        translated_values = translate_field(values)
        new_translations[key + '_translated'] = translated_values
    
    # Update the original dictionary with the new translations
    entity.update(new_translations)

# Print or save the modified JSON with translations
print(json.dumps(data, ensure_ascii=False, indent=4))

# Optionally, write the translated JSON to a file
with open('translated_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
