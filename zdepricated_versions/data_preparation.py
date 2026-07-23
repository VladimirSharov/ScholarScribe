import json

def filter_and_prepare_data(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)  # Load the original complex nested JSON data

        # Define the keys we are interested in
        relevant_keys = {"dc.contributor.tiedekunta", "dc.subject.yso", "dc.subject.other"}

        # Prepare a list to hold simplified entries
        simplified_data = []

        # Traverse through the nested lists and dictionaries
        for entry_list in data:
            simplified_entry = {}
            for item in entry_list:
                if item['key'] in relevant_keys:
                    # Create a list for each key if it doesn't exist
                    if item['key'] not in simplified_entry:
                        simplified_entry[item['key']] = []
                    # Append the value to the appropriate key
                    simplified_entry[item['key']].append(item['value'])

            # Only add entries that have all necessary keys
            if all(key in simplified_entry for key in relevant_keys):
                simplified_data.append(simplified_entry)

        # Save the simplified data to a new JSON file
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(simplified_data, file, indent=4, ensure_ascii=False)
        
        print(f"Data prepared and saved to {output_file}")
    except Exception as e:
        print(f"Error processing the file: {str(e)}")

if __name__ == "__main__":
    input_file_path = 'output.json'
    output_file_path = 'prepared_data.json'
    filter_and_prepare_data(input_file_path, output_file_path)
