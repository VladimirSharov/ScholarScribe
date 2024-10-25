import requests
import os
import time
import xml.etree.ElementTree as ET
import json
import signal
import sys

# Global flag to control the loop execution
continue_running = True

# Initialize necessary URLs and file paths
base_url = 'https://jyx.jyu.fi'
collection_url = f'{base_url}/rest/collections/3e5cf62e-86f2-44c0-beed-88ff6836d55b/items'
output_file = 'data_collection/output.json'
state_file = 'data_collection/state.json'

def signal_handler(signum, frame):
    global continue_running
    print("Signal received, stopping gracefully...")
    continue_running = False

# Set up signal handling
signal.signal(signal.SIGINT, signal_handler)

def get_next_fetching_parameters():
    try:
        with open(state_file, 'r') as file:
            data = json.load(file)
        return data['offset'], data['limit']
    except (FileNotFoundError, json.JSONDecodeError):
        save_current_fetching_parameters(0, 10)
        return 0, 10

def save_current_fetching_parameters(offset, limit):
    with open(state_file, 'w') as file:
        json.dump({'offset': offset, 'limit': limit}, file)

def fetch_full_item_data(item_url):
    response = requests.get(item_url, headers={'Accept': 'application/json'})
    return response.text

def check_output_file(file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write('[')  # Start with an empty JSON array

def fetch_data(start_offset, limit, file_path):
    params = {'limit': limit, 'offset': start_offset}
    response = requests.get(collection_url, params=params, headers={'Accept': 'application/xml'})
    root = ET.fromstring(response.content)

    items_fetched = 0
    with open(file_path, 'r+', encoding='utf-8') as file:
        file.seek(0, os.SEEK_END)
        if file.tell() > 2:
            file.seek(file.tell() - 1)
            file.write(', ')

        for item in root.findall('item'):
            if items_fetched > 0:
                file.write(', ')
            item_link = item.find('link').text
            item_url = f"{base_url}{item_link}/metadata"
            full_item_data = fetch_full_item_data(item_url)
            file.write(full_item_data)
            items_fetched += 1

        file.write(']')

    return items_fetched

def main():
    print("Program started. To stop the program, press Ctrl+C in the console.")
    check_output_file(output_file)
    start_offset, limit = get_next_fetching_parameters()

    while continue_running:
        items_fetched = fetch_data(start_offset, limit, output_file)
        if items_fetched < limit:
            print("Fetched all available items or reached the end of the dataset.")
            break

        start_offset += items_fetched
        save_current_fetching_parameters(start_offset, limit)

        print(f"Fetched data starting from offset {start_offset}, {items_fetched} items fetched.")
        time.sleep(1)  # Adjust based on API's rate limiting rules

    print("Completed fetching process or stopped by user.")
    sys.exit(0)

if __name__ == "__main__":
    main()
