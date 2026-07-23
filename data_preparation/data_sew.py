import json
import os

# Directory containing your data files
data_dir = "data_split_v4"
output_file = "full_dataset.json"

# Collect all data
all_data = []

# List of files to combine
file_names = ["full_dataset_v3_test.json", "full_dataset_v3_train.json", "full_dataset_v3_val.json"]

for filename in file_names:
    file_path = os.path.join(data_dir, filename)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.extend(data)

# Save combined data
with open(os.path.join(data_dir, output_file), 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print(f"Combined {len(all_data)} entries into {output_file}")