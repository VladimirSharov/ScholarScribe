import json
import os
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import torch
import numpy as np

#ToDo
## mode selector detached or full embeddings result, full is now commented out, not important 
## adjust embedding file naming to match based on path to data, not important
## fields validation....

# Settings

# path_to_data = 'prepared_datasets/title_abstract.json'
path_to_data = 'prepared_datasets/faculty_related.json'

# Fields to include in the model input
# input_fields = ['thesis_title', 'abstract']  # Modify as needed
input_fields = ['thesis_title', 'faculty', 'subject_tags', 'additional_tags']

def load_data(filename, limit=None):
    """Load JSON data from a file and optionally limit the number of records returned."""
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data[:limit] if limit else data


# Function to prepare the input text batch for the model
def prepare_input_text(papers, fields):
    text_batch = []
    for paper in papers:
        text_parts = []
        for field in fields:
            if field in paper and paper[field]:  # Check if field exists and has content
                text_parts.append(paper[field][0])  # Assume each field is a list of strings
        text_batch.append(tokenizer.sep_token.join(text_parts))
    return text_batch


# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
model = AutoAdapterModel.from_pretrained('allenai/specter2_base')

# Load and activate the adapter
model.load_adapter("allenai/specter2", source="hf", load_as="specter2_classification", set_active=True)

# Load papers from JSON file, limiting to 100 entries
papers = load_data(f'{path_to_data}', limit=20)

# Prepare the input for the model
text_batch = prepare_input_text(papers, input_fields)
inputs = tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False, max_length=512)

# Generate embeddings using the model
output = model(**inputs)
embeddings = output.last_hidden_state[:, 0, :]  # Extract embeddings

# Optionally, print or further process the embeddings
print(embeddings)

# Paths for saving the embeddings
output_path_detached = 'embeddings/embedding_F_R.npy'  # For the detached numpy array
#output_path_full = 'embeddings/embedding_A_T_full.pt'  # For the full PyTorch tensor

# Ensure the directory exists for the first file
os.makedirs(os.path.dirname(output_path_detached), exist_ok=True)

# Ensure the directory exists for the second file
#os.makedirs(os.path.dirname(output_path_full), exist_ok=True)

# Save the embeddings after detaching and converting to numpy
np.save(output_path_detached, embeddings.detach().numpy())

# Save the full embeddings with computation graph using torch.save
#torch.save(embeddings, output_path_full)