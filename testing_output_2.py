from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import numpy as np
import json
import sys

# TODO write thesis, pipeline, 27? check meeeing, before analysis: trends

# Load test dataset
test_data = Dataset.from_json("data_split_v4/full_dataset_v3_train.json")

# Load model, tokenizer, and tag2id mapping
model_name = "xlm-roberta-base"  # Replace with your trained model name or path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained("model_embedding/model_output_1n_20250228_131012/checkpoint-668")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load tag2id mapping
with open("model_embedding/model_output_1n_20250228_131012/tag_mapping_20250228_131016.pickle", "rb") as f:
    tag2id = pickle.load(f)
id2tag = {v: k for k, v in tag2id.items()}

# Debug a single example
def debug_single_example(example):
    # Get a single example
    # example = dataset[index]
    
    # Combine thesis title and abstract
    # text = "The impact of electronic radiation."
    text = f"{example['thesis_title'][0]}. {example['abstract'][0]}"
    
    print("Input Text:")
    print(text)
    print("\n--- Original Example Fields ---")
    for key, value in example.items():
        print(f"{key}: {value}")
    length = len(example['additional_tags']) + len(example['subject_tags'])      
    # Tokenize the text
    tokenized_input = tokenizer(
        text, 
        truncation=True, 
        padding=True, 
        max_length=512, 
        return_tensors="pt"
    )
    # print(tokenized_input)
    # sys.exit()
    # Move to device
    input_ids = tokenized_input['input_ids'].to(device)
    attention_mask = tokenized_input['attention_mask'].to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get logits
        logits = outputs.logits.cpu()[0]
        
        # Calculate probabilities
        # probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
        # probabilities = (torch.tensor(logits))
        # Binary predictions
        # binary_predictions = (probabilities >= 0.18).astype(int)
        binary_predictions = torch.argsort(logits, descending = True)[:length]
        # Get predicted tags
        predicted_tags = [id2tag[int(idx)] for idx in (binary_predictions)]
        '''
        print("\n--- Model Output ---")
        # print("Probabilities:", probabilities)
        # print("Logits:", logits)
        # print("Binary Predictions:", np.nonzero(binary_predictions))
        print("Binary Predictions:", (binary_predictions))
        '''
        print("Predicted Tags:", predicted_tags)
        return predicted_tags
        
# Run the debug function
predictions = []

for i in range(10):
    example= test_data[i]
    print(example)
    abc = [
          debug_single_example(example),
          example['identifier']
    ]
    predictions.append(abc)


with open('xlm_roberta_prediction5.json', 'w', encoding='utf-8') as file:
    json.dump(predictions, file, indent=4, ensure_ascii=False)
