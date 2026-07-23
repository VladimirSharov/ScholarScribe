import os
import json
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle

# Create debug output directory
DEBUG_DIR = "comprehensive_debug_output1"
os.makedirs(DEBUG_DIR, exist_ok=True)

# Load test dataset
test_data = Dataset.from_json("data_split_v4/full_dataset_v3_train.json")

# Load model, tokenizer, and tag2id mapping
model_name = "xlm-roberta-base"  # Replace with your trained model name or path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained("model_embedding/model_output_1n_20250227_232233/checkpoint-668")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load tag2id mapping
with open("model_embedding/model_output_1n_20250227_232233/tag_mapping_20250227_232238.pickle", "rb") as f:
    tag2id = pickle.load(f)
id2tag = {v: k for k, v in tag2id.items()}

# Comprehensive debug function
def debug_single_example(dataset, index=1):
    # Get a single example
    example = dataset[index]
    
    # Combine thesis title and abstract
    text = f"{example['thesis_title'][0]}. {example['abstract'][0]}"
    
    # Prepare debug dictionary
    debug_info = {
        "input": {
            "text": text,
            "original_example": {key: value for key, value in example.items()}
        }
    }
    
    # Tokenize the text
    tokenized_input = tokenizer(
        text, 
        truncation=True, 
        padding=True, 
        max_length=512, 
        return_tensors="pt"
    )
    
    # Move to device
    input_ids = tokenized_input['input_ids'].to(device)
    attention_mask = tokenized_input['attention_mask'].to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get logits
        logits = outputs.logits.cpu().numpy()[0]
        
        # Calculate probabilities
        probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
        
        # Binary predictions
        binary_predictions = (probabilities >= 0.14).astype(int)
        
        # Get predicted tags
        predicted_tags = [id2tag[idx] for idx, val in enumerate(binary_predictions) if val == 1]
        
        # Prepare debug information
        debug_info["model_output"] = {
            "logits": logits.tolist(),
            "probabilities": probabilities.tolist(),
            "binary_predictions": binary_predictions.tolist(),
            "predicted_tags": predicted_tags
        }
        
        # Prepare detailed tag information
        tag_details = []
        for idx, (prob, binary_pred) in enumerate(zip(probabilities, binary_predictions)):
            tag_details.append({
                "tag": id2tag.get(idx, f"Unknown_Tag_{idx}"),
                "probability": float(prob),
                "binary_prediction": int(binary_pred)
            })
        debug_info["tag_details"] = tag_details
    
    # Write debug information to files
    # Full debug JSON
    with open(os.path.join(DEBUG_DIR, f"debug_example_{index}.json"), "w", encoding="utf-8") as f:
        json.dump(debug_info, f, indent=2, ensure_ascii=False)
    
    # Separate files for easier viewing
    with open(os.path.join(DEBUG_DIR, f"input_text_{index}.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    
    with open(os.path.join(DEBUG_DIR, f"tag_probabilities_{index}.csv"), "w", encoding="utf-8") as f:
        f.write("Tag,Probability,Binary Prediction\n")
        for detail in tag_details:
            f.write(f"{detail['tag']},{detail['probability']},{detail['binary_prediction']}\n")
    
    print(f"Debug output written to {DEBUG_DIR}")
    
    return debug_info

# Run the debug function
result = debug_single_example(test_data)