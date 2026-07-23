from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

def analyze_model(model_path, model_name=""):
    print(f"\n--- Analyzing {model_name} at {model_path} ---")
    
    try:
        # Load model with Hugging Face methods
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Analyze linear layers
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                print(f"Layer: {name}")
                print(f"  Weight shape: {module.weight.shape}")
                print(f"  Weight mean abs: {torch.mean(torch.abs(module.weight)).item():.6f}")
                print(f"  Weight variance: {torch.var(module.weight).item():.6f}")
                print(f"  Weight min: {torch.min(module.weight).item():.6f}, max: {torch.max(module.weight).item():.6f}")
                if module.bias is not None:
                    print(f"  Bias mean abs: {torch.mean(torch.abs(module.bias)).item():.6f}")
                    print(f"  Bias variance: {torch.var(module.bias).item():.6f}")
                    print(f"  Bias min: {torch.min(module.bias).item():.6f}, max: {torch.max(module.bias).item():.6f}")
                print()
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Check your fine-tuned model
finetuned_path = "model_embedding/model_output_1n_20250228_131012/checkpoint-668"
analyze_model(finetuned_path, "Fine-tuned model")

# For the original pre-trained model
# You'll need to determine which base model you used
# Common options might be:
pretrained_models = [
    "microsoft/deberta-v3-base",
    "roberta-base",
    "bert-base-uncased",
    "xlm-roberta-base"
]

# Try to determine which base model was used from config
import json
try:
    with open(f"{finetuned_path}/config.json", "r") as f:
        config = json.load(f)
        if "_name_or_path" in config:
            base_model = config["_name_or_path"]
            print(f"Found base model from config: {base_model}")
            pretrained_models = [base_model]
except Exception as e:
    print(f"Couldn't determine base model from config: {e}")

# Analyze original pre-trained model
for model_name in pretrained_models:
    print(f"\nTrying to load base model: {model_name}")
    base_model = analyze_model(model_name, "Base pre-trained model")
    if base_model:
        break