import os
import requests

# Define the model URL and the directory to store the model
model_url = "https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B-GGUF/resolve/main/Hermes-3-Llama-3.1-8B.Q4_K_M.gguf"
model_dir = "llama_model"
model_path = os.path.join(model_dir, "Hermes-3-Llama-3.1-8B.Q4_K_M.gguf")

# Create the directory if it doesn't exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Check if the model file already exists
if not os.path.exists(model_path):
    print("Downloading the model...")
    response = requests.get(model_url)
    with open(model_path, 'wb') as f:
        f.write(response.content)
    print("Model downloaded and saved to", model_path)
else:
    print("Model already exists at", model_path)

print("Setup complete.")
