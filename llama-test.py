from llama_cpp import Llama

# This assumes you have a compatible model downloaded (e.g., model.gguf or model.ggml)
llm = Llama(model_path="llama_model/Hermes-3-Llama-3.1-8B.Q4_K_M.gguf")
output = llm("Who are you?")
print(output)
