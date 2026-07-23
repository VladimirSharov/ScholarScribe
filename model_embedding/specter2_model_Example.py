from transformers import AutoTokenizer
from adapters import AutoAdapterModel

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
model = AutoAdapterModel.from_pretrained('allenai/specter2_base')

# Load and activate the adapter
model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)

# Define the input papers
papers = [
    {'title': 'BERT', 'abstract': 'We introduce a new language representation model called BERT'},
    {'title': 'Attention is all you need', 'abstract': 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks'}
]

# Prepare the input for the model
text_batch = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in papers]
inputs = tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False, max_length=512)

# Generate embeddings using the model
output = model(**inputs)
embeddings = output.last_hidden_state[:, 0, :]  # Extract embeddings

# Optionally, print or further process the embeddings
print(embeddings)
