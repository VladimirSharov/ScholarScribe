from transformers import AutoTokenizer
from adapters import AutoAdapterModel

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
model = AutoAdapterModel.from_pretrained('allenai/specter2_base')

# Load and activate the adapter
model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)

# Define the input papers
papers = [
    {
        "thesis_title": "Alakouluikäisten käyttäytymisen ja tunteiden hallinnan yhteys minäpystyvyyteen",
        "abstract": "Tässä Pro gradu -tutkielmassa tarkasteltiin alakouluikäisten 1) minäpystyvyyden\nyhteyttä 2) vahvuuteen ihmissuhteissa ja 3) vahvuuteen tunne-elämän hallinnassa.\nTutkittavat olivat 3. ja 4. luokkalaisia. Lisäksi tarkasteltiin sukupuolen\nja luokka-asteen yhteyttä kyseisiin muuttujiin.\nTutkimusaineisto oli osa Jyväskylän yliopiston ja Niilo Mäki instituutin yhteistä\nminäpystyvyy... (trimmed for brevity)"
    }
]

# Prepare the input for the model
text_batch = [d['thesis_title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in papers]
inputs = tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False, max_length=512)

# Generate embeddings using the model
output = model(**inputs)
embeddings = output.last_hidden_state[:, 0, :]  # Extract embeddings

# Optionally, print or further process the embeddings
print(embeddings)
