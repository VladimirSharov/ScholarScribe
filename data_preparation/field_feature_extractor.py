from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import json
import re
import nltk

# Download NLTK stop words if not already present
nltk.download('stopwords')

# Initialize stemmer and stop words for both English and Finnish
stemmer = PorterStemmer()
stop_words_en = set(stopwords.words('english'))
stop_words_fi = set(['ja', 'on', 'oli', 'että', 'kuin'])  # Example Finnish stop words, add more as needed

# Function to preprocess text (lowercasing, removing punctuation, stop words, and stemming)
def preprocess_text(text):
    # Remove punctuation
    text = re.sub(r'\W+', ' ', text)
    # Tokenize and lower case
    tokens = text.lower().split()
    # Remove stop words and apply stemming
    processed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words_en and token not in stop_words_fi]
    return ' '.join(processed_tokens)

# Function to apply TF-IDF on specified fields
def apply_tfidf(input_path, output_path, fields_to_process, max_features=500):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Process each field separately (like 'thesis_title', 'abstract', etc.)
    for field in fields_to_process:
        # Extract and preprocess field data
        documents = [preprocess_text(" ".join(entity.get(field, []))) for entity in data]

        # Initialize TF-IDF vectorizer with stop word removal, bigrams, and specified max features
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=max_features)
        tfidf_matrix = vectorizer.fit_transform(documents)

        # Get feature names (terms) and corresponding scores
        feature_names = vectorizer.get_feature_names_out()

        # Store results in the entities
        for i, entity in enumerate(data):
            # Only update entities that have the specified field
            if field in entity:
                # Convert the TF-IDF score for the document into a dictionary
                tfidf_compressed = {feature_names[idx]: score for idx, score in zip(tfidf_matrix[i].indices, tfidf_matrix[i].data)}
                entity[f'tfidf_{field}'] = tfidf_compressed

    # Save the new dataset with TF-IDF applied
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_path = 'data_preparation/dataset/full_dataset.json'
    output_path = 'data_preparation/tempFieldMod/full_dataset_e.json'

    # Specify the fields you want to process, e.g., 'thesis_title' and 'abstract'
    fields_to_process = ['thesis_title', 'abstract']

    # Apply TF-IDF with a larger number of features (e.g., 500)
    apply_tfidf(input_path, output_path, fields_to_process, max_features=500)
