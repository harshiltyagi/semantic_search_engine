import os
import re
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from elasticsearch import Elasticsearch
from gensim.models import Word2Vec

# Initialize Elasticsearch and other settings
ELASTICSEARCH_URL = 'https://elastic:ML4SEc-8wHtqH2W067cC@localhost:9200'
es = Elasticsearch([ELASTICSEARCH_URL], verify_certs=False)
index_name = 'citeseer_papers_index_wv'

# Initialize NLTK and download resources
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

# Path to the folder containing Citeseer documents
documents_folder = r'C:\Users\harsh\Downloads\citeseer2.tar\2'

# Training Word2Vec model
sentences = []
for root, dirs, files in os.walk(documents_folder):
    for file_name in files:
        if file_name.endswith('.txt'):
            file_path = os.path.join(root, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().lower()
                words = nltk.word_tokenize(content)
                words = [word for word in words if word.isalpha() and word not in stop_words]
                sentences.append(words)

# Train Word2Vec model on the corpus
w2v_model = Word2Vec(sentences, vector_size=300, window=5, min_count=1, sg=0)  # Adjust parameters as needed

# Indexing documents into Elasticsearch with metadata and embeddings
for root, dirs, files in os.walk(documents_folder):
    for file_name in files:
        if file_name.endswith('.txt'):
            file_path = os.path.join(root, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().lower()
                words = nltk.word_tokenize(content)
                words = [word for word in words if word.isalpha() and word not in stop_words]
                doc_embedding = np.mean([w2v_model.wv[word] for word in words if word in w2v_model.wv], axis=0)
                es.index(
                    index=index_name,
                    body={
                        'content': content,
                        'embedding': doc_embedding.tolist(),
                    }
                )


# Function to perform similarity search using brute-force cosine similarity
def brute_force_search(query_embedding, threshold=0.5):
    all_embeddings = []  # Collect all embeddings
    all_ids = []  # Corresponding document IDs
    for doc in es.search(index=index_name, _source=['embedding'], size=1000)['hits']['hits']:
        all_embeddings.append(doc['_source']['embedding'])
        all_ids.append(doc['_id'])

    # Calculate cosine similarity between the query and all embeddings
    similarities = cosine_similarity([query_embedding], all_embeddings)[0]

    # Retrieve IDs of documents with similarity above threshold
    similar_ids = [all_ids[i] for i, sim in enumerate(similarities) if sim > threshold]
    return similar_ids


# Example usage: Perform a search using Elasticsearch and brute-force similarity
query = "machine learning"
query_words = nltk.word_tokenize(query.lower())
query_embedding = np.mean([w2v_model.wv[word] for word in query_words if word in w2v_model.wv], axis=0)

# Use brute-force cosine similarity search based on the query embedding
similar_ids = brute_force_search(query_embedding)
print(similar_ids)
