import os
import re
import nltk
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from elasticsearch import Elasticsearch

# Initialize Elasticsearch and other settings
ELASTICSEARCH_URL = 'https://elastic:ML4SEc-8wHtqH2W067cC@localhost:9200'
es = Elasticsearch([ELASTICSEARCH_URL], verify_certs=False)
index_name = 'citeseer_papers_index_bert'

# Initialize NLTK and download resources
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

# Path to the folder containing Citeseer documents
documents_folder = r'C:\Users\harsh\Downloads\citeseer2.tar\2'

# Load BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = torch.mean(outputs.last_hidden_state, dim=1)  # Mean pooling of token embeddings
    return embeddings.numpy()


# Indexing documents into Elasticsearch with metadata and embeddings
for root, dirs, files in os.walk(documents_folder):
    for file_name in files:
        if file_name.endswith('.txt'):
            file_path = os.path.join(root, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().lower()
                doc_embedding = get_bert_embedding(content)
                es.index(
                    index=index_name,
                    body={
                        'content': content,
                        'embedding': doc_embedding.tolist(),
                    }
                )


def brute_force_search(query_embedding, threshold=0.5):
    all_embeddings = []  # Collect all embeddings
    all_ids = []  # Corresponding document IDs
    for doc in es.search(index=index_name, _source=['embedding'], size=1000)['hits']['hits']:
        all_embeddings.append(doc['_source']['embedding'])
        all_ids.append(doc['_id'])
    similarities = cosine_similarity([query_embedding], all_embeddings)[0]
    similar_ids = [all_ids[i] for i, sim in enumerate(similarities) if sim > threshold]
    return similar_ids


# Example usage: Perform a search using Elasticsearch and brute-force similarity
query = "query text"
query_embedding = get_bert_embedding(query.lower())

similar_ids = brute_force_search(query_embedding)
print(similar_ids)
