from annoy import AnnoyIndex
import os
import nltk
from gensim.models import Word2Vec
import numpy as np
from elasticsearch import Elasticsearch
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Annoy index
num_trees = 10
dim = 300  # Dimensionality of the embeddings
annoy_index = AnnoyIndex(dim, 'angular')

# Initialize Elasticsearch and other settings
ELASTICSEARCH_URL = 'https://elastic:ML4SEc-8wHtqH2W067cC@localhost:9200'
es = Elasticsearch([ELASTICSEARCH_URL], verify_certs=False)
index_name = 'citeseer_papers_index_wva'

w2v_model = Word2Vec.load('C:\Users\harsh\Downloads\word2vec_swahili_1\saved_model.pb')


def get_all_embeddings():
    all_embeddings = []
    all_ids = []
    for doc in es.search(index=index_name, _source=['embedding'], size=1000)['hits']['hits']:
        all_embeddings.append(doc['_source']['embedding'])
        all_ids.append(doc['_id'])
    return all_embeddings, all_ids


doc_embeddings, doc_ids = get_all_embeddings()
for idx, embedding in enumerate(doc_embeddings):
    annoy_index.add_item(idx, embedding)

# Build the Annoy index
annoy_index.build(num_trees)

# Example usage: Perform a search using Annoy and Elasticsearch
query = "query text"
query_words = nltk.word_tokenize(query.lower())
query_embedding = np.mean([w2v_model.wv[word] for word in query_words if word in w2v_model.wv], axis=0)

num_neighbors = 5  # Number of nearest neighbors to retrieve
nearest_neighbors = annoy_index.get_nns_by_vector(query_embedding, num_neighbors)

# Corresponding documents from Elasticsearch using the nearest neighbor IDs
similar_ids = [doc_ids[idx] for idx in nearest_neighbors]
print(similar_ids)
