import os
import nltk
from gensim.models import Word2Vec
import numpy as np
from elasticsearch import Elasticsearch
from sklearn.metrics.pairwise import cosine_similarity
from hnswlib import Index

ELASTICSEARCH_URL = 'https://elastic:ML4SEc-8wHtqH2W067cC@localhost:9200'
es = Elasticsearch([ELASTICSEARCH_URL], verify_certs=False)
index_name = 'citeseer_papers_index_wva'

# Load existing Word2Vec model or train if necessary
w2v_model = Word2Vec.load('C:\Users\harsh\Downloads\word2vec_swahili_1\saved_model.pb')


def get_all_embeddings():
    all_embeddings = []
    all_ids = []
    for doc in es.search(index=index_name, _source=['embedding'], size=1000)['hits']['hits']:
        all_embeddings.append(doc['_source']['embedding'])
        all_ids.append(doc['_id'])
    return all_embeddings, all_ids


doc_embeddings, doc_ids = get_all_embeddings()

# Initialize HNSW index
dim = 300  # Dimensionality of the embeddings
hnsw_index = Index(space='l2', dim=dim)

# Add items to HNSW
hnsw_index.init_index(max_elements=len(doc_embeddings), ef_construction=200, M=16)
hnsw_index.set_ef(50)

for idx, embedding in enumerate(doc_embeddings):
    hnsw_index.add_item(embedding, idx)

# Build HNSW index
hnsw_index.build(missing_data_cover_trees=True)

# Example usage: Perform a search using HNSW and Elasticsearch
query = "machine learning"
query_words = nltk.word_tokenize(query.lower())
query_embedding = np.mean([w2v_model.wv[word] for word in query_words if word in w2v_model.wv], axis=0)

num_neighbors = 5
nearest_neighbors, distances = hnsw_index.knn_query(query_embedding, k=num_neighbors)

# Retrieve corresponding documents from Elasticsearch using the nearest neighbor IDs
similar_ids = [doc_ids[idx] for idx in nearest_neighbors]
print(similar_ids)
