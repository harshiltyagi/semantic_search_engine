from elasticsearch import Elasticsearch

# Replace these with your Elasticsearch server details
ELASTICSEARCH_URL = 'https://elastic:ML4SEc-8wHtqH2W067cC@localhost:9200'
es = Elasticsearch([ELASTICSEARCH_URL], verify_certs=False)

# Name of the index to delete
index_name = 'citeseer_papers_index_v2'  # Replace with your actual index name

# Delete the previous index (if it exists)
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)
    print(f"Index '{index_name}' deleted.")
else:
    print(f"Index '{index_name}' doesn't exist.")
