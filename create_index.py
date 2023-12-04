import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from elasticsearch import Elasticsearch

# Replace these with your Elasticsearch server details
ELASTICSEARCH_URL = 'https://elastic:ML4SEc-8wHtqH2W067cC@localhost:9200'
es = Elasticsearch([ELASTICSEARCH_URL], verify_certs=False)

# Name of the index to create
index_name = 'citeseer_papers_index'  # Replace with your desired index name

# Path to the folder containing Citeseer documents
documents_folder = r'C:\Users\harsh\Downloads\citeseer2.tar\2'

# Create the index with settings and mappings (similar to previous code)

# Download NLTK resources (run once if not downloaded)
nltk.download('punkt')
nltk.download('stopwords')

# Initialize NLTK stopwords
stop_words = set(stopwords.words('english'))


# Function to extract year from the text
def extract_year_from_text(text):
    # Search for four-digit year patterns in the first 10 lines
    year_pattern = re.compile(r'\b\d{4}\b')  # Regex pattern for four-digit year
    lines = text.split('\n')[:10]  # Get first 10 lines
    for line in lines:
        match = re.search(year_pattern, line)
        if match:
            return match.group()  # Return the first four-digit year found
    return None  # Return None if no year is found in the first 10 lines


# Indexing documents into Elasticsearch with metadata and keywords
for root, dirs, files in os.walk(documents_folder):
    for file_name in files:
        if file_name.endswith('.txt'):
            file_path = os.path.join(root, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                first_line = file.readline().strip()  # Read the first line as the title
                remaining_content = file.read()  # Read the rest of the content

                # Extract keywords from the content (similar to previous code)
                words = word_tokenize(remaining_content)
                keywords = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
                top_keywords = list(set(keywords[:5]))  # Consider top 5 unique keywords

                # Extract publication year from the content
                extracted_year = extract_year_from_text(remaining_content)

                # Index document into Elasticsearch with metadata fields and keywords
                es.index(
                    index=index_name,
                    body={
                        'title': first_line,
                        'content': remaining_content,
                        'published_year': extracted_year,  # Include extracted year
                        'keywords': top_keywords
                        # Include other metadata fields and category keywords if needed
                    }
                )
