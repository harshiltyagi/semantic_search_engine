from flask import Flask, render_template, request, jsonify
from elasticsearch import Elasticsearch

app = Flask(__name__)

ELASTICSEARCH_URL = 'https://elastic:ML4SEc-8wHtqH2W067cC@localhost:9200'
es = Elasticsearch([ELASTICSEARCH_URL], verify_certs=False)
index_name = 'citeseer_papers_index'  # Replace with your index name

@app.route('/', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form['query']
        year_filter = request.form['year_filter']
        keyword_filter = request.form['keyword_filter']
        page = int(request.form.get('page', 1))

        # Elasticsearch query dynamically
        must_query = {"multi_match": {"query": query, "fields": ["title", "content"]}}
        filter_query = []

        if year_filter:
            filter_query.append({"term": {"published_year": year_filter}})

        if keyword_filter:
            filter_query.append({"terms": {"keywords": [keyword_filter]}})

        search_body = {"query": {"bool": {"must": must_query}}}

        if filter_query:
            search_body['query']['bool']['filter'] = {"bool": {"must": filter_query}}

        page_size = 10  # Number of results per page
        from_record = (page - 1) * page_size
        size = page_size

        search_body['from'] = from_record
        search_body['size'] = size

        search_results = es.search(index=index_name, body=search_body)['hits']
        total_hits = search_results['total']['value']
        hits = search_results['hits']
        results = []
        for hit in hits:
            result = {
                'title': hit['_source']['title'],
                'content_snippet': hit['_source']['content'][:200],  # Show first 200 characters of content
                'score': hit['_score']  # The score for each document
            }
            results.append(result)

        next_page = page + 1 if from_record + size < total_hits else None

        return render_template(
            'search_results.html',
            results=results,
            query=query,
            year_filter=year_filter,
            keyword_filter=keyword_filter,
            next_page=next_page,
            total_hits=total_hits  # Pass the total number of hits to the template
        )

    return render_template('search.html')


@app.route('/suggest')
def suggest():
    partial_query = request.args.get('q')
    suggestion_list = []

    if partial_query:
        # Elasticsearch suggestion query
        suggestion_query = {
            "suggest": {
                "suggest_title": {
                    "prefix": partial_query,
                    "completion": {
                        "field": "title.suggest"
                    }
                }
            }
        }

        try:
            suggestions = es.search(index=index_name, body=suggestion_query)
            suggestion_list = [hit['text'] for hit in suggestions['suggest']['suggest_title'][0]['options']]
        except Exception as e:
            print(f"Error fetching suggestions: {e}")

    return jsonify({'suggestions': suggestion_list})


if __name__ == '__main__':
    app.run(debug=True)
