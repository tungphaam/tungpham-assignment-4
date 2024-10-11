from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)


# Initialize Flask app
app = Flask(__name__)

# Load dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Preprocess the dataset
stop_words = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(documents)

# Apply SVD for dimensionality reduction (LSA)
svd = TruncatedSVD(n_components=100) # Adjust n if needed
X_reduced = svd.fit_transform(X)

# Define search function
def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # Transform query using the same vectorizer and reduce its dimensionality
    query_vector = vectorizer.transform([query])
    query_reduced = svd.transform(query_vector)

    # Compute cosine similarity between the query and all documents
    similarities = cosine_similarity(query_reduced, X_reduced)[0]

    # Get top 5 documents with highest cosine similarity scores
    top_indices = np.argsort(similarities)[-5:][::-1]
    top_documents = [documents[i] for i in top_indices]
    top_similarities = similarities[top_indices]

    return top_documents, top_similarities.tolist(), top_indices.tolist()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
