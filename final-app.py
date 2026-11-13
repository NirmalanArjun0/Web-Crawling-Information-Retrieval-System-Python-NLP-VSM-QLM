import dash
from dash import dcc, html, Input, Output, State
import requests
from bs4 import BeautifulSoup
import time
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import random
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# NLTK data files downld with tokenizer model punkt 
nltk.download('punkt')
nltk.download('stopwords')


# initializing with the starting page and assigned th emax pg as 250 to crwl
seed_url = "https://dailydozes.com/" 
max_pages = 250
save_dir = "crawled_pages"
os.makedirs(save_dir, exist_ok=True)

# selenium WebDriver for headless browsing
options = webdriver.ChromeOptions()
options.add_argument('--headless')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Crawling process headless browser
def crawl(url, page_num, visited_urls):
    unwanted_keywords = ["share", "sharer", "facebook.com", "twitter.com", "linkedin.com", "pinterest.com"]
    
    if any(keyword in url for keyword in unwanted_keywords):
        print(f"Skipping social media or share link: {url}")
        return set()

#load the page in browser until previous content to finish loading
    try:
        print(f"Crawling page {page_num}: {url}")
        driver.get(url)
        time.sleep(3) 

#take all html of the particular page as string by extract data from web page
        page_content = driver.page_source
        soup = BeautifulSoup(page_content, 'html.parser')
        
        text_content = soup.get_text(separator=" ", strip=True)
        save_page(text_content, page_num)

# to keep it as uniques
        links = set()
        for anchor in soup.find_all('a', href=True):
            full_url = anchor['href']
            if full_url.startswith('http') and full_url not in visited_urls:
                links.add(full_url)

        return links
        
    except Exception as e:
        print(f"Failed to crawl {url}: {e}")
        return set()

# Save page content as html page
def save_page(content, page_num):
    file_path = os.path.join(save_dir, f"page_{page_num}.html")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

# startin the crawling prcess, start with 0 to get exact count to track how many pgs crawled
# to reduce the ferq of req to the server delay point mentioned
def main_crawl(seed_url, max_pages):
    to_crawl = [seed_url]
    visited_urls = set()
    page_count = 0
    while to_crawl and page_count < max_pages:
        current_url = to_crawl.pop(0)
        if current_url in visited_urls:
            continue
        new_links = crawl(current_url, page_count, visited_urls)
        to_crawl.extend(new_links)
        visited_urls.add(current_url)
        page_count += 1
        time.sleep(random.uniform(1, 2))




# Indexing

# Tokenize and normalize text, with steam to reduce word tot heir main word.
def tokenize_and_normalize(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens



# Build inverted index
def build_inverted_index():
    inverted_index = defaultdict(list)
    for i in range(max_pages):
        file_path = os.path.join(save_dir, f"page_{i}.html")
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping.")
            continue
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            tokens = tokenize_and_normalize(content)
            for token in set(tokens):
                inverted_index[token].append(i)
    return inverted_index

# Save inverted index
def save_inverted_index(inverted_index):
    with open("inverted_index.json", "w", encoding="utf-8") as f:
        json.dump(inverted_index, f, indent=4)

# Load documents
def load_documents():
    documents = []
    for i in range(max_pages):
        file_path = os.path.join(save_dir, f"page_{i}.html")
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            tokens = tokenize_and_normalize(content)
            documents.append(" ".join(tokens))
    return documents



# VSM model

def compute_tfidf(documents):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix

def query_to_vector(query, vectorizer):
    tokens = tokenize_and_normalize(query)
    query_string = " ".join(tokens)
    return vectorizer.transform([query_string])

def rank_documents(query_vector, tfidf_matrix):
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranked_doc_indices = np.argsort(cosine_similarities)[::-1]
    return ranked_doc_indices, cosine_similarities



# QLM Model

def compute_term_frequencies(documents):
    term_frequencies = []
    for doc in documents:
        tokens = doc.split()
        freq_dict = defaultdict(int)
        for token in tokens:
            freq_dict[token] += 1
        term_frequencies.append(freq_dict)
    return term_frequencies

def compute_collection_frequency(term_frequencies):
    collection_freq = defaultdict(int)
    for freq_dict in term_frequencies:
        for term, freq in freq_dict.items():
            collection_freq[term] += freq
    return collection_freq

def calculate_document_lengths(term_frequencies):
    doc_lengths = [sum(freq_dict.values()) for freq_dict in term_frequencies]
    collection_length = sum(doc_lengths)
    return doc_lengths, collection_length

def jelinek_mercer_smoothing(term, doc_freq, doc_length, collection_freq, collection_length, lambda_param=0.5):
    p_td = doc_freq / doc_length if doc_length > 0 else 0
    p_tc = collection_freq / collection_length if collection_length > 0 else 0
    return (1 - lambda_param) * p_td + lambda_param * p_tc

def compute_query_likelihood(query_tokens, term_frequencies, doc_lengths, collection_freq, collection_length, lambda_param=0.5):
    scores = []
    for doc_index, freq_dict in enumerate(term_frequencies):
        doc_length = doc_lengths[doc_index]
        score = 1.0
        for term in query_tokens:
            doc_freq = freq_dict.get(term, 0)
            collection_term_freq = collection_freq.get(term, 0)
            smoothed_prob = jelinek_mercer_smoothing(term, doc_freq, doc_length, collection_term_freq, collection_length, lambda_param)
            score *= smoothed_prob
        scores.append((doc_index, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores



# Final Appication to adjust estimates for terms like unseen words smoothing.

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__)
app.title = "Information Retrieval System"

app.layout = html.Div([
    html.H1("Information Retrieval System", style={'text-align': 'center'}),
    html.Div([
        html.Label("Enter your search query:"),
        dcc.Input(id="query-input", type="text", placeholder="Enter query here", style={'width': '100%'}),
        html.Br(),
        html.Button("Search", id="search-button", n_clicks=0, style={'margin-top': '10px'})
    ], style={'width': '50%', 'margin': 'auto'}),

    html.Div(id="tabs-content"),
    html.Br(),
    dcc.Tabs([
        dcc.Tab(label='VSM Results', children=[
            html.Div(id="vsm-results")
        ]),
        dcc.Tab(label='QLM Results', children=[
            html.Div(id="qlm-results")
        ])
    ])
])

# Callback to handle search and display results
@app.callback(
    [Output("vsm-results", "children"),
     Output("qlm-results", "children")],
    [Input("search-button", "n_clicks")],
    [State("query-input", "value")]
)
def display_results(n_clicks, query):
    if n_clicks > 0 and query:
        documents = load_documents()
        
        # VSM Results
        vectorizer, tfidf_matrix = compute_tfidf(documents)
        query_vector = query_to_vector(query, vectorizer)
        ranked_doc_indices, cosine_similarities = rank_documents(query_vector, tfidf_matrix)

        vsm_result = []
        for rank, index in enumerate(ranked_doc_indices[:10]):
            similarity_score = cosine_similarities[index]
            doc_path = os.path.join(save_dir, f"page_{index}.html")
            with open(doc_path, 'r', encoding='utf-8') as file:
                content = file.read()
                preview = content[:300]
            vsm_result.append(html.Div([
                html.H4(f"Rank {rank + 1}: Document {index}"),
                html.P(f"Similarity Score: {similarity_score:.4f}"),
                html.P(f"Preview: {preview}...")
            ]))

        # to display QLM Results 
        lambda_param = 0.5
        term_frequencies = compute_term_frequencies(documents)
        collection_freq = compute_collection_frequency(term_frequencies)
        doc_lengths, collection_length = calculate_document_lengths(term_frequencies)
        
        query_tokens = tokenize_and_normalize(query)
        scores = compute_query_likelihood(query_tokens, term_frequencies, doc_lengths, collection_freq, collection_length, lambda_param)

        qlm_result = []
        for rank, (doc_index, score) in enumerate(scores[:10]):
            doc_path = os.path.join(save_dir, f"page_{doc_index}.html")
            with open(doc_path, 'r', encoding='utf-8') as file:
                content = file.read()
                preview = content[:300]
            qlm_result.append(html.Div([
                html.H4(f"Rank {rank + 1}: Document {doc_index}"),
                html.P(f"Likelihood Score: {score:.6f}"),
                html.P(f"Preview: {preview}...")
            ]))

        return vsm_result, qlm_result
    return "", ""

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)

