# Web-Crawling-Information-Retrieval-System-Python-NLP-VSM-QLM
Developed a web-based search engine using Selenium, BeautifulSoup, NLP preprocessing, and inverted indexing; implemented Vector Space Model (VSM) with TF-IDF and cosine similarity, and Query Likelihood Model (QLM) with Jelinek-Mercer smoothing for document ranking through an interactive Dash interface.

# Web Crawling & Information Retrieval System

<img width="1872" height="887" alt="QLM" src="https://github.com/user-attachments/assets/0b33edcd-17ee-40a2-84e5-998e781a1e70" />
<img width="1851" height="1056" alt="VSM" src="https://github.com/user-attachments/assets/7ac532e6-7d97-4c58-b024-55f9f6d39abe" />

### Python | Web Scraping | NLP | Inverted Indexing | Search Engine Algorithms

## Project Overview

Developed a Python-based **Information Retrieval (IR) System** that automates web crawling, extracts and processes webpage content, builds an inverted index, and retrieves relevant documents using two information retrieval models.

The system integrates **Vector Space Model (VSM), TF-IDF, Query Likelihood Model (QLM), Jelinek–Mercer Smoothing, and Cosine Similarity** through an interactive Dash-based search interface.

## Key Features & Techniques

- **Automated Web Crawling:** Selenium WebDriver, headless Chrome, BeautifulSoup and HTML content extraction.
- **Document Indexing:** Inverted Index construction, JSON serialization and document-term mapping.
- **NLP Preprocessing:** Tokenization, lowercasing, stop-word removal and Porter stemming using NLTK.
- **Vector Space Model (VSM):** TF-IDF weighting with unigram and bigram features.
- **Document Similarity:** Cosine Similarity for relevance ranking.
- **Query Likelihood Model (QLM):** Probabilistic retrieval with Jelinek–Mercer smoothing.
- **Search Ranking:** Top 10 relevant document results with similarity/likelihood scores.
- **Interactive Interface:** Python Dash application with separate VSM and QLM search results.

## Model Configuration

| Component | Implementation |
|---|---|
| Web Crawling | Selenium + BeautifulSoup |
| Crawl Limit | 250 Pages |
| Text Processing | NLTK |
| Indexing | Inverted Index |
| Index Storage | JSON |
| VSM | TF-IDF |
| N-Gram Range | 1–2 |
| Similarity Metric | Cosine Similarity |
| QLM | Jelinek–Mercer Smoothing |
| Smoothing Parameter | Lambda = 0.5 |
| Search Results | Top 10 Documents |
| Interface | Python Dash + CSS |

## Technology Stack

**Programming:** Python, NumPy

**Web Scraping:** Selenium, BeautifulSoup, Requests

**Natural Language Processing:** NLTK, Porter Stemmer, Tokenization, Stop-Word Removal

**Information Retrieval:** Inverted Index, TF-IDF, Vector Space Model, Query Likelihood Model, Jelinek–Mercer Smoothing, Cosine Similarity

**Application Development:** Dash, HTML, CSS, JSON

## System Architecture

Web Crawling (Selenium + BeautifulSoup)
→ HTML Text Extraction
→ NLP Preprocessing
→ Inverted Index Construction
→ TF-IDF / Query Likelihood Modelling
→ Document Relevance Ranking
→ Interactive Dash Search Interface

## Project Outcomes

- Implemented automated web crawling and text extraction.
- Developed an inverted index for document retrieval.
- Implemented and compared vector-space and probabilistic search ranking methods.
- Created an interactive search interface displaying ranked results and document previews.

## Note

This project is an academic information retrieval prototype. Crawling is configured for up to 250 pages and depends on website accessibility.

---

**Developed by Nirmalan Arjun and project collaborators**
