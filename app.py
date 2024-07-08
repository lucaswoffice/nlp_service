from flask import Flask, request, jsonify, abort
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import os

app = Flask(__name__)
nlp = spacy.load("pt_core_news_lg")

# Defina a chave de API
API_KEY = '58663dd6-07c7-405d-8abf-cbad6aac7390'

def normalize_text(text):
    return text.lower()

def preprocess_text(text):
    doc = nlp(text)
    lemmatized_text = []
    for token in doc:
        if not token.is_stop and not token.is_punct:
            lemmatized_text.append(token.lemma_ if token.is_alpha else token.text)
    return " ".join(lemmatized_text)

def segment_text(text):
    text = normalize_text(text)
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences

def extract_entities(text):
    text = normalize_text(text)
    doc = nlp(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return entities

def calculate_tfidf(texts):
    preprocessed_texts = texts
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))  # Configurando para considerar unigrams, bigrams e trigrams
    X = vectorizer.fit_transform(preprocessed_texts)
    feature_names = vectorizer.get_feature_names_out()
    scores = X.toarray()
    tfidf = []
    for doc_scores in scores:
        tfidf.append([{ "text": feature_names[i], "score": score } for i, score in enumerate(doc_scores) if score > 0])
    return tfidf

def require_api_key(func):
    def wrapper_func(*args, **kwargs):
        api_key = request.headers.get('x-api-key')
        if api_key and api_key == API_KEY:
            return func(*args, **kwargs)
        else:
            abort(401, description="Unauthorized")
    wrapper_func.__name__ = func.__name__
    return wrapper_func

@app.route('/segment', methods=['POST'])
@require_api_key
def segment():
    text = request.json.get('text')
    sentences = segment_text(text)
    return jsonify(sentences)

@app.route('/entities', methods=['POST'])
@require_api_key
def entities():
    text = request.json.get('text')
    entities = extract_entities(text)
    return jsonify(entities)

@app.route('/tfidf', methods=['POST'])
@require_api_key
def tfidf():
    texts = request.json.get('texts')
    tfidf_scores = calculate_tfidf(texts)
    return jsonify(tfidf_scores)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
