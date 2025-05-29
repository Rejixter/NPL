import pandas as pd
import os
from gensim.models import Word2Vec

# MODELLERİN KAYDEDİLECEĞİ KLASÖR
os.makedirs("models", exist_ok=True)

# Lemma ve stemmed verileri oku
lemma_df = pd.read_csv('books_lemmatized.csv')
stem_df  = pd.read_csv('books_stemmed.csv')

# Cümleleri tokenize et (split ile çünkü cümleler boşlukla ayrılmış)
lemma_sentences = [str(s).split() for s in lemma_df['lemmatized']]
stem_sentences  = [str(s).split() for s in stem_df['stemmed']]

# Model parametreleri
params = [
    {"data": "lemma", "arch": "cbow",     "window": 4,  "dim": 300,  "sg": 0},
    {"data": "lemma", "arch": "cbow",     "window": 4,  "dim": 1000, "sg": 0},
    {"data": "lemma", "arch": "cbow",     "window": 10, "dim": 300,  "sg": 0},
    {"data": "lemma", "arch": "cbow",     "window": 10, "dim": 1000, "sg": 0},
    {"data": "lemma", "arch": "skipgram", "window": 4,  "dim": 300,  "sg": 1},
    {"data": "lemma", "arch": "skipgram", "window": 4,  "dim": 1000, "sg": 1},
    {"data": "lemma", "arch": "skipgram", "window": 10, "dim": 300,  "sg": 1},
    {"data": "lemma", "arch": "skipgram", "window": 10, "dim": 1000, "sg": 1},
    {"data": "stem",  "arch": "cbow",     "window": 4,  "dim": 300,  "sg": 0},
    {"data": "stem",  "arch": "cbow",     "window": 4,  "dim": 1000, "sg": 0},
    {"data": "stem",  "arch": "cbow",     "window": 10, "dim": 300,  "sg": 0},
    {"data": "stem",  "arch": "cbow",     "window": 10, "dim": 1000, "sg": 0},
    {"data": "stem",  "arch": "skipgram", "window": 4,  "dim": 300,  "sg": 1},
    {"data": "stem",  "arch": "skipgram", "window": 4,  "dim": 1000, "sg": 1},
    {"data": "stem",  "arch": "skipgram", "window": 10, "dim": 300,  "sg": 1},
    {"data": "stem",  "arch": "skipgram", "window": 10, "dim": 1000, "sg": 1},
]

for p in params:
    sentences = lemma_sentences if p['data'] == 'lemma' else stem_sentences
    model = Word2Vec(
        sentences,
        vector_size=p['dim'],
        window=p['window'],
        sg=p['sg'],
        min_count=1,
        workers=4,
        epochs=30  # dilersen arttırabilirsin
    )
    model_name = f"models/word2vec_{p['data']}_{p['arch']}_win{p['window']}_dim{p['dim']}.model"
    model.save(model_name)
    print(f"Saved: {model_name}")
