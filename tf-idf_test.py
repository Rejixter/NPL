import numpy as np
import pickle
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

def load_tfidf(npz_file, vectorizer_file, book_names_file):
    npz = np.load(npz_file)
    tfidf_matrix = sparse.csr_matrix((npz['data'], npz['indices'], npz['indptr']), shape=npz['shape'])
    with open(vectorizer_file, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(book_names_file, 'rb') as f:
        book_names = pickle.load(f)
    return tfidf_matrix, vectorizer, book_names

def print_top5(model_name, tfidf_matrix, vectorizer, book_names, test_sentence):
    test_vec = vectorizer.transform([test_sentence])
    sims = cosine_similarity(test_vec, tfidf_matrix)[0]
    top5_idx = np.argsort(sims)[::-1][:5]
    print(f"\n{model_name} - En benzer 5 kitap:")
    for rank, idx in enumerate(top5_idx, 1):
        print(f"{rank}. Satır: {idx} | {book_names[idx]:<40} (Benzerlik: {sims[idx]:.3f})")

# Dosya adları
files = {
    "TF-IDF Lemma": {
        "npz": "tfidf_full_lemma.npz",
        "vectorizer": "tfidf_vectorizer_lemma.pkl",
        "book_names": "book_names_lemma.pkl"
    },
    "TF-IDF Stem": {
        "npz": "tfidf_full_stem.npz",
        "vectorizer": "tfidf_vectorizer_stem.pkl",
        "book_names": "book_names_stem.pkl"
    }
}

# Test cümlesi
test_sentence = "meaning of life and happiness"

# Her iki model için çalıştır
for model_name, f in files.items():
    tfidf_matrix, vectorizer, book_names = load_tfidf(f["npz"], f["vectorizer"], f["book_names"])
    print_top5(model_name, tfidf_matrix, vectorizer, book_names, test_sentence)
