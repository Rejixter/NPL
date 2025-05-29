import numpy as np
import pickle
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os

# Parametreler (stem dosyaları)
npz_file = 'tfidf_full_stem.npz'
vectorizer_file = 'tfidf_vectorizer_stem.pkl'
book_names_file = 'book_names_stem.pkl'
sentences_file = 'books_stemmed.csv'

query_idx = 0  # sorgu satırı

output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)
output_csv = os.path.join(output_dir, 'tfidf_top10_stem_unique_books.csv')

npz = np.load(npz_file)
tfidf_matrix = sparse.csr_matrix((npz['data'], npz['indices'], npz['indptr']), shape=npz['shape'])

with open(vectorizer_file, 'rb') as f:
    vectorizer = pickle.load(f)
with open(book_names_file, 'rb') as f:
    book_names = pickle.load(f)

df = pd.read_csv(sentences_file)

query_vec = tfidf_matrix[query_idx]
sims = cosine_similarity(query_vec, tfidf_matrix)[0]

sorted_idx = np.argsort(sims)[::-1]

rows = []
seen_books = set()

for match_idx in sorted_idx:
    book = book_names[match_idx]
    if book not in seen_books:
        rows.append({
            'Match Index': match_idx,
            'Query Index': query_idx,
            'Book Name': book,
            'Sentence': df.iloc[match_idx, 1],
            'Similarity Score': sims[match_idx]
        })
        seen_books.add(book)
    if len(rows) == 10:
        break

out_df = pd.DataFrame(rows)
out_df.to_csv(output_csv, index=False)
print(f"{output_csv} dosyası oluşturuldu.")
