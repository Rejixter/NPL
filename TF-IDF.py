import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from scipy import sparse
import numpy as np

# Dosya adları
lemma_csv = 'books_lemmatized.csv'
stem_csv  = 'books_stemmed.csv'

# Dosyaları oku
lemma_df = pd.read_csv(lemma_csv)
stem_df  = pd.read_csv(stem_csv)

# Vectorizer ve matrisler
vectorizer_lemma = TfidfVectorizer()
vectorizer_stem  = TfidfVectorizer()

tfidf_matrix_lemma = vectorizer_lemma.fit_transform(lemma_df['lemmatized'])
tfidf_matrix_stem  = vectorizer_stem.fit_transform(stem_df['stemmed'])

# Kitap isimlerini ayrı kaydet (npz'nin yanında .txt olarak saklamak yaygındır, istersen pickle da olur)
np.savez('tfidf_full_lemma.npz', data=tfidf_matrix_lemma.data, indices=tfidf_matrix_lemma.indices,
         indptr=tfidf_matrix_lemma.indptr, shape=tfidf_matrix_lemma.shape)
np.savez('tfidf_full_stem.npz', data=tfidf_matrix_stem.data, indices=tfidf_matrix_stem.indices,
         indptr=tfidf_matrix_stem.indptr, shape=tfidf_matrix_stem.shape)

# Kitap adlarını csv ile veya ayrı bir txt ile de saklayabilirsin. Ben pickle ile örnek bırakıyorum:
with open('book_names_lemma.pkl', 'wb') as f:
    pickle.dump(list(lemma_df['book_name']), f)
with open('book_names_stem.pkl', 'wb') as f:
    pickle.dump(list(stem_df['book_name']), f)

# Vectorizer nesnelerini kaydet
with open('tfidf_vectorizer_lemma.pkl', 'wb') as f:
    pickle.dump(vectorizer_lemma, f)
with open('tfidf_vectorizer_stem.pkl', 'wb') as f:
    pickle.dump(vectorizer_stem, f)

print('Bitti! tfidf_full_lemma.npz, tfidf_full_stem.npz, tfidf_vectorizer_lemma.pkl, tfidf_vectorizer_stem.pkl ve kitap adları pkl dosyaları oluşturuldu.')
