# similarity.py

import pickle
from sklearn.metrics.pairwise import cosine_similarity

# TF-IDF matrisini yükle
with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

# Cosine Similarity hesapla
similarity_matrix = cosine_similarity(tfidf_matrix)

# Benzerlik matrisini kontrol etmek için boyutunu yazdır
print("Benzerlik matrisinin boyutu:", similarity_matrix.shape)

# Benzerlik matrisini kaydet (ileride kullanılmak üzere)
with open('similarity_matrix.pkl', 'wb') as f:
    pickle.dump(similarity_matrix, f)

print("Benzerlik matrisi kaydedildi.")