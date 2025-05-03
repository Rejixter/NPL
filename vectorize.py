# vectorize.py (hata düzeltmeli versiyon)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Temiz veriyi yükle
df = pd.read_csv('clean_dataset.csv')

# 'clean_summary' sütunundaki kalan boş değerleri kaldır
df = df.dropna(subset=['clean_summary'])

# Ekstra kontrol (bazı satırlar boş kalmış olabilir)
df = df[df['clean_summary'].str.strip().astype(bool)]

# TF-IDF vektörleştirme
vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df['clean_summary'])

# Boyutları yazdır
print("TF-IDF matris boyutu:", tfidf_matrix.shape)

# TF-IDF matrisini ve vektörleyiciyi kaydet
with open('tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("TF-IDF matrisi ve vektörleyici kaydedildi.")

# Son halini tekrar kaydet
df.to_csv('clean_dataset_final.csv', index=False)