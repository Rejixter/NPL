# tfidf_lemmatized.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# === Dosya yolu ayarları ===
current_dir = os.path.dirname(__file__)
input_path = os.path.abspath(os.path.join(current_dir, "..", "processed_sentences.csv"))
output_path = os.path.join(current_dir, "tfidf_lemmatized.csv")

# === Veriyi oku ===
df = pd.read_csv(input_path)
corpus = df["Lemmatized"].dropna().astype(str)

# === TF-IDF işlemi ===
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# === Sonuçları DataFrame olarak kaydet ===
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
df_tfidf.to_csv(output_path, index=False)

print(f"✅ Lemmatized TF-IDF matris CSV olarak kaydedildi: {output_path}")
