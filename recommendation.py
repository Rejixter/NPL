import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === CSV dosyasını oku ===
df = pd.read_csv("books_summary.csv")
df.drop_duplicates(subset=["book_name"], inplace=True)

# === Boş özetleri çıkar ===
df.dropna(subset=["summaries"], inplace=True)
df.reset_index(drop=True, inplace=True)

# === TF-IDF Vektörleştirme ===
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df["summaries"])

# === Kitap ismine göre en benzer kitapları bul ===
def recommend_books(book_title, top_n=5):
    if book_title not in df["book_name"].values:
        print("Kitap bulunamadı.")
        return

    # Kitabın index'ini al
    idx = df[df["book_name"] == book_title].index[0]

    # Cosine benzerlik skorlarını hesapla
    cosine_similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # En benzer kitapları sırala (kendisi hariç)
    similar_indices = cosine_similarities.argsort()[::-1][1:top_n+1]

    print(f"\n📚 '{book_title}' için önerilen kitaplar:")
    for i in similar_indices:
        print(f"→ {df.iloc[i]['book_name']}")
        # Örnek çağrı
recommend_books("1984")  # Buraya senin csv'de bulunan bir kitap adı yazmalısın