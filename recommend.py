# recommend.py

import pandas as pd
import pickle
import sys

# 1) Temizlenmiş veri setini yükle
df = pd.read_csv('clean_dataset_final.csv')  # veya clean_dataset.csv

# 2) similarity matrisini yükle
with open('similarity_matrix.pkl', 'rb') as f:
    similarity_matrix = pickle.load(f)

# 3) Öneri fonksiyonu
def recommend(book_title, sim_matrix, data_frame, top_n=5):
    # Kitabın indeksini bul
    try:
        idx = data_frame[data_frame['book_name'] == book_title].index[0]
    except IndexError:
        print(f"⚠️ '{book_title}' başlıklı bir kitap bulunamadı.")
        return []

    # Tüm kitaplarla benzerlik skorlarını al
    sim_scores = list(enumerate(sim_matrix[idx]))
    # Kendisini ele ve benzerliğe göre sırala
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Tekil kitap isimleri ile öneri listesi oluştur
    recommendations = []
    for i, score in sim_scores:
        title = data_frame.iloc[i]['book_name']
        if title == book_title:
            continue               # kendisi ise atla
        if title in recommendations:
            continue               # daha önce eklenmişse atla
        recommendations.append(title)
        if len(recommendations) >= top_n:
            break

    return recommendations

# 4) Komut satırından kitap adı girilebilsin
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python recommend.py \"Kitap Adı\" [öneri_sayısı]")
        sys.exit(1)

    book = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    recs = recommend(book, similarity_matrix, df, top_n=n)
    if recs:
        print(f"\n📚 '{book}' için önerilen {len(recs)} kitap:")
        for i, title in enumerate(recs, 1):
            print(f"{i}. {title}")
