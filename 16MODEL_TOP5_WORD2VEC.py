import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Model parametreleri
model_params = [
    # (veri_tipi, mimari, window, dim)
    ("lemma", "cbow", 4, 300),
    ("lemma", "cbow", 4, 1000),
    ("lemma", "cbow", 10, 300),
    ("lemma", "cbow", 10, 1000),
    ("lemma", "skipgram", 4, 300),
    ("lemma", "skipgram", 4, 1000),
    ("lemma", "skipgram", 10, 300),
    ("lemma", "skipgram", 10, 1000),
    ("stem", "cbow", 4, 300),
    ("stem", "cbow", 4, 1000),
    ("stem", "cbow", 10, 300),
    ("stem", "cbow", 10, 1000),
    ("stem", "skipgram", 4, 300),
    ("stem", "skipgram", 4, 1000),
    ("stem", "skipgram", 10, 300),
    ("stem", "skipgram", 10, 1000),
]

query_idx = 0  # Sorgulanan cümlenin indexi
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# CSV dosyalarını yükle
lemma_df = pd.read_csv('books_lemmatized.csv')
stem_df = pd.read_csv('books_stemmed.csv')

for veri_tipi, mimari, window, dim in model_params:
    # Model ve veri seçimi
    model_path = f"models/word2vec_{veri_tipi}_{mimari}_win{window}_dim{dim}.model"
    if not os.path.exists(model_path):
        print("Model bulunamadı:", model_path)
        continue
    model = Word2Vec.load(model_path)

    if veri_tipi == "lemma":
        df = lemma_df
        sentences_col = "lemmatized"
    else:
        df = stem_df
        sentences_col = "stemmed"

    book_names = df['book_name'].tolist()
    sentences = df[sentences_col].tolist()

    # Sorgu cümlesini al ve vektörleştir
    query_sentence = sentences[query_idx]
    def get_mean_vector(model, tokens):
        vectors = [model.wv[w] for w in tokens if w in model.wv]
        if len(vectors) == 0:
            return np.zeros(model.vector_size)
        return np.mean(vectors, axis=0)
    query_tokens = query_sentence.split()
    query_vec = get_mean_vector(model, query_tokens).reshape(1, -1)

    # Tüm cümlelerin vektörleri
    all_vecs = []
    for s in sentences:
        tokens = s.split()
        all_vecs.append(get_mean_vector(model, tokens))
    all_vecs = np.vstack(all_vecs)

    # Cosine similarity
    sims = cosine_similarity(query_vec, all_vecs)[0]
    sorted_idx = np.argsort(sims)[::-1]

    # 10 farklı kitap için çıktı
    rows = []
    seen_books = set()
    for match_idx in sorted_idx:
        book = book_names[match_idx]
        if book not in seen_books:
            rows.append({
                'Match Index': match_idx,
                'Query Index': query_idx,
                'Book Name': book,
                'Sentence': sentences[match_idx],
                'Similarity Score': sims[match_idx]
            })
            seen_books.add(book)
        if len(rows) == 10:
            break

    # Çıktı dosya ismi
    out_name = f"w2v_{veri_tipi}_{mimari}_win{window}_dim{dim}_top10.csv"
    out_path = os.path.join(output_dir, out_name)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"{out_path} kaydedildi.")

print("Tüm modeller için çıktı dosyaları oluşturuldu.")
