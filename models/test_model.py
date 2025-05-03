from gensim.models import Word2Vec

# === Ayarlar ===
model_path = "models/word2vec_lemma_cbow_w4_d1000.model"  # test etmek istediğin modelin yolu
kelime = "philosophy"  # test etmek istediğin kelime

# === Modeli yükle ===
model = Word2Vec.load(model_path)

# === En benzer 5 kelimeyi bul ===
if kelime in model.wv:
    benzerler = model.wv.most_similar(kelime, topn=100)
    print(f"🔍 '{kelime}' kelimesine en benzer 5 kelime:")
    for kelime, skor in benzerler:
        print(f"  {kelime:<20} -> {skor:.4f}")
else:
    print(f"❌ '{kelime}' kelimesi modelde bulunamadı.")
