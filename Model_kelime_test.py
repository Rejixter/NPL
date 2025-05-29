import os
from gensim.models import Word2Vec

# Test edilecek kelime
target_word = "life"

# Modellerin bulunduğu klasör
models_dir = "models"

# .model ile biten tüm dosyaları tara
model_files = [f for f in os.listdir(models_dir) if f.endswith(".model")]

for model_file in sorted(model_files):
    model_path = os.path.join(models_dir, model_file)
    print(f"\nModel: {model_file}")
    model = Word2Vec.load(model_path)

    if target_word in model.wv:
        print(f"En yakın 5 kelime ('{target_word}'):")
        for word, sim in model.wv.most_similar(target_word, topn=5):
            print(f"  {word:<20}  Sim: {sim:.3f}")
    else:
        print(f"Kelime modelde yok: {target_word}")
