import os
import pandas as pd
from gensim.models import Word2Vec

column = "Lemmatized"
window = 2
vector_size = 50
sg = 0  # CBOW

input_path = os.path.abspath("processed_lemmatized_stemmed.csv")
df = pd.read_csv(input_path)
sentences = df[column].dropna().apply(lambda x: x.split()).tolist()

model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, min_count=2, workers=4, sg=sg)

model_path = f"models/word2vec_lemmatized_cbow_w4_d300.model"
model.save(model_path)

vec_path = f"models/vectors_lemmatized_cbow_w4_d300.csv"
pd.DataFrame([model.wv[word] for word in model.wv.index_to_key], index=model.wv.index_to_key).to_csv(vec_path)

print("✅ model_1 tamamlandı.")
