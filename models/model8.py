# model_8.py
import os
from gensim.models import Word2Vec
import pandas as pd

column = "Lemmatized"
window = 10
vector_size = 1000
sg = 1

current_dir = os.path.dirname(__file__)
input_path = os.path.abspath(os.path.join(current_dir, "..", "processed_sentences.csv"))

df = pd.read_csv(input_path)
sentences = df[column].dropna().apply(lambda x: x.strip().split()).tolist()

model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, min_count=2, workers=4, sg=sg)

model_name = f"word2vec_lemma_skipgram_w{window}_d{vector_size}.model"
model_path = os.path.join(current_dir, model_name)
model.save(model_path)

vector_out_path = os.path.join(current_dir, f"vectors_lemma_skipgram_w{window}_d{vector_size}.csv")
vectors = [model.wv[word] for word in model.wv.index_to_key]
words = model.wv.index_to_key
pd.DataFrame(vectors, index=words).to_csv(vector_out_path)

print(f"✅ Model ve CSV kaydedildi: {model_name}, {vector_out_path}")
