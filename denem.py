import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# Gerekli NLTK bileşenlerini indir
nltk.download('punkt')
nltk.download('wordnet')

# === Giriş ve çıkış dosyaları ===
input_file = "tokenized_sentences.txt"
output_csv = "processed_lemmatized_stemmed.csv"

# === Araçlar ===
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# === Dosyayı oku ve işle ===
with open(input_file, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

lemmatized_sentences = []
stemmed_sentences = []

for sentence in sentences:
    tokens = word_tokenize(sentence)
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    stemmed = [stemmer.stem(word) for word in tokens]
    
    lemmatized_sentences.append(" ".join(lemmatized))
    stemmed_sentences.append(" ".join(stemmed))

# === CSV olarak kaydet ===
df = pd.DataFrame({
    "Lemmatized": lemmatized_sentences,
    "Stemmed": stemmed_sentences
})

df.to_csv(output_csv, index=False, encoding="utf-8")
print(f"✅ CSV dosyası kaydedildi: {output_csv} ({len(df)} satır)")
