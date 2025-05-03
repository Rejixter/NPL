import pandas as pd

# === Doğru dosya adları ===
tokenized_text = "tokenized_sentences.txt"
lemmatized_text = "lemmatized_books_summary.txt"
stemmed_text = "stemmed_books_summary.txt"
output_csv = "processed_sentences.csv"

# === Her dosyadan cümleleri oku (cümle yapısını koruyarak) ===
with open(tokenized_text, "r", encoding="utf-8") as f:
    base_sentences = [line.strip() for line in f if line.strip()]

with open(lemmatized_text, "r", encoding="utf-8") as f:
    lemmatized_sentences = [line.strip() for line in f if line.strip()]

with open(stemmed_text, "r", encoding="utf-8") as f:
    stemmed_sentences = [line.strip() for line in f if line.strip()]

# === Eşit uzunlukta mı kontrol et ===
min_len = min(len(base_sentences), len(lemmatized_sentences), len(stemmed_sentences))
base_sentences = base_sentences[:min_len]
lemmatized_sentences = lemmatized_sentences[:min_len]
stemmed_sentences = stemmed_sentences[:min_len]

# === DataFrame oluştur ve CSV'ye yaz ===
df = pd.DataFrame({
    "Base Sentence": base_sentences,
    "Lemmatized": lemmatized_sentences,
    "Stemmed": stemmed_sentences
})

df.to_csv(output_csv, index=False, encoding="utf-8")
print(f"✅ Kaydedildi: {output_csv} ({len(df)} satır)")
