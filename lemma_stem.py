import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer

# === GEREKLİ NLTK VERİLERİNİ İNDİR ===
required_packages = ['wordnet', 'omw-1.4']
for package in required_packages:
    nltk.download(package)

# === DOSYA AYARLARI ===
input_file = "tokenized_books_summary.txt"    # Önceki çıkan dosya
lemmatized_output = "lemmatized_books_summary.txt"
stemmed_output = "stemmed_books_summary.txt"

# === Araçlar ===
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# === TOKENLERİ OKU ===
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

tokens = text.split()  # Boşluklardan ayırıp liste yapıyoruz

# === LEMMATIZATION İŞLEMİ ===
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

# === STEMMING İŞLEMİ ===
stemmed_tokens = [stemmer.stem(token) for token in tokens]

# === SONUÇLARI DOSYALARA KAYDET ===
with open(lemmatized_output, "w", encoding="utf-8") as f:
    f.write(" ".join(lemmatized_tokens))

with open(stemmed_output, "w", encoding="utf-8") as f:
    f.write(" ".join(stemmed_tokens))

print(f"\n✅ Lemmatization ve Stemming işlemleri tamamlandı!")
print(f"📄 Lemmatized dosya: {lemmatized_output}")
print(f"📄 Stemmed dosya: {stemmed_output}")
