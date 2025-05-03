import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize  # Bunu eklemeyi unutma

# Gerekli nltk veri setlerini indir
nltk.download('punkt')
nltk.download('stopwords')

# === Dosya ayarları ===
input_csv = r"C:\Users\kbbkk\OneDrive\Masaüstü\KitapProjesi\books_summary.csv"
output_txt = r"C:\Users\kbbkk\OneDrive\Masaüstü\KitapProjesi\tokenized_sentences.txt"
text_column = "summaries"  # CSV'deki özet kolonunun adı

# === Stopword listesi (isteğe bağlı kullanılır) ===
stop_words = set(stopwords.words('english'))

# === Temizleme ve cümle tokenize + kelime tokenize ===
def process_text(text):
    if pd.isna(text):
        return []

    sentences = text.split('.')  # Basit cümle ayrımı
    tokenized_sentences = []

    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub(r"[^a-z\s]", "", sentence)  # Noktalama temizliği (isteğe bağlı)
        tokens = wordpunct_tokenize(sentence)  # Değişiklik burada
        tokens = [word for word in tokens if word not in stop_words]
        if tokens:
            tokenized_sentences.append(" ".join(tokens))

    return tokenized_sentences

# === CSV'yi oku ===
df = pd.read_csv(input_csv)

# === Tokenize edilmiş tüm cümleleri topla ===
all_sentences = []

for text in df[text_column]:
    tokenized_sents = process_text(text)
    all_sentences.extend(tokenized_sents)

# === Dosyaya yaz ===
with open(output_txt, "w", encoding="utf-8") as f:
    for sentence in all_sentences:
        f.write(sentence + "\n")

print(f"\n✅ {len(all_sentences)} cümle işlendi ve '{output_txt}' dosyasına yazıldı.")
