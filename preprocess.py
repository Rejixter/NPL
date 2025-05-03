# preprocess.py (hata düzeltmeli versiyon)

import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Gerekli NLTK paketlerini indir
nltk.download('stopwords')
nltk.download('wordnet')

# Veri setini oku
df = pd.read_csv('books_summary.csv')

# 'summaries' sütunundaki boş değerleri kaldır
df = df.dropna(subset=['summaries'])

# Veri temizleme fonksiyonu
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()  # Küçük harfe dönüştür
    text = re.sub(r'[^a-z\s]', '', text)  # Noktalama işaretlerini ve sayıları kaldır
    words = text.split()  # Kelimelere ayır
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatize et ve stopwords kaldır
    return ' '.join(words)

# DataFrame'e yeni temizlenmiş sütunu ekle
df['clean_summary'] = df['summaries'].apply(preprocess)

# Sonucu kontrol et
print(df[['summaries', 'clean_summary']].head())

# Temiz veriyi kaydet (ileride kullanılabilir)
df.to_csv('clean_dataset.csv', index=False)
