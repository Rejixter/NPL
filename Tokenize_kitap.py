import pandas as pd
import re
from nltk.tokenize import word_tokenize

# Gerekirse:
# import nltk
# nltk.download('punkt')

# CSV'yi oku
df = pd.read_csv('books_summary.csv')
df.dropna(inplace=True)

# Temizleme + boşluklu tokenize fonksiyonu
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    return ' '.join(tokens)

# İçerik: kitap adı hariç diğer sütunlar birleştiriliyor
df['content'] = df.drop(['book_name'], axis=1).astype(str).agg(' '.join, axis=1)
df['tokenized'] = df['content'].apply(clean_text)

# Çıktı: kitap adı ve tokenized metin
df_tokenized = df[['book_name', 'tokenized']]
df_tokenized.to_csv('books_tokenized.csv', index=False)
