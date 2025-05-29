import pandas as pd
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# Gerekirse:
# import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

df = pd.read_csv('books_tokenized.csv')
df.dropna(inplace=True)

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def to_lemmatized(text):
    tokens = word_tokenize(str(text))
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])

def to_stemmed(text):
    tokens = word_tokenize(str(text))
    return ' '.join([stemmer.stem(token) for token in tokens])

df['lemmatized'] = df['tokenized'].apply(to_lemmatized)
df['stemmed'] = df['tokenized'].apply(to_stemmed)

df[['book_name', 'lemmatized']].to_csv('books_lemmatized.csv', index=False)
df[['book_name', 'stemmed']].to_csv('books_stemmed.csv', index=False)
