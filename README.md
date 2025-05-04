 Öğrenci Bilgileri
👨‍🎓 Ad Soyad: Kağamberk Eroğlu
🧪 Ders: Doğal Dil İşleme# NPL
 
🎯 Veri Seti Kullanım Amacı
Bu veri seti, kitaplara ait özet metinleri içermektedir 

Kelime Temsilleri Oluşturma: Word2Vec ve TF-IDF gibi yöntemlerle kelimelerin vektörel temsillerini çıkararak dil modelleri eğitmek.

Benzerlik Analizi: Belirli kelime ya da kavramlara en yakın anlamda benzer kelimeleri tespit etmek.

Öneri Sistemleri: Kitap özetlerinden yola çıkarak içerik tabanlı kitap öneri sistemleri geliştirmek.

Dil Dağılımı ve İstatistiksel Analiz: Zipf yasası gibi dağılım yasalarını test etmek, dil yapısını incelemek.

Ön İşleme Tekniklerini Karşılaştırma: Lemmatization ve stemming gibi işlemlerin etkilerini karşılaştırmak.


Kurulum ve Kütüphaneler

Aşağıdaki kütüphanelerin yüklü olduğundan emin olun:

```bash
pip install pandas nltk gensim matplotlib scikit-learn

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

⚙️ Nasıl Çalıştırılır?

tokenize_text.py: Ham metni temizleyerek tokenized_sentences.txt üretir.

lemma_stem.py: Yukarıdaki dosyadan lemmatize ve stem edilmiş dosyaları (lemmatized_books_summary.txt, stemmed_books_summary.txt) üretir.

csvdonusturucu.py: Lemma ve stem dosyalarını processed_sentences.csv formatına çevirir (Base + Lemma + Stem sütunlu).

models/model_1.py - model_16.py: 16 farklı Word2Vec modeli üretir (varyasyonlar: CBOW/SkipGram, window=4/10, dim=300/1000, lemma/stem).

Her model .model ve .csv dosyaları oluşturur.

tfidf_lemmatized.py, tfidf_stemmed.py: Her biri için TF-IDF matrisini oluşturur (tfidf_lemmatized.csv, tfidf_stemmed.csv).

test_model.py: Örnek kelimelerle eğitimli modelleri test eder.

📊 Zipf Yasası
Ham Veri, Lemmatized ve Stemmed için log-log Zipf grafikleri çıkarılmıştır.

Zipf yasasına büyük ölçüde uyumlu olduğu gözlemlenmiştir.

📦KitapProjesi
 ┣ 📂models                # Word2Vec modelleri (model_1.py ... model_16.py)
 ┣ 📂test_outputs          # Zipf grafikleri, test sonuçları
 ┣ processed_sentences.csv
 ┣ tfidf_lemmatized.csv
 ┣ tfidf_stemmed.csv
 ┣ tokenized_sentences.txt
 ┣ lemmatized_books_summary.txt
 ┣ stemmed_books_summary.txt
 ┗ README.md