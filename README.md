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
1. sırasıyla şu kodları çalıştırın.
2. tokenize_kitap.py, lemma_stem.py, TF-IDF.py ve model_yap.py 
3. bu kodları visual studio'da python extension'u ile birlikte 1 tıklama ile çalıştırabilirsiniz.
4. test kodlarını çalıştırıp anlık çıktı görmek isterseniz. "Books_summary.cvs" dışında diğer tüm .csv, .pkl, .txt ve .model dosyalarını silebilirsiniz. geriye kalan kodlar tablo oluşturmaya yarıyor.

📊 Zipf Yasası
Ham Veri, Lemmatized ve Stemmed için log-log Zipf grafikleri çıkarılmıştır.

Zipf yasasına büyük ölçüde uyumlu olduğu gözlemlenmiştir.

