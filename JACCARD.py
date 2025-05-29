import os
import pandas as pd
import numpy as np

def calculate_jaccard_matrix(output_dir, output_csv="jaccard_matrix.csv", top_n=5):
    # Klasördeki csv dosyalarını sırala
    csv_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".csv")])
    model_names = [f.replace(".csv", "") for f in csv_files]
    # İlk 5 öneri için kitap isimlerini oku
    top5_books = []
    for fname in csv_files:
        df = pd.read_csv(os.path.join(output_dir, fname))
        # Sütun adı "Book Name" olmalı
        names = set(df["Book Name"].iloc[1:top_n+1])  # 1:top_n+1 -> ilk satır giriş, sonra öneriler
        top5_books.append(names)
    # Jaccard matrisi oluştur
    n = len(top5_books)
    jac_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            inter = len(top5_books[i] & top5_books[j])
            union = len(top5_books[i] | top5_books[j])
            jac_mat[i, j] = inter / union if union > 0 else 0.0
    # DataFrame olarak kaydet
    jac_df = pd.DataFrame(jac_mat, columns=model_names, index=model_names)
    jac_df.to_csv(os.path.join(output_dir, output_csv))
    print(f"Jaccard matrisi kaydedildi: {os.path.join(output_dir, output_csv)}")
    # Word veya Docs'a kolayca eklemek için markdown tablo olarak da göster:
    print(jac_df.round(2).to_markdown())
    return jac_df

# KULLANIM:
# calculate_jaccard_matrix("outputs")
