import os
import pandas as pd
import numpy as np

def calculate_jaccard_matrix(output_dir, output_csv="jaccard_matrix.csv", top_n=5):
    csv_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".csv")])
    model_names = []
    top5_books = []
    for fname in csv_files:
        fpath = os.path.join(output_dir, fname)
        try:
            df = pd.read_csv(fpath)
            if df.empty or "Book Name" not in df.columns:
                print(f"⚠️ Atlaniyor (bos ya da eksik): {fname}")
                continue
            names = set(df["Book Name"].iloc[1:top_n+1])
            top5_books.append(names)
            model_names.append(fname.replace(".csv", ""))
        except Exception as e:
            print(f"❌ Hata ({fname}): {e}")
            continue
    n = len(top5_books)
    jac_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            inter = len(top5_books[i] & top5_books[j])
            union = len(top5_books[i] | top5_books[j])
            jac_mat[i, j] = inter / union if union > 0 else 0.0
    jac_df = pd.DataFrame(jac_mat, columns=model_names, index=model_names)
    jac_df.to_csv(os.path.join(output_dir, output_csv))
    print(f"Jaccard matrisi kaydedildi: {os.path.join(output_dir, output_csv)}")
    print(jac_df.round(2).to_markdown())
    return jac_df

if __name__ == "__main__":
    calculate_jaccard_matrix("outputs")
