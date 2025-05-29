import os
import pandas as pd

# === Ayarlar ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
input_dir = os.path.join(BASE_DIR, "outputs")  # outputs klasöründe arama yapacak
output_txt = os.path.join(BASE_DIR, "manual_scoring_full_input.txt")  # analiz klasörü yoksa base'e kaydeder

# === Tüm CSV dosyalarını bul
csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

lines = []

for fname in sorted(csv_files):
    model_name = fname.replace(".csv", "")
    fpath = os.path.join(input_dir, fname)
    
    try:
        df = pd.read_csv(fpath)
        if len(df) < 2:
            print(f"⚠️ Yetersiz veri: {fname}")
            continue

        lines.append(f"Model: {model_name}")
        for i, row in df.iloc[1:11].iterrows():  # İlk satır giriş cümlesi, sonra 10 öneri
            lines.append(f"{i}.")
            lines.append(f"  Match Index     : {row.get('Match Index', '')}")
            lines.append(f"  Similarity Score: {row.get('Similarity Score', ''):.4f}")
            lines.append(f"  Book Name       : {row.get('Book Name', '')}")
            lines.append(f"  Sentence        : {row.get('Sentence', '')}")
            lines.append("")  # boşluk

        lines.append("-" * 50)  # model ayırıcı
        lines.append("")

    except Exception as e:
        print(f"❌ Hata oluştu ({fname}): {e}")

# === Dosyaya yaz
with open(output_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"✅ Anlamsal değerlendirme için .txt dosyası oluşturuldu → {output_txt}")
