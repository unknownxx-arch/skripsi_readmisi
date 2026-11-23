# -*- coding: utf-8 -*-
"""
04_buat_laporan.py
===================
Mengumpulkan semua hasil di folder output/ menjadi:
  1. output/laporan_hasil.html   -> dibuka otomatis di BROWSER (tanpa VS Code)
  2. output/hasil_penelitian.xlsx -> tabel hasil dalam Excel

Laporan hanya memuat ANGKA AGREGAT (jumlah data, persentase, hasil model),
tidak ada baris data pasien, sehingga aman ditampilkan saat sidang.

Letakkan file ini di folder src/ (sejajar 03_training_model.py).
Cara jalankan:   python src/04_buat_laporan.py
"""

import os
import sys
import base64
import html
import webbrowser
import pandas as pd
import config

OUT = config.FOLDER_OUTPUT


def wajib_ada(path, petunjuk):
    if not os.path.exists(path):
        print(f"[ERROR] File belum ada: {path}")
        print(petunjuk)
        sys.exit(1)


p_eval = os.path.join(OUT, "hasil_evaluasi_model.csv")
p_imp = os.path.join(OUT, "feature_importance.csv")
p_data = os.path.join(config.FOLDER_PROCESSED, "dataset_model.csv")
for p in (p_eval, p_imp, p_data):
    wajib_ada(p, "Jalankan dulu tahap 01-03 (atau: python run_all.py).")

df_eval = pd.read_csv(p_eval)
df_imp = pd.read_csv(p_imp)
data = pd.read_csv(p_data, usecols=["id_pasien", "readmisi_30_hari"])

# ---- ringkasan data (agregat saja) ----
n_episode = len(data)
n_pasien = data["id_pasien"].nunique()
n_pos = int(data["readmisi_30_hari"].sum())
pct_pos = n_pos / n_episode * 100

terbaik = df_eval.sort_values("AUC-ROC", ascending=False).iloc[0]
top3 = df_imp.head(3)["Fitur"].tolist()


def gambar(nama, lebar="100%"):
    path = os.path.join(OUT, nama)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" alt="{html.escape(nama)}" style="max-width:{lebar}">'


def tabel_html(df, sorot_model=None):
    baris = ["<table><thead><tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns) + "</tr></thead><tbody>"]
    for _, r in df.iterrows():
        kelas = ' class="best"' if sorot_model and str(r.iloc[0]) == sorot_model else ""
        baris.append(f"<tr{kelas}>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r.values) + "</tr>")
    baris.append("</tbody></table>")
    return "\n".join(baris)


df_imp_tampil = df_imp.head(10).copy()
for c in df_imp_tampil.columns[1:]:
    df_imp_tampil[c] = df_imp_tampil[c].round(4)

cms = [f for f in sorted(os.listdir(OUT)) if f.startswith("confusion_matrix_") and f.endswith(".png")]
cm_html = "".join(f'<div class="cm">{gambar(f)}</div>' for f in cms)

HTML = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<title>Hasil Prediksi Readmisi 30 Hari</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{ --bg:#ffffff; --fg:#1f2933; --muted:#5f6b7a; --card:#f4f7fb; --line:#d5dde8; --accent:#1F4E78; --best:#e3f1e3; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#14181d; --fg:#e6eaf0; --muted:#9aa6b5; --card:#1c222a; --line:#313a46; --accent:#7fb2e5; --best:#1f3a26; }}
  }}
  body {{ margin:0; background:var(--bg); color:var(--fg); font-family:"Segoe UI",Arial,sans-serif; line-height:1.55; }}
  main {{ max-width:1000px; margin:0 auto; padding:32px 20px 60px; }}
  h1 {{ color:var(--accent); font-size:1.7rem; margin-bottom:4px; }}
  h2 {{ color:var(--accent); margin-top:40px; border-bottom:2px solid var(--line); padding-bottom:6px; font-size:1.25rem; }}
  .sub {{ color:var(--muted); margin-top:0; }}
  .kartu {{ display:flex; gap:14px; flex-wrap:wrap; margin:18px 0; }}
  .kartu div {{ flex:1 1 180px; background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
  .kartu b {{ display:block; font-size:1.6rem; color:var(--accent); }}
  .kartu span {{ color:var(--muted); font-size:.9rem; }}
  table {{ border-collapse:collapse; width:100%; margin:12px 0; font-size:.93rem; }}
  th, td {{ border:1px solid var(--line); padding:8px 10px; text-align:left; }}
  th {{ background:var(--accent); color:#fff; }}
  tr.best td {{ background:var(--best); font-weight:600; }}
  .scroll {{ overflow-x:auto; }}
  .cmwrap {{ display:flex; gap:12px; flex-wrap:wrap; }}
  .cm {{ flex:1 1 280px; }} .cm img {{ width:100%; }}
  .catatan {{ background:var(--card); border-left:4px solid var(--accent); padding:12px 16px; border-radius:6px; margin:14px 0; }}
  img {{ height:auto; }}
  @media print {{ body {{ background:#fff; color:#000; }} }}
</style>
</head>
<body>
<main>
  <h1>Prediksi Risiko Readmisi Pasien dalam 30 Hari</h1>
  <p class="sub">Ringkasan hasil pemodelan machine learning. Hanya memuat data agregat.</p>

  <h2>1. Ringkasan Data</h2>
  <div class="kartu">
    <div><b>{n_episode:,}</b><span>episode rawat inap dianalisis</span></div>
    <div><b>{n_pasien:,}</b><span>pasien unik</span></div>
    <div><b>{n_pos:,}</b><span>kasus readmisi 30 hari</span></div>
    <div><b>{pct_pos:.1f}%</b><span>proporsi readmisi</span></div>
  </div>

  <h2>2. Perbandingan Model</h2>
  <div class="scroll">{tabel_html(df_eval, terbaik["Model"])}</div>
  <div class="catatan">
    Model dengan AUC-ROC tertinggi pada data uji: <b>{html.escape(str(terbaik["Model"]))}</b>
    (AUC-ROC = {terbaik["AUC-ROC"]}, Recall = {terbaik["Recall"]}, F1-Score = {terbaik["F1-Score"]}).
  </div>

  <h2>3. Kurva ROC</h2>
  {gambar("roc_curve_comparison.png", "620px")}

  <h2>4. Confusion Matrix</h2>
  <div class="cmwrap">{cm_html}</div>

  <h2>5. Faktor Paling Berpengaruh</h2>
  <div class="scroll">{tabel_html(df_imp_tampil)}</div>
  {gambar("feature_importance.png", "720px")}
  <div class="catatan">
    Tiga faktor paling berpengaruh pada model terbaik:
    <b>{html.escape(", ".join(top3))}</b>. Nilai pada tabel menunjukkan seberapa besar
    AUC turun ketika faktor tersebut diacak (semakin besar, semakin penting).
  </div>
</main>
</body>
</html>
"""

os.makedirs(OUT, exist_ok=True)
path_html = os.path.join(OUT, "laporan_hasil.html")
with open(path_html, "w", encoding="utf-8") as f:
    f.write(HTML)

# ---- Excel ----
path_xlsx = os.path.join(OUT, "hasil_penelitian.xlsx")
ringkasan = pd.DataFrame({
    "Keterangan": ["Episode rawat inap", "Pasien unik", "Kasus readmisi 30 hari", "Proporsi readmisi (%)", "Model terbaik"],
    "Nilai": [n_episode, n_pasien, n_pos, round(pct_pos, 1), terbaik["Model"]],
})
with pd.ExcelWriter(path_xlsx, engine="openpyxl") as w:
    ringkasan.to_excel(w, sheet_name="Ringkasan", index=False)
    df_eval.to_excel(w, sheet_name="Perbandingan Model", index=False)
    df_imp.to_excel(w, sheet_name="Faktor Berpengaruh", index=False)
    for ws in w.book.worksheets:
        for kolom in ws.columns:
            lebar = max(len(str(c.value)) if c.value is not None else 0 for c in kolom)
            ws.column_dimensions[kolom[0].column_letter].width = min(max(lebar + 3, 12), 45)

print("Laporan selesai dibuat:")
print(f"  Browser : {path_html}")
print(f"  Excel   : {path_xlsx}")

# buka otomatis di browser (lewati dengan: python src/04_buat_laporan.py --tanpa-buka)
if "--tanpa-buka" not in sys.argv:
    webbrowser.open("file://" + os.path.abspath(path_html).replace("\\", "/"))
    print("\nLaporan dibuka di browser. Jika tidak terbuka, klik dua kali file laporan_hasil.html di folder output.")
