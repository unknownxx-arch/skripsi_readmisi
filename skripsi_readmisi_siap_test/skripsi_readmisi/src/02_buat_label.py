# -*- coding: utf-8 -*-
"""
02_buat_label.py
=================
Mengubah data mentah RS (1 baris = 1 episode rawat inap) menjadi dataset
siap-model, dengan langkah:

  1. Membersihkan data (tanggal rusak, discharge < admisi)
  2. Membuat fitur turunan:
       - los_hari
       - jumlah_admisi_365hari_sebelumnya
       - episode_ini_adalah_readmisi
  3. Membuat LABEL  readmisi_30_hari :
       1 = pasien yang SAMA masuk rawat inap lagi <= 30 hari setelah discharge
       0 = tidak
     (admisi berikutnya yang TERENCANA tidak dihitung sebagai readmisi)
  4. Mengeluarkan episode yang tidak layak dianalisis:
       - pasien meninggal saat rawat inap (tidak mungkin readmisi)
       - episode yang discharge-nya terlalu dekat dengan akhir periode data
         (masa pengamatan 30 harinya belum lengkap -> label tidak pasti)

Hasil: data/processed/dataset_model.csv

Cara jalankan:   python src/02_buat_label.py
"""

import os
import numpy as np
import pandas as pd
import config
from utils import baca_data_mentah, parse_tanggal

print("=" * 70)
print("PEMBUATAN LABEL READMISI 30 HARI")
print("=" * 70)

df = baca_data_mentah()
n_awal = len(df)

# ------------------------------------------------------------------
# 1. Pembersihan
# ------------------------------------------------------------------
df["tanggal_admisi"] = parse_tanggal(df["tanggal_admisi"])
df["tanggal_discharge"] = parse_tanggal(df["tanggal_discharge"])

df = df.dropna(subset=["id_pasien", "tanggal_admisi", "tanggal_discharge"])
n_tgl = n_awal - len(df)
df = df[df["tanggal_discharge"] >= df["tanggal_admisi"]]
n_terbalik = n_awal - n_tgl - len(df)
df = df.drop_duplicates()
n_dup = n_awal - n_tgl - n_terbalik - len(df)

print(f"\nBaris awal                       : {n_awal}")
print(f"Dibuang (tanggal kosong/rusak)   : {n_tgl}")
print(f"Dibuang (discharge < admisi)     : {n_terbalik}")
print(f"Dibuang (duplikat)               : {n_dup}")

df = df.sort_values(["id_pasien", "tanggal_admisi"]).reset_index(drop=True)

# ------------------------------------------------------------------
# 2. Fitur turunan
# ------------------------------------------------------------------
df["los_hari"] = (df["tanggal_discharge"] - df["tanggal_admisi"]).dt.days

g = df.groupby("id_pasien")
df["_next_admisi"] = g["tanggal_admisi"].shift(-1)
df["_prev_discharge"] = g["tanggal_discharge"].shift(1)
if "jalur_masuk" in df.columns:
    df["_next_jalur"] = g["jalur_masuk"].shift(-1)
else:
    df["_next_jalur"] = np.nan

# jumlah admisi dalam 365 hari sebelum admisi ini
jumlah_365 = np.zeros(len(df), dtype=int)
for _, idx in g.indices.items():
    tgl = df.loc[idx, "tanggal_admisi"].values
    for k, pos in enumerate(idx):
        batas_bawah = tgl[k] - np.timedelta64(365, "D")
        jumlah_365[pos] = int(np.sum((tgl[:k] >= batas_bawah)))
df["jumlah_admisi_365hari_sebelumnya"] = jumlah_365

# episode ini sendiri merupakan readmisi dari episode sebelumnya?
selisih_prev = (df["tanggal_admisi"] - df["_prev_discharge"]).dt.days
df["episode_ini_adalah_readmisi"] = (
    (selisih_prev >= 0) & (selisih_prev <= config.JENDELA_READMISI_HARI)
).astype(int)

# ------------------------------------------------------------------
# 3. LABEL readmisi
# ------------------------------------------------------------------
selisih_next = (df["_next_admisi"] - df["tanggal_discharge"]).dt.days
terencana = df["_next_jalur"].isin(config.JALUR_MASUK_TERENCANA)

df["readmisi_30_hari"] = (
    (selisih_next >= 0) & (selisih_next <= config.JENDELA_READMISI_HARI) & (~terencana)
).astype(int)

# ------------------------------------------------------------------
# 4. Pengecualian
# ------------------------------------------------------------------
n_sebelum = len(df)

# 4a. status pulang yang dikeluarkan (mis. Meninggal)
mask_status = df["status_pulang"].isin(config.STATUS_DIKELUARKAN)
n_status = int(mask_status.sum())
df = df[~mask_status]

# 4b. masa pengamatan belum lengkap (censoring)
tanggal_akhir_data = df["tanggal_admisi"].max()
batas = tanggal_akhir_data - pd.Timedelta(days=config.JENDELA_READMISI_HARI)
mask_censor = df["tanggal_discharge"] > batas
n_censor = int(mask_censor.sum())
df = df[~mask_censor]

print(f"\nDikeluarkan (status {config.STATUS_DIKELUARKAN}) : {n_status}")
print(f"Dikeluarkan (masa pengamatan 30 hari belum lengkap, discharge > {batas.date()}): {n_censor}")
print(f"Episode tersisa untuk model      : {len(df)}")

# ------------------------------------------------------------------
# 5. Simpan
# ------------------------------------------------------------------
fitur_ada = [c for c in (config.FITUR_NUMERIK + config.FITUR_KATEGORIKAL) if c in df.columns]
kolom_simpan = ["id_pasien", "tanggal_admisi", "tanggal_discharge"] + fitur_ada + ["readmisi_30_hari"]
kolom_simpan = list(dict.fromkeys(kolom_simpan))   # hilangkan duplikat nama kolom
hasil = df[kolom_simpan].copy()

os.makedirs(config.FOLDER_PROCESSED, exist_ok=True)
path_out = os.path.join(config.FOLDER_PROCESSED, "dataset_model.csv")
hasil.to_csv(path_out, index=False)

n_pos = int(hasil["readmisi_30_hari"].sum())
print("\n" + "=" * 70)
print("HASIL LABEL")
print("=" * 70)
print(f"Total episode            : {len(hasil)}")
print(f"Readmisi 30 hari (1)     : {n_pos}  ({n_pos / len(hasil):.1%})")
print(f"Tidak readmisi (0)       : {len(hasil) - n_pos}  ({1 - n_pos / len(hasil):.1%})")
print(f"Fitur yang dipakai       : {len(fitur_ada)}")
print(f"\nDataset tersimpan di: data/processed/dataset_model.csv")
print("Langkah berikutnya:  python src/03_training_model.py")

if n_pos < 50:
    print("\n[PERINGATAN] Jumlah kasus readmisi sangat sedikit (<50). "
          "Model akan sulit belajar. Tambah periode data atau cek definisi readmisi.")
# Konversi target readmission_risk menjadi biner (1 = Risiko Tinggi, 0 = Rendah/Sedang)
df['target_readmisi'] = df['readmission_risk'].map({
    'High': 1,
    'Medium': 0,
    'Low': 0
})