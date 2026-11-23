# -*- coding: utf-8 -*-
"""
01_cek_data.py
===============
Memeriksa data mentah RS SEBELUM dipakai:
  - apakah file terbaca dan kolom sesuai config
  - jumlah baris, pasien unik, rentang tanggal
  - data kosong (missing) per kolom
  - baris duplikat, tanggal tidak valid, discharge < admisi
  - berapa pasien yang punya lebih dari 1 episode (syarat membuat label readmisi)

Cara jalankan:   python src/01_cek_data.py
"""

import os
import pandas as pd
import config
from utils import baca_data_mentah, parse_tanggal

print("=" * 70)
print("PEMERIKSAAN DATA MENTAH")
print("=" * 70)

df = baca_data_mentah()
print(f"\nFile terbaca: {config.NAMA_FILE_DATA}")
print(f"Jumlah baris : {len(df)}")
print(f"Jumlah kolom : {df.shape[1]}")

# --- tanggal ---
df["tanggal_admisi"] = parse_tanggal(df["tanggal_admisi"])
df["tanggal_discharge"] = parse_tanggal(df["tanggal_discharge"])

masalah = []

n_tgl_rusak = df["tanggal_admisi"].isna().sum() + df["tanggal_discharge"].isna().sum()
if n_tgl_rusak > 0:
    masalah.append(f"{n_tgl_rusak} sel tanggal tidak terbaca (cek FORMAT_TANGGAL di config.py)")

n_terbalik = (df["tanggal_discharge"] < df["tanggal_admisi"]).sum()
if n_terbalik > 0:
    masalah.append(f"{n_terbalik} baris tanggal discharge SEBELUM tanggal admisi (data janggal)")

n_dup = df.duplicated().sum()
if n_dup > 0:
    masalah.append(f"{n_dup} baris duplikat persis")

print("\n--- Informasi umum ---")
print(f"Pasien unik        : {df['id_pasien'].nunique()}")
print(f"Rentang tanggal    : {df['tanggal_admisi'].min().date()}  s/d  {df['tanggal_discharge'].max().date()}")
per_pasien = df.groupby("id_pasien").size()
print(f"Pasien dgn >1 episode : {(per_pasien > 1).sum()} "
      f"({(per_pasien > 1).mean():.1%} dari semua pasien)")
if (per_pasien > 1).sum() == 0:
    masalah.append("Tidak ada pasien dengan lebih dari 1 episode: label readmisi TIDAK bisa dibuat. "
                   "Pastikan id_pasien konsisten antar kunjungan (No. RM yang sama).")

print("\n--- Distribusi status pulang ---")
print(df["status_pulang"].value_counts(dropna=False).to_string())

print("\n--- Data kosong (missing) per kolom ---")
missing = df.isna().sum()
persen = (missing / len(df) * 100).round(1)
tabel = pd.DataFrame({"jumlah_kosong": missing, "persen": persen})
print(tabel.to_string())
for kol, p in persen.items():
    if p > 30:
        masalah.append(f"Kolom '{kol}' kosong {p}% (pertimbangkan dibuang atau minta data ulang)")

print("\n--- Statistik kolom numerik ---")
num = df.select_dtypes(include="number")
if not num.empty:
    print(num.describe().round(2).T.to_string())

print("\n" + "=" * 70)
if masalah:
    print("TEMUAN YANG PERLU DIPERHATIKAN:")
    for m in masalah:
        print(f"  ! {m}")
else:
    print("Tidak ditemukan masalah besar. Lanjut ke:  python src/02_buat_label.py")
print("=" * 70)

# simpan laporan
os.makedirs(config.FOLDER_OUTPUT, exist_ok=True)
with open(os.path.join(config.FOLDER_OUTPUT, "laporan_cek_data.txt"), "w", encoding="utf-8") as f:
    f.write(f"Baris: {len(df)}\nPasien unik: {df['id_pasien'].nunique()}\n")
    f.write("Temuan:\n" + ("\n".join(f"- {m}" for m in masalah) if masalah else "- tidak ada\n"))
    f.write("\n\nMissing per kolom:\n" + tabel.to_string())
print("Laporan disimpan di: output/laporan_cek_data.txt")
