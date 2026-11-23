# -*- coding: utf-8 -*-
"""
anonimkan_data.py
==================
Menghilangkan identitas langsung pasien dari file tarikan SIMRS SEBELUM
dianalisis. Jalankan ini kalau file dari RS masih memuat No. RM / nama / NIK.

Yang dilakukan:
  1. Menghapus kolom identitas langsung (nama, NIK, alamat, telepon, dll.)
  2. Mengganti No. RM dengan kode acak (hash SHA-256 + 'garam' rahasia).
     Pasien yang sama TETAP mendapat kode yang sama, jadi riwayat kunjungan
     antar episode tetap bisa dilacak.

Cara pakai:
  1. Taruh file asli di data/raw/ (mis. tarikan_simrs_asli.xlsx)
  2. Edit 3 pengaturan di bawah ini
  3. Jalankan:  python src/anonimkan_data.py
  4. Pakai file hasil (berakhiran _anonim.csv) di src/config.py, dan
     SIMPAN FILE ASLI di tempat aman / serahkan kembali ke RS.

PENTING: file asli jangan di-upload ke internet, drive publik, atau chat AI.
"""

import os
import sys
import hashlib
import secrets
import pandas as pd
import config

# ================== EDIT BAGIAN INI ==================
FILE_ASLI = "tarikan_simrs_asli.xlsx"        # nama file asli di data/raw/
KOLOM_ID = "NO_RM"                           # kolom No. RM / ID pasien di file asli
KOLOM_DIHAPUS = ["NAMA", "NIK", "ALAMAT", "NO_TELP", "NAMA_IBU"]   # kolom identitas langsung
# =====================================================

path_asli = os.path.join(config.FOLDER_RAW, FILE_ASLI)
if not os.path.exists(path_asli):
    print(f"[ERROR] File tidak ditemukan: {path_asli}")
    print("Pastikan file ada di folder data/raw/ dan FILE_ASLI sudah benar.")
    sys.exit(1)

if path_asli.lower().endswith((".xlsx", ".xls")):
    df = pd.read_excel(path_asli)
else:
    df = pd.read_csv(path_asli, sep=None, engine="python", encoding_errors="replace")
df.columns = [str(c).strip() for c in df.columns]

if KOLOM_ID not in df.columns:
    print(f"[ERROR] Kolom ID '{KOLOM_ID}' tidak ada. Kolom yang tersedia:")
    for c in df.columns:
        print("  -", c)
    sys.exit(1)

# 'garam' rahasia: dibuat sekali, disimpan lokal (JANGAN dibagikan / di-commit)
path_garam = os.path.join(config.FOLDER_PROYEK, ".garam_anonimisasi.txt")
if os.path.exists(path_garam):
    garam = open(path_garam).read().strip()
else:
    garam = secrets.token_hex(16)
    open(path_garam, "w").write(garam)


def kode_anonim(nilai):
    h = hashlib.sha256((garam + str(nilai).strip()).encode("utf-8")).hexdigest()
    return "P" + h[:10].upper()


df[KOLOM_ID] = df[KOLOM_ID].map(kode_anonim)

ada = [k for k in KOLOM_DIHAPUS if k in df.columns]
tidak_ada = [k for k in KOLOM_DIHAPUS if k not in df.columns]
df = df.drop(columns=ada)

nama_out = os.path.splitext(FILE_ASLI)[0] + "_anonim.csv"
df.to_csv(os.path.join(config.FOLDER_RAW, nama_out), index=False)

print("Anonimisasi selesai.")
print(f"  Kolom identitas dihapus : {ada}")
if tidak_ada:
    print(f"  (kolom tidak ditemukan  : {tidak_ada})")
print(f"  Kolom ID diganti kode   : {KOLOM_ID}")
print(f"  File hasil              : data/raw/{nama_out}")
print("\nLangkah berikutnya: ubah NAMA_FILE_DATA di src/config.py ke file hasil ini.")
print("Periksa juga kolom lain yang mungkin memuat identitas (mis. catatan teks bebas).")
