# -*- coding: utf-8 -*-
"""
run_all.py
===========
Menjalankan seluruh tahap secara berurutan, berhenti jika ada yang gagal.

Letakkan file ini di folder utama proyek (sejajar dengan folder src dan data),
lalu jalankan:   python run_all.py
"""

import os
import subprocess
import sys

FOLDER = os.path.dirname(os.path.abspath(__file__))

TAHAP = [
    ("1. Cek data", "src/01_cek_data.py"),
    ("2. Buat label readmisi", "src/02_buat_label.py"),
    ("3. Training & evaluasi model", "src/03_training_model.py"),
]

for nama, skrip in TAHAP:
    print("\n" + "#" * 70)
    print(f"# TAHAP {nama}")
    print("#" * 70)
    hasil = subprocess.run([sys.executable, os.path.join(FOLDER, skrip)], cwd=FOLDER)
    if hasil.returncode != 0:
        print(f"\n[BERHENTI] Tahap '{nama}' gagal. Perbaiki error di atas lalu jalankan lagi.")
        sys.exit(hasil.returncode)

print("\n" + "=" * 70)
print("SEMUA TAHAP SELESAI. Hasil ada di folder 'output'.")
print("=" * 70)