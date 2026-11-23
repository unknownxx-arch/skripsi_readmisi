# -*- coding: utf-8 -*-
"""utils.py -- fungsi bantu (membaca file + mengganti nama kolom ke standar)."""

import os
import sys
import pandas as pd
import config


def baca_data_mentah():
    """Membaca file di data/raw/ (csv atau xlsx) lalu mengubah nama kolom ke standar."""
    path = os.path.join(config.FOLDER_RAW, config.NAMA_FILE_DATA)

    if not os.path.exists(path):
        print("\n[ERROR] File data tidak ditemukan:")
        print(f"        {path}")
        print("\nSolusi:")
        print("  1. Pastikan file sudah dimasukkan ke folder  data/raw/")
        print("  2. Pastikan nama file di src/config.py (NAMA_FILE_DATA) sama persis,")
        print("     termasuk huruf besar/kecil dan ekstensi (.csv / .xlsx).")
        sys.exit(1)

    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name=config.NAMA_SHEET)
    else:
        # sep=None -> pandas menebak pemisah otomatis (koma / titik koma / tab)
        df = pd.read_csv(path, sep=None, engine="python", encoding_errors="replace")

    # bersihkan spasi di nama kolom
    df.columns = [str(c).strip() for c in df.columns]

    # cek kolom di file vs config
    kolom_hilang = [asli for std, asli in config.PEMETAAN_KOLOM.items()
                    if asli not in df.columns]
    wajib_hilang = [config.PEMETAAN_KOLOM[k] for k in config.KOLOM_WAJIB
                    if config.PEMETAAN_KOLOM.get(k) in kolom_hilang]

    if wajib_hilang:
        print("\n[ERROR] Kolom WAJIB tidak ditemukan di file data kamu:")
        for k in wajib_hilang:
            print(f"        - {k}")
        print("\nKolom yang ADA di file kamu:")
        for c in df.columns:
            print(f"        - {c}")
        print("\nSolusi: sesuaikan bagian PEMETAAN_KOLOM di src/config.py")
        sys.exit(1)

    if kolom_hilang:
        print("[INFO] Kolom fitur berikut tidak ada di file dan akan dilewati:")
        for k in kolom_hilang:
            print(f"        - {k}")

    # ganti nama ke standar, hanya kolom yang ada
    rename_map = {asli: std for std, asli in config.PEMETAAN_KOLOM.items()
                  if asli in df.columns}
    df = df.rename(columns=rename_map)
    df = df[[c for c in config.PEMETAAN_KOLOM.keys() if c in df.columns]].copy()
    return df


def parse_tanggal(seri):
    """Mengubah kolom tanggal menjadi datetime.

    - Format ISO (2024-03-14) dibaca apa adanya.
    - Format lain (14/03/2024) dibaca sebagai HARI/BULAN/TAHUN (standar Indonesia).
    - Jika FORMAT_TANGGAL diisi di config.py, format itu yang dipakai.
    """
    if config.FORMAT_TANGGAL:
        return pd.to_datetime(seri, format=config.FORMAT_TANGGAL, errors="coerce")
    if pd.api.types.is_datetime64_any_dtype(seri):
        return seri
    contoh = seri.dropna().astype(str).head(20)
    if len(contoh) and contoh.str.match(r"^\d{4}-\d{2}-\d{2}").all():
        return pd.to_datetime(seri, errors="coerce")
    return pd.to_datetime(seri, errors="coerce", dayfirst=True)
