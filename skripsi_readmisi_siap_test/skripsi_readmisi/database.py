from datetime import datetime
import json
import os
import sqlite3
import pandas as pd

# Menyimpan file database lokal otomatis di folder yang sama dengan database.py
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db_readmisi.sqlite")


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    # Otomatis membuat tabel riwayat_prediksi jika belum ada
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS riwayat_prediksi (
            id_prediksi INTEGER PRIMARY KEY AUTOINCREMENT,
            waktu_prediksi TEXT,
            kode_pasien TEXT,
            algoritma TEXT,
            probabilitas_risiko REAL,
            hasil_klasifikasi TEXT,
            detail_fitur TEXT
        )
    """)
    conn.commit()
    return conn


def simpan_prediksi(kode_pasien, algoritma, probabilitas, hasil, fitur_dict):
    conn = get_connection()
    cursor = conn.cursor()
    query = """
        INSERT INTO riwayat_prediksi 
        (waktu_prediksi, kode_pasien, algoritma, probabilitas_risiko, hasil_klasifikasi, detail_fitur)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    values = (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        kode_pasien,
        algoritma,
        float(probabilitas),
        hasil,
        json.dumps(fitur_dict),
    )
    cursor.execute(query, values)
    conn.commit()
    cursor.close()
    conn.close()


def ambil_riwayat():
    conn = get_connection()
    query = """
        SELECT id_prediksi, waktu_prediksi, kode_pasien, algoritma, 
               ROUND(probabilitas_risiko * 100, 2) AS risiko_persen, hasil_klasifikasi 
        FROM riwayat_prediksi 
        ORDER BY id_prediksi DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def buat_kode_pasien_baru():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COALESCE(MAX(id_prediksi), 0) + 1 FROM riwayat_prediksi")
    nomor_urut = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return f"PSN-{nomor_urut:03d}"
