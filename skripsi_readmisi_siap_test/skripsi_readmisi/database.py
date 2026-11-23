from datetime import datetime
import json
import mysql.connector
import pandas as pd


def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="db_readmisi",
    )


def simpan_prediksi(kode_pasien, algoritma, probabilitas, hasil, fitur_dict):
    conn = get_connection()
    cursor = conn.cursor()
    query = """
        INSERT INTO riwayat_prediksi 
        (waktu_prediksi, kode_pasien, algoritma, probabilitas_risiko, hasil_klasifikasi, detail_fitur)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    values = (
        datetime.now(),
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
        ORDER BY waktu_prediksi DESC
    """
    df = pd.read_sql(query, conn)
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