# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import config

os.makedirs(config.FOLDER_RAW, exist_ok=True)
np.random.seed(config.RANDOM_SEED)

n_pasien = 2500
n_episode = 4200

id_pasien_list = [f"PSN{i:05d}" for i in range(1, n_pasien + 1)]
pasien_pilih = np.random.choice(id_pasien_list, size=n_episode, replace=True)

base_date = pd.to_datetime("2024-01-01")
tgl_admisi = base_date + pd.to_timedelta(np.random.randint(0, 700, size=n_episode), unit="D") + \
             pd.to_timedelta(np.random.randint(0, 24, size=n_episode), unit="h")
los = np.random.choice([1, 2, 3, 4, 5, 6, 7, 10, 14], size=n_episode, p=[0.1, 0.2, 0.25, 0.18, 0.1, 0.07, 0.05, 0.03, 0.02])
tgl_discharge = tgl_admisi + pd.to_timedelta(los, unit="D")

df = pd.DataFrame({
    "ID Pasien (anonim)": pasien_pilih,
    "ID Episode/No. Kunjungan": [f"ADM2024{i:06d}" for i in range(1, n_episode + 1)],
    "Tanggal & Jam Admisi": tgl_admisi.strftime("%Y-%m-%d %H:%M"),
    "Tanggal & Jam Discharge (Pulang)": tgl_discharge.strftime("%Y-%m-%d %H:%M"),
    "Status Pulang": np.random.choice(["Pulang Normal", "APS (Pulang Paksa)", "Rujuk", "Meninggal"], size=n_episode, p=[0.85, 0.07, 0.04, 0.04]),
    "Usia": np.random.randint(18, 88, size=n_episode),
    "Jenis Kelamin": np.random.choice(["Laki-laki", "Perempuan"], size=n_episode),
    "Jenis Pembayaran/Asuransi": np.random.choice(["BPJS", "Asuransi Swasta", "Umum"], size=n_episode, p=[0.75, 0.15, 0.10]),
    "Domisili (Kab/Kota)": np.random.choice(["Jakarta Selatan", "Jakarta Timur", "Jakarta Barat", "Tangerang", "Depok", "Bekasi"], size=n_episode),
    "Jalur Masuk": np.random.choice(["IGD", "Rawat Jalan (Poli)", "Rujukan RS Lain", "Elektif"], size=n_episode, p=[0.55, 0.25, 0.12, 0.08]),
    "Ruangan / Kelas Rawat": np.random.choice(["Kelas 3", "Kelas 2", "Kelas 1", "VIP", "ICU", "HCU"], size=n_episode, p=[0.35, 0.25, 0.20, 0.08, 0.06, 0.06]),
    "Departemen / SMF": np.random.choice(["Penyakit Dalam", "Bedah", "Jantung", "Saraf", "Paru", "Obgyn", "Anak"], size=n_episode),
    "Kode Dokter DPJP (anonim)": [f"DPJP-{i:02d}" for i in np.random.randint(1, 25, size=n_episode)],
    "Diagnosis Utama (ICD-10)": np.random.choice(["I50.0 (Gagal Jantung)", "E11.9 (Diabetes Melitus)", "I10 (Hipertensi)", "J18.9 (Pneumonia)", "N18.9 (Ginjal Kronis)", "I63.9 (Stroke)"], size=n_episode),
    "Diagnosis Sekunder (ICD-10)": np.random.choice(["Tidak Ada", "E11.9", "I10", "N18.3", "E78.5", "I50.9"], size=n_episode),
    "Tindakan/Prosedur (ICD-9-CM)": np.random.choice(["Tidak Ada", "88.72 (Ekokardiografi)", "39.95 (Hemodialisis)", "96.71 (Ventilasi Mekanik)", "87.44 (Foto Toraks)"], size=n_episode),
    "Komorbiditas": np.random.choice(["Tidak Ada", "Hipertensi", "Diabetes", "Hipertensi, Diabetes", "Penyakit Jantung", "Penyakit Ginjal"], size=n_episode),
    "jumlah_komorbiditas": np.random.choice([0, 1, 2, 3, 4], size=n_episode, p=[0.25, 0.35, 0.22, 0.12, 0.06]),
    "tekanan_darah_sistolik": np.random.randint(95, 185, size=n_episode),
    "tekanan_darah_diastolik": np.random.randint(60, 115, size=n_episode),
    "nadi": np.random.randint(60, 125, size=n_episode),
    "suhu": np.round(np.random.uniform(36.0, 39.2, size=n_episode), 1),
    "spo2": np.random.randint(88, 100, size=n_episode),
    "skor_gcs": np.random.choice([15, 14, 13, 12, 10], size=n_episode, p=[0.78, 0.10, 0.06, 0.04, 0.02]),
    "hb_masuk": np.round(np.random.uniform(7.5, 16.5, size=n_episode), 1),
    "leukosit_masuk": np.round(np.random.uniform(4.0, 22.0, size=n_episode), 1),
    "kreatinin_masuk": np.round(np.random.uniform(0.6, 4.5, size=n_episode), 2),
    "gula_darah_masuk": np.random.randint(80, 340, size=n_episode),
    "albumin_masuk": np.round(np.random.uniform(2.2, 4.8, size=n_episode), 1),
    "Kondisi/Tanda Vital Saat Pulang": np.random.choice(["Stabil", "Perbaikan Klinis", "Perlu Pemantauan Ketat"], size=n_episode, p=[0.70, 0.22, 0.08]),
    "Jumlah Admisi 6-12 Bulan Terakhir": np.random.choice([0, 1, 2, 3, 4], size=n_episode, p=[0.50, 0.25, 0.15, 0.07, 0.03]),
    "Jumlah Kunjungan IGD (periode sama)": np.random.choice([0, 1, 2, 3, 4, 5], size=n_episode, p=[0.40, 0.25, 0.18, 0.10, 0.05, 0.02]),
    "Riwayat Readmisi Sebelumnya": np.random.choice(["Tidak", "Ya"], size=n_episode, p=[0.75, 0.25]),
    "LOS Admisi Sebelumnya": np.random.choice([0, 2, 3, 4, 5, 7, 10], size=n_episode, p=[0.50, 0.10, 0.12, 0.10, 0.08, 0.06, 0.04]),
    "Edukasi Discharge Terstruktur": np.random.choice(["Ya", "Tidak", "Tidak tercatat"], size=n_episode, p=[0.60, 0.20, 0.20]),
    "Jadwal Kontrol/Follow-up Ditetapkan": np.random.choice(["Ya", "Tidak"], size=n_episode, p=[0.80, 0.20]),
    "Kehadiran Kontrol Sesuai Jadwal": np.random.choice(["Ya", "Tidak", "Tidak tercatat"], size=n_episode, p=[0.55, 0.25, 0.20]),
})

df = df.sort_values(["ID Pasien (anonim)", "Tanggal & Jam Admisi"]).reset_index(drop=True)
out_file = os.path.join(config.FOLDER_RAW, config.NAMA_FILE_DATA)
df.to_csv(out_file, index=False)
print(f"[SUKSES] Data simulasi 27 variabel RS dibuat: {out_file} ({len(df)} baris)")