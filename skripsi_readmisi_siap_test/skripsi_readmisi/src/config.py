# -*- coding: utf-8 -*-
import os

# Ubah sesuai nama file CSV yang ditaruh di folder data/raw/
NAMA_FILE_DATA = "hospital_readmission_risk.csv"
NAMA_SHEET = 0

PEMETAAN_KOLOM = {
    # A. Identitas Episode Rawat Inap
    "id_pasien":          "patient_id",

    # B. Variabel Target / Outcome
    "target_readmisi":    "readmission_risk",
    "status_pulang":      "discharge_disposition",

    # C. Demografi Pasien
    "usia":               "age",
    "jenis_kelamin":      "gender",
    "jenis_pembayaran":   "insurance_type",

    # D. Data Administratif Masuk
    "jalur_masuk":        "admission_type",
    "departemen":         "department",

    # E. Data Klinis
    "diagnosis_utama":        "primary_diagnosis",
    "komorbiditas":           "chronic_conditions",
    "jumlah_komorbiditas":    "comorbidities_count",
    "jumlah_obat":            "medications_count",
    "jumlah_prosedur":        "procedures_count",
    "tekanan_darah_sistolik": "systolic_bp",
    "tekanan_darah_diastolik":"diastolic_bp",
    "nadi":                   "heart_rate",
    "suhu":                   "temperature",
    "spo2":                   "oxygen_saturation",
    "hb_masuk":               "last_hemoglobin",
    "kreatinin_masuk":        "last_creatinine",
    "gula_darah_masuk":       "last_glucose",

    # F. Riwayat Kunjungan & Lama Rawat
    "los_hari":                         "length_of_stay",
    "jumlah_admisi_365hari_sebelumnya": "num_previous_admissions",
    "jumlah_kunjungan_igd":             "num_emergency_visits",
}

# Kolom wajib disesuaikan agar tidak mencari tanggal_admisi/discharge yang tidak ada di dataset Kaggle
KOLOM_WAJIB = ["id_pasien", "usia", "los_hari", "target_readmisi"]

JENDELA_READMISI_HARI = 30
STATUS_DIKELUARKAN = ["Expired", "Meninggal"]
JALUR_MASUK_TERENCANA = ["Elective", "Elektif", "Elektif (terjadwal)"]
FORMAT_TANGGAL = None

FITUR_NUMERIK = [
    "usia",
    "los_hari",
    "jumlah_komorbiditas",
    "jumlah_obat",
    "jumlah_prosedur",
    "tekanan_darah_sistolik",
    "tekanan_darah_diastolik",
    "nadi",
    "suhu",
    "spo2",
    "hb_masuk",
    "kreatinin_masuk",
    "gula_darah_masuk",
    "jumlah_admisi_365hari_sebelumnya",
    "jumlah_kunjungan_igd",
]

FITUR_KATEGORIKAL = [
    "jenis_kelamin",
    "jenis_pembayaran",
    "jalur_masuk",
    "departemen",
    "diagnosis_utama",
    "komorbiditas",
    "status_pulang",
]

RANDOM_SEED = 42

FOLDER_PROYEK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDER_RAW = os.path.join(FOLDER_PROYEK, "data", "raw")
FOLDER_PROCESSED = os.path.join(FOLDER_PROYEK, "data", "processed")
FOLDER_OUTPUT = os.path.join(FOLDER_PROYEK, "output")