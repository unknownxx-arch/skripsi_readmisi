# Prediksi Risiko Readmisi 30 Hari — Panduan Lengkap (VS Code)

Proyek ini membaca data rawat inap dari RS, membuat label readmisi 30 hari,
melatih beberapa model machine learning, lalu menghasilkan tabel & grafik
untuk Bab 4 skripsi.

```
skripsi_readmisi/
├── data/
│   ├── raw/          <- TARUH DATA RS DI SINI
│   └── processed/    <- hasil olahan (dibuat otomatis)
├── output/           <- tabel, grafik, model (dibuat otomatis)
├── src/
│   ├── config.py               <- SATU-SATUNYA file yang kamu edit
│   ├── 00_buat_data_contoh.py  <- data simulasi untuk latihan
│   ├── anonimkan_data.py       <- hapus identitas pasien (jika perlu)
│   ├── 01_cek_data.py          <- periksa kualitas data
│   ├── 02_buat_label.py        <- buat label readmisi + fitur riwayat
│   ├── 03_training_model.py    <- latih & bandingkan model
│   └── utils.py
├── requirements.txt
└── README.md
```

---

## BAGIAN A — Persiapan (cukup sekali)

### A1. Install Python
1. Buka https://www.python.org/downloads/ lalu download Python 3.10 atau lebih baru.
2. Saat installer terbuka, **centang "Add python.exe to PATH"** (di bagian bawah jendela pertama), lalu klik *Install Now*.
3. Cek: buka *Command Prompt* dan ketik `python --version`. Harus muncul versi Python.

### A2. Install VS Code
1. Download di https://code.visualstudio.com/ lalu install.
2. Buka VS Code, klik ikon **Extensions** (kotak-kotak di sisi kiri, atau `Ctrl+Shift+X`).
3. Install dua extension ini:
   - **Python** (oleh Microsoft)
   - **Rainbow CSV** (memudahkan membaca file CSV)

### A3. Buka proyek
1. Ekstrak file `skripsi_readmisi.zip` ke folder, misalnya `D:\skripsi_readmisi`.
2. Di VS Code: **File → Open Folder…** → pilih folder `skripsi_readmisi`. Klik *Yes, I trust the authors*.
3. Di panel kiri (Explorer) kamu akan melihat folder `data`, `src`, dll.

### A4. Buat lingkungan Python & install library
1. Buka terminal di VS Code: menu **Terminal → New Terminal** (atau `Ctrl+` ` `).
2. Ketik perintah berikut satu per satu, tekan Enter di tiap baris.

**Windows:**
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Mac / Linux:**
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tanda berhasil: di awal baris terminal muncul `(.venv)`.

> **Jika muncul error "running scripts is disabled" (Windows PowerShell):**
> ketik `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, Enter, lalu ulangi perintah `.venv\Scripts\activate`.

3. Pilih interpreter: tekan `Ctrl+Shift+P`, ketik **Python: Select Interpreter**, pilih yang bertuliskan `.venv`.

> Setiap kali membuka VS Code lagi, jalankan ulang `.venv\Scripts\activate` (Windows) atau `source .venv/bin/activate` (Mac/Linux) di terminal baru.

---

## BAGIAN B — Latihan dengan data contoh (lakukan dulu!)

Jalankan berurutan di terminal (pastikan ada `(.venv)`):

```
python src/00_buat_data_contoh.py
python src/01_cek_data.py
python src/02_buat_label.py
python src/03_training_model.py
```

Setelah selesai, buka folder `output/`: ada tabel perbandingan model
(`hasil_evaluasi_model.csv`), kurva ROC, confusion matrix, dan grafik faktor
paling berpengaruh.

> ⚠️ Data contoh adalah SIMULASI. Jangan pernah dilaporkan sebagai hasil penelitian.
> Fungsinya hanya agar kamu paham alurnya sebelum memakai data RS asli.

---

## BAGIAN C — Memasukkan DATA RS ASLI (langkah demi langkah)

### C1. Pastikan bentuk data yang kamu terima benar
Data dari SIMRS harus berbentuk **tabel datar**: **1 baris = 1 episode rawat inap**.

| Syarat | Alasan |
|---|---|
| Ada kolom **ID pasien** (No. RM) yang **sama** untuk kunjungan berbeda dari pasien yang sama | Tanpa ini riwayat & label readmisi tidak bisa dibuat |
| Ada **tanggal admisi** dan **tanggal discharge** | Untuk menghitung jarak ke admisi berikutnya |
| Ada **status pulang** (pulang, APS, rujuk, meninggal) | Pasien meninggal dikeluarkan dari analisis |
| Periode data minimal 1–2 tahun | Agar ada cukup kasus readmisi |
| Format `.csv` atau `.xlsx` | Dua format ini didukung |

Daftar kolom lengkap ada di file `Daftar_Permintaan_Data_RS.xlsx` yang sudah kita siapkan.

### C2. Hilangkan identitas pasien (WAJIB sebelum dianalisis)
- Jika file dari RS **sudah anonim** (tidak ada nama/NIK/alamat; No. RM sudah berupa kode) → lanjut ke C3.
- Jika file **masih memuat No. RM asli, nama, NIK, alamat**:
  1. Taruh file asli di `data/raw/`
  2. Buka `src/anonimkan_data.py`, edit 3 baris di bagian *EDIT BAGIAN INI* (nama file, nama kolom ID, daftar kolom identitas yang dihapus)
  3. Jalankan: `python src/anonimkan_data.py`
  4. Hasilnya: file baru berakhiran `_anonim.csv`. **Pakai file ini** untuk langkah berikutnya.
  5. File asli disimpan di tempat aman atau dikembalikan ke RS, **jangan dikirim ke mana pun**.

### C3. Masukkan file ke folder `data/raw/`
Pilih salah satu cara:
- **Cara termudah (drag & drop):** di VS Code, buka panel Explorer, lalu **tarik file dari File Explorer Windows** dan jatuhkan ke folder `data` → `raw`.
- **Atau:** buka File Explorer Windows, salin file, lalu tempel (paste) ke `skripsi_readmisi\data\raw\`.
- **Atau:** klik kanan folder `raw` di VS Code → **Reveal in File Explorer** → tempel file di situ.

Hapus file data contoh (`data_rs_mentah.csv`) dari folder itu supaya tidak tertukar.

> Jika file `.xlsx` sedang terbuka di Excel, tutup dulu. Kalau tidak, akan muncul `PermissionError`.

### C4. Atur `src/config.py`
Buka `src/config.py`, lalu ubah **hanya** bagian berikut.

**1) Nama file** (harus persis sama, termasuk huruf besar/kecil dan ekstensi):
```python
NAMA_FILE_DATA = "tarikan_simrs_anonim.csv"
```
Untuk Excel dengan banyak sheet: `NAMA_SHEET = "Sheet1"` (isi nama sheet; biarkan `0` untuk sheet pertama).

**2) Pemetaan kolom** — kolom kanan diisi dengan nama kolom **persis seperti di file RS-mu**.
Contoh jika di file RS namanya berbeda:
```python
PEMETAAN_KOLOM = {
    "id_pasien":         "NO_RM",
    "tanggal_admisi":    "TGL_MASUK",
    "tanggal_discharge": "TGL_KELUAR",
    "status_pulang":     "CARA_PULANG",
    "usia":              "UMUR",
    "jenis_kelamin":     "JK",
    # ... dst
}
```
- Kolom fitur yang **tidak ada** di file RS → **hapus barisnya** (program otomatis melewatinya).
- 4 kolom WAJIB (id, tanggal admisi, tanggal discharge, status pulang) tidak boleh dihapus.

**3) Cocokkan nilai status pulang.** Lihat nilai asli di file (hasil `01_cek_data.py` menampilkannya), lalu tulis persis:
```python
STATUS_DIKELUARKAN = ["Meninggal"]       # atau "MENINGGAL", "Meninggal Dunia", dll.
```

**4) Admisi terencana** (kontrol/kemoterapi terjadwal tidak dihitung readmisi):
```python
JALUR_MASUK_TERENCANA = ["Elektif"]      # sesuaikan dengan isi kolom jalur masuk; [] jika tidak ada
```

**5) Format tanggal** — biasanya biarkan `None` (otomatis). Jika tanggal terbaca salah/kosong, isi mis. `"%d/%m/%Y"` (untuk 14/03/2024).

### C5. Jalankan pemeriksaan data
```
python src/01_cek_data.py
```
Baca bagian **"TEMUAN YANG PERLU DIPERHATIKAN"** di akhir. Contoh masalah umum:
- *Kolom kosong >30%* → pertimbangkan membuang kolom itu dari `PEMETAAN_KOLOM`
- *Discharge sebelum admisi* → data janggal, program otomatis membuang baris tersebut
- *Tidak ada pasien dengan >1 episode* → ID pasien tidak konsisten, minta data ulang ke IT RS

### C6. Buat label & latih model
```
python src/02_buat_label.py
python src/03_training_model.py
```
Cek di `02_buat_label.py` bahwa **jumlah dan persentase readmisi masuk akal**
(umumnya 5–25% untuk data RS umum). Jika hasilnya 0% atau hampir 100%, cek ulang
pemetaan kolom ID pasien dan format tanggal.

---

## BAGIAN D — Membaca hasil untuk Bab 4

| File di `output/` | Dipakai untuk |
|---|---|
| `laporan_cek_data.txt` | Bab 3/4: deskripsi dan kualitas data |
| `hasil_evaluasi_model.csv` | Bab 4: tabel perbandingan Accuracy, Precision, Recall, F1, AUC-ROC |
| `roc_curve_comparison.png` | Bab 4: gambar kurva ROC |
| `confusion_matrix_*.png` | Bab 4: gambar confusion matrix tiap model |
| `feature_importance.csv/.png` | Bab 4: faktor paling berpengaruh (menjawab rumusan masalah 1) |
| `model_terbaik.joblib` | Model tersimpan untuk dipakai ulang |

Catatan metodologi yang bisa langsung kamu tulis di Bab 3:
- Pembagian data latih:uji = 80:20 **per pasien** (mencegah kebocoran data karena pasien yang sama tidak muncul di kedua bagian).
- Ketidakseimbangan kelas ditangani dengan pembobotan kelas (*class weight*).
- Kestabilan model dinilai dengan validasi silang 5-fold (*GroupKFold*).
- Episode dengan pasien meninggal dan episode yang masa pengamatan 30 harinya belum lengkap dikeluarkan.
- Interpretasi: *permutation importance* pada data uji.

Ingin menambahkan XGBoost: `pip install xgboost` di terminal; skrip otomatis ikut membandingkannya.

---

## BAGIAN E — Masalah yang sering muncul

| Pesan / gejala | Penyebab & solusi |
|---|---|
| `'python' is not recognized` | Python belum masuk PATH. Install ulang dan centang *Add python.exe to PATH*, lalu tutup & buka lagi VS Code |
| `ModuleNotFoundError: No module named 'pandas'` | Belum aktif `.venv` atau belum install. Jalankan `.venv\Scripts\activate` lalu `pip install -r requirements.txt` |
| `running scripts is disabled` | Jalankan `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `File data tidak ditemukan` | Nama file di `NAMA_FILE_DATA` tidak sama persis dengan file di `data/raw/` |
| `Kolom WAJIB tidak ditemukan` | Nama kolom di sisi kanan `PEMETAAN_KOLOM` tidak sama dengan header file. Daftar kolom yang ada ikut ditampilkan di pesan error |
| `PermissionError` | File Excel masih terbuka. Tutup Excel lalu ulangi |
| Karakter aneh (�) di data CSV | Simpan ulang CSV dari Excel dengan *CSV UTF-8* |
| Readmisi 0% / terlalu banyak | ID pasien tidak konsisten atau tanggal salah format; cek `FORMAT_TANGGAL` |
| Jumlah readmisi < 50 | Data terlalu sedikit; minta periode lebih panjang ke RS |

---

## BAGIAN F — Aturan penting soal data pasien

1. **Jangan upload data pasien asli** ke internet, Google Drive publik, GitHub, maupun chat AI (termasuk Claude). Data asli hanya di komputer yang kamu kontrol.
2. Jika kamu butuh bantuan saat error, **cukup salin pesan error-nya** atau beberapa baris data **palsu** (hasil `00_buat_data_contoh.py`). Jangan tempel baris data pasien sungguhan.
3. Pastikan izin RS dan ethical clearance sudah beres sebelum menarik data.
4. Hasil dari data contoh bukan hasil penelitian. Hanya hasil dari data RS asli yang boleh masuk Bab 4.
