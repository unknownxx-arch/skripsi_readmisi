from datetime import datetime
import io
import os
import sys
from database import  ambil_riwayat, ambil_semua_raw, buat_kode_pasien_baru, simpan_perubahan_admin, simpan_prediksi
import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import config

st.set_page_config(
    page_title="Prediksi Readmisi 30 Hari Pasien Rumah Sakit",
    page_icon="🏥",
    layout="wide",
)

st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 1.2rem;
    }
    .main-header h1 { margin: 0; font-size: 1.75rem; color: #F8FAFC; }
    .main-header p { margin: 0.35rem 0 0 0; color: #94A3B8; font-size: 0.92rem; }
    .cat-header {
        background-color: #161B22;
        padding: 0.65rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin: 0.8rem 0;
        font-weight: 600;
        color: #E2E8F0;
    }
    .result-box-high {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.2), rgba(153, 27, 27, 0.28));
        border: 1px solid #EF4444;
        border-radius: 12px;
        padding: 1.4rem;
    }
    .result-box-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(6, 95, 70, 0.28));
        border: 1px solid #10B981;
        border-radius: 12px;
        padding: 1.4rem;
    }
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(90deg, #2563EB, #3B82F6);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        border-radius: 8px;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="main-header">
    <h1>🏥 Sistem Prediksi Risiko Readmisi & Lama Rawat Inap Pasien</h1>
    <p>Clinical Decision Support Dashboard • Simulasi Satuan & Prediksi Massal Berbasis Machine Learning</p>
</div>
""",
    unsafe_allow_html=True,
)

model_path = os.path.join(config.FOLDER_OUTPUT, "model_terbaik.joblib")
los_model_path = os.path.join(config.FOLDER_OUTPUT, "model_los.joblib")
data_path = os.path.join(config.FOLDER_PROCESSED, "dataset_model.csv")
eval_path = os.path.join(config.FOLDER_OUTPUT, "hasil_evaluasi_model.csv")

if not os.path.exists(model_path) or not os.path.exists(data_path):
    st.error(
        "File model_terbaik.joblib atau dataset_model.csv belum ditemukan."
        " Jalankan pipeline training terlebih dahulu."
    )
    st.stop()

model = joblib.load(model_path)
df_ref = pd.read_csv(data_path)

# Ambil nama model terbaik dari hasil evaluasi jika tersedia
nama_model_aktif = "Model Terbaik"
if os.path.exists(eval_path):
    nama_model_aktif = str(pd.read_csv(eval_path).iloc[0]["Model"])

# Model LOS bersifat opsional: jika belum dilatih, dashboard tetap berjalan seperti biasa
los_bundle = (
    joblib.load(los_model_path) if os.path.exists(los_model_path) else None
)
los_ada = los_bundle is not None

fitur_num = [c for c in config.FITUR_NUMERIK if c in df_ref.columns]
fitur_cat = [c for c in config.FITUR_KATEGORIKAL if c in df_ref.columns]
semua_fitur = fitur_num + fitur_cat


def siapkan_df_los(df_in: pd.DataFrame) -> pd.DataFrame:
    """Menyiapkan data input agar sesuai dengan fitur model LOS (nilai kosong -> NaN)."""
    df_out = df_in.copy()
    for col in los_bundle["fitur"]:
        if col not in df_out.columns:
            df_out[col] = np.nan
    df_out = df_out[los_bundle["fitur"]]
    for col in los_bundle["fitur_num"]:
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce")
    for col in los_bundle["fitur_cat"]:
        df_out[col] = df_out[col].astype("object").where(df_out[col].notna(), np.nan)
    return df_out


def prediksi_los(df_in: pd.DataFrame):
    """Mengembalikan array estimasi LOS (hari)."""
    return np.expm1(los_bundle["model"].predict(siapkan_df_los(df_in)))


def bersihkan_fitur_json(data_dict: dict) -> dict:
    """Konversi tipe NumPy/Pandas ke tipe bawaan Python agar aman disimpan ke JSON MySQL."""
    hasil = {}
    for k, v in data_dict.items():
        if pd.isna(v):
            hasil[k] = None
        elif isinstance(v, (np.integer, int)) and not isinstance(v, bool):
            hasil[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            hasil[k] = float(v)
        else:
            hasil[k] = str(v)
    return hasil


with st.sidebar:
    st.markdown("### ⚙️ Pengaturan Prediksi")
    threshold = st.slider(
        "Ambang Batas (Threshold) Risiko Tinggi",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
    )
    st.divider()
    st.markdown("### 📊 Statistik Dataset Latih")
    st.metric("Total Episode Rawat", f"{len(df_ref):,}")
    st.metric("Jumlah Fitur Aktif", f"{len(semua_fitur)} Variabel")
    if os.path.exists(eval_path):
        best_row = pd.read_csv(eval_path).iloc[0]
        st.info(
            f"**Model Readmisi:** {best_row['Model']}\n\n**AUC-ROC:**"
            f" {best_row['AUC-ROC']} | **Recall:** {best_row['Recall']}"
        )
    if los_ada:
        m = los_bundle["metrik"]
        st.info(
            "**Model Lama Rawat (LOS):** Gradient Boosting\n\n**MAE:**"
            f" {m['MAE']:.2f} hari | **R²:** {m['R2']:.2f}"
        )
    else:
        st.warning(
            "Model LOS belum tersedia. Jalankan `python train_los.py` untuk"
            " mengaktifkan estimasi lama rawat."
        )

LABEL_FITUR = {
    "usia": "Usia Pasien (Tahun)",
    "jenis_kelamin": "Jenis Kelamin",
    "jenis_pembayaran": "Jenis Pembayaran / Asuransi",
    "domisili": "Domisili (Kab/Kota)",
    "jalur_masuk": "Jalur Masuk",
    "kelas_rawat": "Ruangan / Kelas Rawat",
    "departemen": "Departemen / SMF",
    "kode_dpjp": "Kode Dokter DPJP (Anonim)",
    "los_hari": "Lama Rawat Inap / LOS (Hari)",
    "status_pulang": "Status Pulang",
    "diagnosis_utama": "Diagnosis Utama (ICD-10)",
    "diagnosis_sekunder": "Diagnosis Sekunder (ICD-10)",
    "tindakan_prosedur": "Tindakan / Prosedur (ICD-9-CM)",
    "komorbiditas": "Jenis Komorbiditas Utama",
    "jumlah_komorbiditas": "Jumlah Komorbiditas",
    "tekanan_darah_sistolik": "TD Sistolik Saat Masuk (mmHg)",
    "tekanan_darah_diastolik": "TD Diastolik Saat Masuk (mmHg)",
    "nadi": "Denyut Nadi (x/menit)",
    "suhu": "Suhu Tubuh (°C)",
    "spo2": "Saturasi Oksigen / SpO2 (%)",
    "skor_gcs": "Skor GCS (3 - 15)",
    "kondisi_saat_pulang": "Kondisi / Tanda Vital Saat Pulang",
    "hb_masuk": "Hemoglobin / Hb (g/dL)",
    "leukosit_masuk": "Leukosit (10³/µL)",
    "kreatinin_masuk": "Kreatinin (mg/dL)",
    "gula_darah_masuk": "Gula Darah Sewaktu (mg/dL)",
    "albumin_masuk": "Albumin (g/dL)",
    "jumlah_admisi_365hari_sebelumnya": "Jumlah Admisi 6–12 Bulan Terakhir",
    "jumlah_kunjungan_igd": "Jumlah Kunjungan IGD (Periode Sama)",
    "riwayat_readmisi_sebelumnya": "Riwayat Readmisi Sebelumnya",
    "los_admisi_sebelumnya": "LOS Admisi Sebelumnya (Hari)",
    "episode_ini_adalah_readmisi": "Status Episode Saat Ini",
    "edukasi_discharge": "Edukasi Discharge Terstruktur",
    "jadwal_kontrol": "Jadwal Kontrol / Follow-up Ditetapkan",
    "kehadiran_kontrol": "Kehadiran Kontrol Sesuai Jadwal",
}


def render_input_widget(col_name, container):
    label = LABEL_FITUR.get(col_name, col_name.replace("_", " ").title())
    with container:
        if col_name in fitur_num:
            val_median = (
                float(df_ref[col_name].median())
                if pd.notna(df_ref[col_name].median())
                else 0.0
            )
            if col_name == "episode_ini_adalah_readmisi":
                return st.selectbox(
                    label,
                    [0, 1],
                    format_func=lambda x: (
                        "0 - Admisi Baru" if x == 0 else "1 - Ya, Readmisi"
                    ),
                )
            elif col_name == "suhu":
                return st.number_input(
                    label, 30.0, 43.0, round(val_median, 1), step=0.1
                )
            elif col_name == "spo2":
                return st.number_input(label, 50, 100, int(val_median), step=1)
            elif col_name == "skor_gcs":
                return st.number_input(
                    label,
                    3,
                    15,
                    int(val_median) if val_median >= 3 else 15,
                    step=1,
                )
            elif pd.api.types.is_integer_dtype(df_ref[col_name].dropna()):
                return st.number_input(
                    label, min_value=0, value=int(val_median), step=1
                )
            else:
                return st.number_input(
                    label, min_value=0.0, value=round(val_median, 2), step=0.1
                )
        elif col_name in fitur_cat:
            seri = df_ref[col_name].dropna().astype(str)
            if col_name == "status_pulang":
                seri = seri[~seri.isin(config.STATUS_DIKELUARKAN)]
            opsi = sorted(seri.unique().tolist()) or ["Tidak tercatat"]
            populer = seri.value_counts().index.tolist()
            idx_awal = opsi.index(populer[0]) if populer else 0
            bantuan = (
                "Pilih dari daftar, atau ketik nilai baru lalu tekan Enter."
            )

            try:
                nilai = st.selectbox(
                    label,
                    options=opsi,
                    index=idx_awal,
                    accept_new_options=True,
                    help=bantuan,
                    key=f"sel_{col_name}",
                )
                return nilai if nilai not in (None, "") else np.nan
            except TypeError:
                nilai = st.selectbox(
                    label, options=opsi, index=idx_awal, key=f"sel_{col_name}"
                )
                manual = st.text_input(
                    f"{label} - atau ketik sendiri",
                    value="",
                    key=f"man_{col_name}",
                    help="Kosongkan untuk memakai pilihan dropdown di atas.",
                ).strip()
                return manual if manual else nilai
    return None


GRUP_DEMOGRAFI_ADMIN = [
    "usia",
    "jenis_kelamin",
    "jenis_pembayaran",
    "domisili",
    "jalur_masuk",
    "kelas_rawat",
    "departemen",
    "kode_dpjp",
    "los_hari",
    "status_pulang",
]
GRUP_KLINIS_VITAL = [
    "diagnosis_utama",
    "diagnosis_sekunder",
    "tindakan_prosedur",
    "komorbiditas",
    "jumlah_komorbiditas",
    "tekanan_darah_sistolik",
    "tekanan_darah_diastolik",
    "nadi",
    "suhu",
    "spo2",
    "skor_gcs",
    "kondisi_saat_pulang",
]
GRUP_LAB = [
    "hb_masuk",
    "leukosit_masuk",
    "kreatinin_masuk",
    "gula_darah_masuk",
    "albumin_masuk",
]
GRUP_RIWAYAT_DISCHARGE = [
    "jumlah_admisi_365hari_sebelumnya",
    "jumlah_kunjungan_igd",
    "riwayat_readmisi_sebelumnya",
    "los_admisi_sebelumnya",
    "episode_ini_adalah_readmisi",
    "edukasi_discharge",
    "jadwal_kontrol",
    "kehadiran_kontrol",
]

tab1, tab2, tab3, tab4, tab_riwayat = st.tabs([
    "Simulasi Satuan",
    "Prediksi Massal",
    "Evaluasi Model",
    "Konfigurasi",
    "Riwayat Prediksi",
])

# ==============================================================================
# TAB 1: SIMULASI PASIEN SATUAN
# ==============================================================================
with tab1:
    input_data = {}
    with st.form("form_prediksi_lengkap"):
        kode_pasien_input = st.text_input(
            "Kode / ID Pasien (Anonim - Opsional)",
            placeholder="Contoh: PSN-002 (kosongkan untuk nomor urut otomatis)",
        )

        st.markdown(
            '<div class="cat-header">1. Demografi & Data Administratif Rawat'
            " Inap </div>",
            unsafe_allow_html=True,
        )
        cols_1 = st.columns(4)
        idx_c = 0
        for col in GRUP_DEMOGRAFI_ADMIN:
            if col in semua_fitur:
                input_data[col] = render_input_widget(col, cols_1[idx_c % 4])
                idx_c += 1

        st.markdown(
            '<div class="cat-header">2. Data Klinis, Diagnosis ICD-10/9 &'
            " Tanda Vital </div>",
            unsafe_allow_html=True,
        )
        cols_2 = st.columns(4)
        idx_c = 0
        for col in GRUP_KLINIS_VITAL:
            if col in semua_fitur:
                input_data[col] = render_input_widget(col, cols_2[idx_c % 4])
                idx_c += 1

        ada_lab = [c for c in GRUP_LAB if c in semua_fitur]
        if ada_lab:
            st.markdown(
                '<div class="cat-header">3. Hasil Laboratorium Utama Saat'
                " Masuk </div>",
                unsafe_allow_html=True,
            )
            cols_3 = st.columns(5)
            for idx_l, col in enumerate(ada_lab):
                input_data[col] = render_input_widget(col, cols_3[idx_l % 5])

        st.markdown(
            '<div class="cat-header">4. Riwayat Kunjungan & Proses Pemulangan'
            " / Discharge Planning </div>",
            unsafe_allow_html=True,
        )
        cols_4 = st.columns(4)
        idx_c = 0
        for col in GRUP_RIWAYAT_DISCHARGE:
            if col in semua_fitur:
                input_data[col] = render_input_widget(col, cols_4[idx_c % 4])
                idx_c += 1

        semua_grup = set(
            GRUP_DEMOGRAFI_ADMIN
            + GRUP_KLINIS_VITAL
            + GRUP_LAB
            + GRUP_RIWAYAT_DISCHARGE
        )
        sisa_fitur = [c for c in semua_fitur if c not in semua_grup]
        if sisa_fitur:
            st.markdown(
                '<div class="cat-header">5. Variabel Tambahan Lainnya</div>',
                unsafe_allow_html=True,
            )
            cols_5 = st.columns(4)
            for idx_s, col in enumerate(sisa_fitur):
                input_data[col] = render_input_widget(col, cols_5[idx_s % 4])

        st.write("")
        submit = st.form_submit_button(
            "🔍 Hitung Prediksi Risiko Readmisi & Lama Rawat Inap",
            use_container_width=True,
        )

    if submit:
        df_input = pd.DataFrame([input_data])[semua_fitur]
        proba = float(model.predict_proba(df_input)[0][1])
        pred = int(proba >= threshold)

        # Simpan otomatis ke MySQL XAMPP dengan nomor urut PSN-001, PSN-002, dst.
        try:
            kode_final = (
                kode_pasien_input.strip()
                if kode_pasien_input.strip()
                else buat_kode_pasien_baru()
            )
            label_hasil = "Risiko Tinggi" if pred == 1 else "Risiko Rendah"
            simpan_prediksi(
                kode_pasien=kode_final,
                algoritma=nama_model_aktif,
                probabilitas=proba,
                hasil=label_hasil,
                fitur_dict=bersihkan_fitur_json(input_data),
            )
            st.success(
                f"✅ Data prediksi pasien **{kode_final}** berhasil disimpan ke"
                " database MySQL (XAMPP)."
            )
        except Exception as e:
            st.error(f"❌ Gagal menyimpan ke database MySQL XAMPP: {e}")

        st.divider()
        st.subheader("Hasil Analisis Risiko Readmisi")
        r1, r2 = st.columns([1, 1.6], gap="large")

        with r1:
            st.metric(
                label="Probabilitas Readmisi (≤ 30 Hari)",
                value=f"{proba:.1%}",
                delta=f"{(proba - threshold):+.1%} vs Threshold ({threshold:.0%})",
                delta_color="inverse",
            )
            st.progress(min(max(proba, 0.0), 1.0))

        with r2:
            if pred == 1:
                st.markdown(
                    f"""
                <div class="result-box-high">
                    <h3 style="color:#FCA5A5; margin-top:0;">⚠️ BERISIKO TINGGI READMISI</h3>
                    <p style="color:#F8FAFC; margin-bottom:0;">
                        Probabilitas readmisi pasien sebesar <b>{proba:.1%}</b> (melampaui ambang batas <b>{threshold:.0%}</b>).
                        Rekomendasi: Lakukan edukasi <i>discharge</i> terstruktur dan pemantauan jadwal kontrol lebih ketat.
                    </p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div class="result-box-low">
                    <h3 style="color:#6EE7B7; margin-top:0;">✅ RISIKO RENDAH READMISI</h3>
                    <p style="color:#F8FAFC; margin-bottom:0;">
                        Probabilitas readmisi pasien sebesar <b>{proba:.1%}</b> (di bawah ambang batas <b>{threshold:.0%}</b>).
                        Rekomendasi: Pasien dapat dipulangkan sesuai prosedur standar dengan jadwal kontrol rutin.
                    </p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # ---------------- Estimasi Lama Rawat Inap (LOS) ----------------
        st.divider()
        st.subheader("Estimasi Lama Rawat Inap (LOS)")
        if los_ada:
            los_est = float(prediksi_los(pd.DataFrame([input_data]))[0])
            los_aktual = input_data.get("los_hari")
            mae_los = los_bundle["metrik"]["MAE"]

            l1, l2, l3 = st.columns(3)
            l1.metric("Estimasi LOS (dari data saat masuk)", f"{los_est:.1f} hari")
            if los_aktual is not None:
                l2.metric(
                    "LOS yang Diinput",
                    f"{float(los_aktual):.0f} hari",
                    delta=f"{float(los_aktual) - los_est:+.1f} hari vs estimasi",
                    delta_color="off",
                )
            if "los_hari" in df_ref.columns:
                persen = float((df_ref["los_hari"].dropna() < los_est).mean())
                l3.metric(
                    "Posisi di Dataset", f"Lebih lama dari {persen:.0%} pasien"
                )

            st.caption(
                f"Perkiraan rentang: {max(los_est - mae_los, 0):.1f} –"
                f" {los_est + mae_los:.1f} hari (berdasarkan rata-rata galat"
                f" model ±{mae_los:.1f} hari). Estimasi hanya memakai data"
                " yang tersedia saat masuk, bukan data saat pulang."
            )
        else:
            st.info(
                "Model LOS belum tersedia. Jalankan `python train_los.py`"
                " terlebih dahulu."
            )

# ==============================================================================
# TAB 2: UPLOAD FILE & PREDIKSI MASSAL (BATCH)
# ==============================================================================
with tab2:
    st.subheader("Upload Data Pasien dari SIMRS (CSV / Excel)")
    st.markdown(
        "Unggah file data pasien untuk memprediksi risiko readmisi dan estimasi"
        " lama rawat ratusan/ribuan episode secara otomatis tanpa input satu per"
        " satu."
    )

    col_tmpl1, col_tmpl2 = st.columns([2, 1])
    with col_tmpl1:
        st.info(
            "Pastikan file memiliki kolom fitur yang digunakan model"
            f" ({len(semua_fitur)} kolom). Jika ada kolom yang kosong, sistem"
            " akan mengisinya otomatis menggunakan imputasi model."
        )
    with col_tmpl2:
        contoh_df = df_ref[semua_fitur].head(10)
        csv_template = contoh_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Template CSV (10 Contoh)",
            data=csv_template,
            file_name="template_prediksi_readmisi.csv",
            mime="text/csv",
            use_container_width=True,
        )

    uploaded_file = st.file_uploader(
        "Pilih file CSV atau Excel (.xlsx)", type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)

            st.write(f"**Pratinjau Data Diunggah ({len(df_upload):,} baris):**")
            st.dataframe(df_upload.head(5), use_container_width=True)

            df_pred_input = df_upload.copy()
            for col in semua_fitur:
                if col not in df_pred_input.columns:
                    df_pred_input[col] = pd.NA

            if st.button(
                "🚀 Jalankan Prediksi Massal Sekarang",
                type="primary",
                use_container_width=True,
            ):
                probas = model.predict_proba(df_pred_input[semua_fitur])[:, 1]
                preds = (probas >= threshold).astype(int)

                df_hasil_upload = df_upload.copy()
                df_hasil_upload["Probabilitas_Readmisi (%)"] = (
                    probas * 100
                ).round(2)
                df_hasil_upload["Status_Risiko"] = [
                    "Risiko Tinggi" if p == 1 else "Risiko Rendah"
                    for p in preds
                ]

                if los_ada:
                    df_hasil_upload["Estimasi_LOS_Hari"] = np.round(
                        prediksi_los(df_pred_input), 1
                    )

                # Simpan hasil batch ke MySQL XAMPP
                try:
                    waktu_batch = datetime.now().strftime("%Y%m%d-%H%M")
                    for idx_row, row_data in df_pred_input.iterrows():
                        kode_row = (
                            str(df_upload.loc[idx_row, "id_episode"])
                            if "id_episode" in df_upload.columns
                            else f"BATCH-{waktu_batch}-{idx_row+1}"
                        )
                        simpan_prediksi(
                            kode_pasien=kode_row,
                            algoritma=nama_model_aktif,
                            probabilitas=float(probas[idx_row]),
                            hasil=df_hasil_upload.loc[idx_row, "Status_Risiko"],
                            fitur_dict=bersihkan_fitur_json(
                                row_data[semua_fitur].to_dict()
                            ),
                        )
                    st.success(
                        f"✅ Sebanyak {len(df_hasil_upload):,} data prediksi"
                        " massal berhasil disimpan ke database MySQL (XAMPP)."
                    )
                except Exception as e:
                    st.warning(
                        "⚠️ Prediksi selesai, namun gagal menyimpan ke MySQL"
                        f" XAMPP: {e}"
                    )

                st.divider()
                m1, m2, m3 = st.columns(3)
                jml_tinggi = int(preds.sum())
                jml_rendah = len(preds) - jml_tinggi
                m1.metric("Total Episode Diprediksi", f"{len(preds):,}")
                m2.metric(
                    "Berisiko Tinggi Readmisi",
                    f"{jml_tinggi:,} ({jml_tinggi/len(preds):.1%})",
                )
                m3.metric(
                    "Risiko Rendah Readmisi",
                    f"{jml_rendah:,} ({jml_rendah/len(preds):.1%})",
                )

                if los_ada:
                    st.metric(
                        "Rata-rata Estimasi LOS",
                        f"{df_hasil_upload['Estimasi_LOS_Hari'].mean():.1f} hari",
                    )

                st.dataframe(df_hasil_upload, use_container_width=True)

                d1, d2 = st.columns(2)
                out_csv = df_hasil_upload.to_csv(index=False).encode("utf-8")
                d1.download_button(
                    label="💾 Download Hasil Prediksi (.CSV)",
                    data=out_csv,
                    file_name="hasil_prediksi_readmisi.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                buffer_xlsx = io.BytesIO()
                with pd.ExcelWriter(buffer_xlsx, engine="openpyxl") as writer:
                    df_hasil_upload.to_excel(
                        writer, index=False, sheet_name="Hasil Prediksi"
                    )
                d2.download_button(
                    label="📊 Download Hasil Prediksi (.XLSX)",
                    data=buffer_xlsx.getvalue(),
                    file_name="hasil_prediksi_readmisi.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Gagal memproses file: {e}")

# ==============================================================================
# TAB 3: EVALUASI MODEL
# ==============================================================================
with tab3:
    st.subheader("Evaluasi Kinerja Algoritma Machine Learning")
    if os.path.exists(eval_path):
        df_eval = pd.read_csv(eval_path)
        st.dataframe(
            df_eval.style.highlight_max(
                subset=[
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1-Score",
                    "AUC-ROC",
                    "PR-AUC",
                ],
                color="#1E3A8A",
            ),
            use_container_width=True,
        )
    st.divider()
    g1, g2 = st.columns(2, gap="large")
    fi_path = os.path.join(config.FOLDER_OUTPUT, "feature_importance.png")
    roc_path = os.path.join(config.FOLDER_OUTPUT, "roc_curve_comparison.png")
    if os.path.exists(fi_path):
        g1.markdown(
            "**Faktor Paling Berpengaruh (Permutation Importance)**"
        )
        g1.image(fi_path, use_container_width=True)
    if os.path.exists(roc_path):
        g2.markdown("**Perbandingan Kurva ROC**")
        g2.image(roc_path, use_container_width=True)

    st.divider()
    st.subheader("Evaluasi Model Estimasi Lama Rawat Inap (LOS)")
    if los_ada:
        m = los_bundle["metrik"]
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("MAE", f"{m['MAE']:.2f} hari")
        e2.metric("RMSE", f"{m['RMSE']:.2f} hari")
        e3.metric("R²", f"{m['R2']:.2f}")
        e4.metric("MAE Baseline (median)", f"{m['MAE_baseline_median']:.2f} hari")
        st.caption(
            f"Data uji: {m['n_test']:,} episode (latih: {m['n_train']:,})."
            " Model dianggap bermanfaat jika MAE lebih kecil daripada MAE"
            f" baseline. Fitur yang dipakai ({len(los_bundle['fitur'])}):"
            " hanya data yang tersedia saat masuk."
        )
    else:
        st.info("Model LOS belum dilatih. Jalankan `python train_los.py`.")

# ==============================================================================
# TAB 4: STRUKTUR 27 VARIABEL RS
# ==============================================================================
with tab4:
    st.subheader("Status Ketersediaan 27 Variabel Permintaan Data RS di Model")
    daftar_cek = []
    for k_std, k_rs in config.PEMETAAN_KOLOM.items():
        status = (
            "✅ Aktif di Model"
            if k_std in semua_fitur
            else (
                "🔑 Kolom Kunci / Target"
                if k_std in config.KOLOM_WAJIB + ["id_episode"]
                else "⏳ Menunggu Data RS"
            )
        )
        daftar_cek.append({
            "Kolom Sistem": k_std,
            "Nama Kolom di Formulir RS": k_rs,
            "Status Penggunaan": status,
        })
    st.dataframe(pd.DataFrame(daftar_cek), use_container_width=True)
# ==============================================================================
# TAB 5: RIWAYAT PREDIKSI (DATABASE MYSQL XAMPP - LIVE)
# ==============================================================================
with tab_riwayat:
    st.subheader("Log Riwayat Prediksi")

    @st.fragment(run_every="3s")
    def tampilkan_riwayat_live():
        try:
            df_riwayat = ambil_riwayat()
            st.dataframe(df_riwayat, use_container_width=True)
            st.caption(
                "Sinkronisasi otomatis dengan MySQL XAMPP setiap 3 detik •"
                f" Total data: {len(df_riwayat)} baris"
            )
        except Exception as e:
            st.error(
                "Gagal terhubung ke MySQL XAMPP. Pastikan Apache & MySQL aktif."
                f" Error: {e}"
            )

    tampilkan_riwayat_live()
# --- EDITOR DATABASE TERSEMBUNYI (HANYA MUNCUL JIKA URL MEMAKAI ?admin=1) ---
if st.query_params.get("admin") == "1":
    st.divider()
    st.subheader("Mode Editor Database (Admin)")
    st.caption("Klik dua kali pada sel untuk mengedit teks/angka, atau pilih baris di sisi kiri lalu tekan Delete untuk menghapus. Otomatis sinkron setiap 3 detik.")

    @st.fragment(run_every="3s")
    def panel_editor_admin():
        df_admin = ambil_semua_raw()
        if df_admin.empty:
            st.info("Belum ada data di database.")
            return

        # Key dinamis berdasarkan jumlah & ID maksimal agar otomatis refresh saat ada data baru masuk
        max_id = int(df_admin["id_prediksi"].max())
        editor_key = f"editor_db_admin_{len(df_admin)}_{max_id}"

        df_edited = st.data_editor(
            df_admin,
            num_rows="dynamic",
            use_container_width=True,
            disabled=["id_prediksi"],
            key=editor_key,
        )

        # Abaikan baris kosong tambahan di paling bawah jika belum diisi
        df_edited_clean = df_edited.dropna(subset=["id_prediksi"])

        if not df_edited_clean.equals(df_admin):
            simpan_perubahan_admin(df_admin, df_edited_clean)
            st.toast("Perubahan database tersimpan!", icon="✅")
            st.rerun()

    panel_editor_admin()
            st.rerun()
