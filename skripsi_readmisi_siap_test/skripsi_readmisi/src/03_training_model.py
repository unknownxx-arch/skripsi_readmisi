# -*- coding: utf-8 -*-
"""
03_training_model.py
=====================
Melatih dan membandingkan model prediksi readmisi 30 hari.

Model     : Logistic Regression, Random Forest, Gradient Boosting
            (+ XGBoost otomatis ikut jika sudah di-install)
Evaluasi  : Accuracy, Precision, Recall, F1, AUC-ROC, PR-AUC
Interpretasi : Permutation Importance (berlaku untuk semua model)

CATATAN METODOLOGI (bisa ditulis di Bab 3):
  * Pembagian data latih/uji dilakukan PER PASIEN (GroupShuffleSplit), supaya
    pasien yang sama tidak muncul di data latih sekaligus data uji
    (menghindari kebocoran data / data leakage).
  * Data tidak seimbang ditangani dengan pembobotan kelas (class weight).
  * Validasi silang 5-fold (GroupKFold) pada data latih untuk menilai
    kestabilan model.

Hasil disimpan di folder output/.
Cara jalankan:   python src/03_training_model.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # agar grafik tersimpan tanpa perlu jendela tampilan
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)

import config

OUT = config.FOLDER_OUTPUT
os.makedirs(OUT, exist_ok=True)

# ==================================================================
# 1. MEMUAT DATASET
# ==================================================================
path = os.path.join(config.FOLDER_PROCESSED, "dataset_model.csv")
if not os.path.exists(path):
    print("[ERROR] dataset_model.csv belum ada. Jalankan dulu:  python src/02_buat_label.py")
    sys.exit(1)

df = pd.read_csv(path)
target = "readmisi_30_hari"

fitur_num = [c for c in config.FITUR_NUMERIK if c in df.columns]
fitur_cat = [c for c in config.FITUR_KATEGORIKAL if c in df.columns]
X = df[fitur_num + fitur_cat]
y = df[target]
grup = df["id_pasien"]

print("=" * 70)
print("1. DATASET")
print("=" * 70)
print(f"Jumlah episode : {len(df)}")
print(f"Jumlah pasien  : {grup.nunique()}")
print(f"Fitur numerik  : {len(fitur_num)} | kategorikal: {len(fitur_cat)}")
print(f"Proporsi readmisi: {y.mean():.1%}")

# ==================================================================
# 2. SPLIT DATA PER PASIEN (80:20)
# ==================================================================
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_SEED)
idx_train, idx_test = next(gss.split(X, y, groups=grup))
X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
g_train = grup.iloc[idx_train]
print(f"\nData latih: {len(X_train)} episode | Data uji: {len(X_test)} episode")
print(f"Readmisi di data latih: {y_train.mean():.1%} | di data uji: {y_test.mean():.1%}")

# ==================================================================
# 3. PREPROCESSING
# ==================================================================
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), fitur_num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore"))]), fitur_cat),
])

# ==================================================================
# 4. DEFINISI MODEL
# ==================================================================
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=config.RANDOM_SEED),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5, class_weight="balanced",
        random_state=config.RANDOM_SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        random_state=config.RANDOM_SEED),
}

try:  # XGBoost opsional
    from xgboost import XGBClassifier
    rasio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    models["XGBoost"] = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss",
        scale_pos_weight=rasio, random_state=config.RANDOM_SEED)
    print("\n[INFO] XGBoost terdeteksi dan ikut dibandingkan.")
except ImportError:
    print("\n[INFO] XGBoost belum terpasang -> dilewati. (Aktifkan: pip install xgboost)")

# ==================================================================
# 5. TRAINING & EVALUASI
# ==================================================================
print("\n" + "=" * 70)
print("2. TRAINING & EVALUASI")
print("=" * 70)

hasil, roc_data, pipes = [], {}, {}
gkf = GroupKFold(n_splits=5)

for nama, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", model)])

    # Gradient Boosting/XGBoost: imbang kelas lewat sample_weight
    fit_params = {}
    if nama == "Gradient Boosting":
        fit_params["clf__sample_weight"] = compute_sample_weight("balanced", y_train)

    pipe.fit(X_train, y_train, **fit_params)
    pipes[nama] = pipe

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # Validasi silang 5-fold per pasien (AUC tidak bergantung pada ambang prediksi)
    cv_auc = cross_val_score(
        Pipeline([("prep", preprocessor), ("clf", model)]),
        X_train, y_train, groups=g_train, cv=gkf, scoring="roc_auc")

    hasil.append({
        "Model": nama,
        "Accuracy": round(accuracy_score(y_test, pred), 3),
        "Precision": round(precision_score(y_test, pred, zero_division=0), 3),
        "Recall": round(recall_score(y_test, pred, zero_division=0), 3),
        "F1-Score": round(f1_score(y_test, pred, zero_division=0), 3),
        "AUC-ROC": round(roc_auc_score(y_test, proba), 3),
        "PR-AUC": round(average_precision_score(y_test, proba), 3),
        "CV AUC (mean)": round(cv_auc.mean(), 3),
        "CV AUC (std)": round(cv_auc.std(), 3),
    })
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_data[nama] = (fpr, tpr, roc_auc_score(y_test, proba))

    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Tidak Readmisi", "Readmisi"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {nama}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"confusion_matrix_{nama.replace(' ', '_')}.png"), dpi=150)
    plt.close()
    print(f"  selesai: {nama}")

df_hasil = pd.DataFrame(hasil).sort_values("AUC-ROC", ascending=False)
df_hasil.to_csv(os.path.join(OUT, "hasil_evaluasi_model.csv"), index=False)
print("\n" + df_hasil.to_string(index=False))

terbaik = df_hasil.iloc[0]["Model"]
print(f"\nModel terbaik (AUC-ROC data uji): {terbaik}")

# ==================================================================
# 6. KURVA ROC
# ==================================================================
plt.figure(figsize=(6, 5))
for nama, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{nama} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Tebakan acak")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Kurva ROC - Prediksi Readmisi 30 Hari")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "roc_curve_comparison.png"), dpi=150)
plt.close()

# ==================================================================
# 7. FEATURE IMPORTANCE (Permutation Importance, model terbaik)
# ==================================================================
print("\n" + "=" * 70)
print("3. FAKTOR PALING BERPENGARUH (Permutation Importance)")
print("=" * 70)

best = pipes[terbaik]
pi = permutation_importance(best, X_test, y_test, scoring="roc_auc",
                            n_repeats=10, random_state=config.RANDOM_SEED, n_jobs=1)
df_imp = pd.DataFrame({"Fitur": X_test.columns,
                       "Penurunan AUC jika diacak": pi.importances_mean,
                       "Std": pi.importances_std}
                      ).sort_values("Penurunan AUC jika diacak", ascending=False)
df_imp.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)
print(df_imp.head(10).round(4).to_string(index=False))

top = df_imp.head(10).iloc[::-1]
plt.figure(figsize=(7, 5))
plt.barh(top["Fitur"], top["Penurunan AUC jika diacak"], color="#1F4E78")
plt.xlabel("Penurunan AUC saat fitur diacak (semakin besar = semakin penting)")
plt.title(f"10 Fitur Paling Berpengaruh - {terbaik}")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=150)
plt.close()

# ==================================================================
# 8. SIMPAN MODEL TERBAIK
# ==================================================================
joblib.dump(best, os.path.join(OUT, "model_terbaik.joblib"))
print(f"\nModel terbaik disimpan: output/model_terbaik.joblib")
print("Semua tabel & grafik ada di folder: output/")
print("Selesai.")
