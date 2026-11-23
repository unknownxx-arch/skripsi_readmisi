"""
Melatih model prediksi Lama Rawat Inap (LOS) dan menyimpannya ke
<FOLDER_OUTPUT>/model_los.joblib

Letakkan file ini sejajar dengan app.py, lalu jalankan:
    python train_los.py
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import config

TARGET = "los_hari"

# Variabel yang baru diketahui SAAT/SETELAH pulang -> dikeluarkan agar tidak terjadi data leakage
FITUR_DIKELUARKAN = {
    "los_hari",
    "status_pulang",
    "kondisi_saat_pulang",
    "edukasi_discharge",
    "jadwal_kontrol",
    "kehadiran_kontrol",
}

data_path = os.path.join(config.FOLDER_PROCESSED, "dataset_model.csv")
out_path = os.path.join(config.FOLDER_OUTPUT, "model_los.joblib")

df = pd.read_csv(data_path)
if TARGET not in df.columns:
    raise SystemExit(f"Kolom target '{TARGET}' tidak ada di {data_path}")

df = df.dropna(subset=[TARGET])
df = df[df[TARGET] >= 0]

fitur_num = [c for c in config.FITUR_NUMERIK if c in df.columns and c not in FITUR_DIKELUARKAN]
fitur_cat = [c for c in config.FITUR_KATEGORIKAL if c in df.columns and c not in FITUR_DIKELUARKAN]
semua_fitur = fitur_num + fitur_cat

X = df[semua_fitur].copy()
for c in fitur_num:
    X[c] = pd.to_numeric(X[c], errors="coerce")
for c in fitur_cat:
    X[c] = X[c].astype("object").where(X[c].notna(), np.nan)

y = np.log1p(df[TARGET])  # LOS cenderung miring ke kanan

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), fitur_num),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]), fitur_cat),
])

model = Pipeline([
    ("pre", pre),
    ("gbr", GradientBoostingRegressor(random_state=42)),
])
model.fit(X_tr, y_tr)

pred = np.expm1(model.predict(X_te))
true = np.expm1(y_te)
baseline = np.full_like(true, np.median(np.expm1(y_tr)), dtype=float)

metrik = {
    "MAE": float(mean_absolute_error(true, pred)),
    "RMSE": float(mean_squared_error(true, pred) ** 0.5),
    "R2": float(r2_score(true, pred)),
    "MAE_baseline_median": float(mean_absolute_error(true, baseline)),
    "n_train": int(len(X_tr)),
    "n_test": int(len(X_te)),
}

print(f"Fitur dipakai ({len(semua_fitur)}): {semua_fitur}")
for k, v in metrik.items():
    print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")

os.makedirs(config.FOLDER_OUTPUT, exist_ok=True)
joblib.dump(
    {
        "model": model,
        "fitur": semua_fitur,
        "fitur_num": fitur_num,
        "fitur_cat": fitur_cat,
        "metrik": metrik,
    },
    out_path,
)
print(f"Model LOS disimpan di: {out_path}")