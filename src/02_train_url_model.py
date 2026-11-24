import os, sys, time
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
sys.path.insert(0, ".")
from config import SEED
from src.features import url_features

df = pd.read_csv("data/processed/urls.csv")
print("URL sayısı:", len(df))

t0 = time.time()
X = pd.DataFrame([url_features(u) for u in df["url"]])
print(f"Özellik çıkarımı bitti: {time.time() - t0:.0f} sn, {X.shape[1]} özellik")

y = df["label"].values
tr = (df["split"] == "train").values
te = (df["split"] == "test").values

model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED,
                               class_weight="balanced")
model.fit(X[tr], y[tr])

pred = model.predict(X[te])
tn, fp, fn, tp = confusion_matrix(y[te], pred).ravel()
print("\n=== URL MODELİ - TEST SONUCU ===")
print(f"Doğruluk    : {accuracy_score(y[te], pred):.4f}")
print(f"Kesinlik    : {precision_score(y[te], pred):.4f}")
print(f"Geri çağırma: {recall_score(y[te], pred):.4f}")
print(f"F1          : {f1_score(y[te], pred):.4f}")
print(f"Yanlış alarm oranı: {fp / (fp + tn):.4f}  (temiz URL'nin oltalama sanılması)")

imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nEn önemli 8 özellik:\n", imp.head(8).round(3))

os.makedirs("models", exist_ok=True)
joblib.dump({"model": model, "features": list(X.columns)}, "models/url_model.joblib")
print("\nKaydedildi: models/url_model.joblib")