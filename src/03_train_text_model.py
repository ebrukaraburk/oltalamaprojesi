import os, sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
sys.path.insert(0, ".")
from config import SEED

df = pd.read_csv("data/processed/texts.csv")
df = df[df["lang"] == "tr"].reset_index(drop=True)
tr = df[df["split"] == "train"]
te = df[df["split"] == "test"]
print("Eğitim:", len(tr), " Test:", len(te))

# kontrol: linksiz metinde link kalıntısı var mı?
kalinti = df["text_nourl"].str.contains(r"http|www\.|\.com|\.tr", case=False, regex=True).mean()
print(f"Linksiz metinde link kalıntısı oranı: {kalinti:.3f}")
print("Örnek linksiz metin:", df["text_nourl"].iloc[0][:200])

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True)
Xtr = vec.fit_transform(tr["text_nourl"])
Xte = vec.transform(te["text_nourl"])

clf = LogisticRegression(C=3, max_iter=2000, class_weight="balanced", random_state=SEED)
clf.fit(Xtr, tr["label"])

proba = clf.predict_proba(Xte)[:, 1]
pred = pd.Series((proba >= 0.5).astype(int), index=te.index)
y = te["label"]

tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
print("\n=== METİN MODELİ (linksiz metin) - TEST SONUCU ===")
print(f"Doğruluk    : {accuracy_score(y, pred):.4f}")
print(f"Kesinlik    : {precision_score(y, pred):.4f}")
print(f"Geri çağırma: {recall_score(y, pred):.4f}")
print(f"F1          : {f1_score(y, pred):.4f}")
print(f"Yanlış alarm oranı: {fp / (fp + tn):.4f}")

print("\nSektör bazında doğruluk ve yanlış alarm:")
for s, g in te.groupby("sector"):
    p = pred[g.index]
    neg = (g["label"] == 0).sum()
    fa = ((p == 1) & (g["label"] == 0)).sum() / max(neg, 1)
    print(f"  {s:<20} doğruluk={accuracy_score(g['label'], p):.3f}  yanlış alarm={fa:.3f}  (n={len(g)})")

names = vec.get_feature_names_out()
order = clf.coef_[0].argsort()
print("\nOltalamayı en çok artıran 15 ifade:", [names[i] for i in order[-15:][::-1]])
print("Meşruya en çok işaret eden 15 ifade:", [names[i] for i in order[:15]])

os.makedirs("models", exist_ok=True)
joblib.dump({"vectorizer": vec, "model": clf}, "models/text_model.joblib")
print("\nKaydedildi: models/text_model.joblib")