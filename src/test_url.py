import sys
sys.path.insert(0, ".")
import joblib
import pandas as pd
from src.features import url_features

obj = joblib.load("models/url_model.joblib")
model = obj
if isinstance(obj, dict):
    model = next(v for v in obj.values() if hasattr(v, "predict_proba"))

URLS = [
    "https://claude.ai/chat/b39875fb-2a98-4254-8e9f-0bc043d132cb",
    "https://claude.ai",
    "https://www.claude.ai",
    "https://www.google.com",
    "https://github.com/anthropics",
    "http://garanti-hesap-guncelle.top/giris",
]

rows = pd.DataFrame([url_features(u) for u in URLS])
if hasattr(model, "feature_names_in_"):
    rows = rows[list(model.feature_names_in_)]

probs = model.predict_proba(rows)[:, 1]
print("\nOltalama olasılığı:")
for u, p in zip(URLS, probs):
    print(f"{p:.3f}  {u}")

print("\nclaude.ai/chat/... için özellik değerleri:")
print(rows.iloc[0].to_string())