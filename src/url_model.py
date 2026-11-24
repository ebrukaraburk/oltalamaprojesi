import sys
import joblib
import pandas as pd
sys.path.insert(0, ".")
from src.features import url_features

_bundle = joblib.load("models/url_model.joblib")


def url_risk(url):
    """Linkin oltalama olma olasılığını (0-1) döndürür."""
    X = pd.DataFrame([url_features(url)])[_bundle["features"]]
    return float(_bundle["model"].predict_proba(X)[0, 1])


if __name__ == "__main__":
    tests = [
        "https://www.google.com",
        "https://www.ziraatbank.com.tr",
        "https://www.turkiye.gov.tr",
        "http://ziraat-giris.xyz/dogrula",
        "http://192.168.1.5/login/verify.php?id=123",
        "https://bit.ly/3xYzAbc",
        "http://garanti-hesap-guncelle.top/giris",
    ]
    for t in tests:
        print(f"{url_risk(t):.2f}  {t}")