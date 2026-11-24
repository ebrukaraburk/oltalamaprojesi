import sys
from urllib.parse import urlparse
import joblib
import pandas as pd
import tldextract
sys.path.insert(0, ".")
from src.features import url_features

_bundle = joblib.load("models/url_model.joblib")

_bundle["model"].n_jobs = 1      # tek tek tahminde paralel işçi açma: hızlı ve uyarısız
_ext = tldextract.TLDExtract(suffix_list_urls=())   # ağ isteği yapmaz

# ---- Güvenilir alan adı katmanı ----
# Model alan adının itibarını bilmez, sadece biçimine bakar. Bu liste, bilinen kurumların
# resmi alan adlarında model puanını sınırlar. Elle hazırlanmış küçük bir listedir;
# kullanmadan önce kendin de kontrol et, gerekirse genişlet.
TRUSTED_DOMAINS = {
    # kamu
    "turkiye.gov.tr", "ptt.gov.tr", "gib.gov.tr", "sgk.gov.tr",
    # bankalar
    "ziraatbank.com.tr", "garantibbva.com.tr", "isbank.com.tr", "akbank.com",
    "yapikredi.com.tr", "vakifbank.com.tr", "halkbank.com.tr", "denizbank.com",
    "qnb.com.tr",
    # operatör, alışveriş, kargo
    "turkcell.com.tr", "vodafone.com.tr", "turktelekom.com.tr",
    "trendyol.com", "hepsiburada.com", "yurticikargo.com", "dhl.com", "fedex.com",
    # genel
    "google.com", "microsoft.com", "apple.com", "amazon.com", "amazon.com.tr",
    "paypal.com", "netflix.com", "facebook.com", "instagram.com", "linkedin.com",
    "youtube.com", "whatsapp.com", "github.com", "wikipedia.org",
    "claude.ai", "anthropic.com",
}
# Bu uzantılar sadece resmi kurumlara verilir, hepsini güvenilir sayıyoruz
TRUSTED_SUFFIXES = {"gov.tr", "edu.tr"}

# Güvenilir alan adının altında kullanıcıların içerik açabildiği adresler:
# oltalama sayfaları burada barınabilir, bunlara güvenme
UNTRUSTED_HOSTS = {
    "sites.google.com", "docs.google.com", "forms.google.com", "drive.google.com",
    "storage.googleapis.com", "forms.gle", "script.google.com",
}

TRUSTED_MAX = 0.10      # güvenilir alan adında model puanı en fazla bu kadar olabilir


def _parts(url):
    u = url if "://" in url else "http://" + url
    try:
        host = (urlparse(u).hostname or "").lower()
    except ValueError:
        return "", "", ""
    e = _ext(host)
    reg = (e.domain + "." + e.suffix) if e.domain and e.suffix else ""
    return host, reg, e.suffix


def is_trusted(url):
    host, reg, suffix = _parts(str(url).strip())
    if not host or host in UNTRUSTED_HOSTS:
        return False
    return reg in TRUSTED_DOMAINS or (suffix in TRUSTED_SUFFIXES and reg != "")


def url_risk_raw(url):
    """Sadece modelin verdiği ham olasılık (0-1)."""
    X = pd.DataFrame([url_features(url)])[_bundle["features"]]
    return float(_bundle["model"].predict_proba(X)[0, 1])


def url_risk(url):
    """Linkin oltalama olma olasılığı (0-1). Güvenilir alan adında model puanı sınırlanır."""
    p = url_risk_raw(url)
    if is_trusted(url):
        p = min(p, TRUSTED_MAX)
    return p


if __name__ == "__main__":
    tests = [
        "https://claude.ai/chat/b39875fb-2a98-4254-8e9f-0bc043d132cb",
        "https://www.google.com/search?q=e+devlet+kariyer+kapisi",
        "https://www.ziraatbank.com.tr",
        "https://www.turkiye.gov.tr",
        "https://sites.google.com/view/banka-giris",
        "http://ziraat-giris.xyz/dogrula",
        "http://192.168.1.5/login/verify.php?id=123",
        "https://bit.ly/3xYzAbc",
        "http://garanti-hesap-guncelle.top/giris",
    ]
    print("nihai  ham    güvenilir  adres")
    for t in tests:
        print(f"{url_risk(t):.2f}   {url_risk_raw(t):.2f}   {str(is_trusted(t)):9}  {t}")