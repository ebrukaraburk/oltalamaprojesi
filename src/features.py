import re
import math
from urllib.parse import urlparse
import tldextract

# ağ isteği yapmaz, pakette gelen alan adı listesini kullanır
_ext = tldextract.TLDExtract(suffix_list_urls=())

SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "ow.ly", "cutt.ly",
              "rebrand.ly", "shorturl.at", "tiny.cc", "buff.ly", "rb.gy", "t.ly"}

BRANDS = ["ptt", "ziraat", "garanti", "isbank", "akbank", "yapikredi", "vakifbank",
          "halkbank", "denizbank", "qnbfinansbank", "turkcell", "vodafone", "turktelekom",
          "trendyol", "hepsiburada", "yurtici", "edevlet", "e-devlet",
          "paypal", "microsoft", "apple", "google", "amazon", "netflix", "facebook",
          "instagram", "dhl", "fedex"]

KEYWORDS = ["login", "signin", "verify", "secure", "update", "account", "banking",
            "confirm", "password", "hesap", "giris", "dogrula", "guncelle", "odeme",
            "sifre", "fatura", "kargo", "bonus", "odul"]

IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _entropy(s):
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def url_features(url):
    url = str(url).strip()
    u = url if "://" in url else "http://" + url
    try:
        p = urlparse(u)
        host = (p.hostname or "").lower()
        port = p.port
    except ValueError:
        p, host, port = urlparse("http://invalid"), "", None

    # "www." öneki veri setinde meşru sitelerle fazla ilişkili; modelin bu
    # kısayolu öğrenmemesi için alan adından siliyoruz
    if host.startswith("www."):
        host = host[4:]

    e = _ext(host)
    reg = e.domain or ""                 # kayıtlı alan adı gövdesi (örn. "google", "garanti-hesap-guncelle")
    low = url.lower()
    brand_hits = [b for b in BRANDS if b in low]

    # Marka adı URL'de geçiyor ama kayıtlı alan adının tamamı o marka değilse taklit şüphesi.
    # google.com -> reg == "google" -> taklit değil
    # garanti-hesap-guncelle.top -> reg != "garanti" -> taklit
    # paypal.com.evil.top -> reg == "evil" -> taklit
    brand_impersonation = int(bool(brand_hits) and reg not in BRANDS)

    host_digits = sum(c.isdigit() for c in host)

    return {
        # alan adı (host) özellikleri
        "host_length": len(host),
        "num_dots": host.count("."),
        "num_hyphens": host.count("-"),
        "num_underscores": host.count("_"),
        "num_digits": host_digits,
        "digit_ratio": host_digits / max(len(host), 1),
        "host_entropy": _entropy(host),
        "subdomain_count": len(e.subdomain.split(".")) if e.subdomain else 0,
        "tld_length": len(e.suffix),
        "is_ip": int(bool(IP_RE.match(host))),
        "has_punycode": int("xn--" in host),
        "has_port": int(port is not None),
        "is_shortener": int((e.domain + "." + e.suffix) in SHORTENERS),
        # tüm URL üzerinden anlamlı olanlar
        "num_at": url.count("@"),
        "has_https": int(p.scheme == "https"),
        "keyword_count": sum(k in low for k in KEYWORDS),
        "brand_impersonation": brand_impersonation,
    }