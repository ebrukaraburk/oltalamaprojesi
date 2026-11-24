import re, sys
import joblib
sys.path.insert(0, ".")
from src.url_model import url_risk, is_trusted
from src.features import url_features

# ---- Ağırlıklar (sonra test setinde ayarlanabilir) ----
W_TEXT = 0.5
W_URL = 0.5
LOW_MAX, MID_MAX = 33, 66          # 0-33 düşük, 34-66 orta, 67-100 yüksek
LINK_ONLY_MAX = 0.85               # yalnızca bağlantı varsa tek model kesin hüküm vermesin

_tb = joblib.load("models/text_model.joblib")
_vec, _clf = _tb["vectorizer"], _tb["model"]
_names = _vec.get_feature_names_out()

URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"')]+", re.IGNORECASE)

# modelin sentetik veriden öğrendiği, anlamsız genel kelimeler: işaretlemeden çıkar
STOP = {"bu", "de", "da", "ve", "ile", "için", "bir", "olarak", "tarafından", "birlikte",
        "ortak", "önceki", "birçok", "benzer", "eğitim", "mayıs", "2024",
        "nedeniyle", "içinde", "merhaba", "yapılacaktır"}

SIGNALS = {
    "aciliyet": ["acil", "hemen", "derhal", "son şans", "son gün", "son fırsat",
                 "süre dol", "24 saat", "kısa süre", "bugün içinde"],
    "tehdit": ["askıya", "kapatıl", "bloke", "iptal edil", "engellen", "yasal işlem",
               "hukuki", "cezai", "erişim kısıt", "güvenlik problemi"],
    "ödül vaadi": ["ödül", "kazandınız", "hediye", "çekiliş", "bonus", "kupon", "indirim"],
    "kişisel bilgi talebi": ["şifre", "parola", "kart bilgi", "kimlik", "doğrulama kodu",
                             "giriş bilgi", "cvv", "iban", "bilgilerinizi", "hesap bilgi"],
}


def tr_lower(s):
    return s.replace("İ", "i").replace("I", "ı").lower()


def extract_urls(message):
    return [u.rstrip(".,;:!?") for u in URL_RE.findall(message)]


def strip_urls(message):
    return re.sub(r"\s+", " ", URL_RE.sub(" ", message)).strip()


def text_risk(body):
    """Metnin oltalama olasılığı (0-1)."""
    return float(_clf.predict_proba(_vec.transform([body]))[0, 1])


def highlight_terms(body, top=6):
    """Modelin oltalama kararına en çok katkı veren ifadeler."""
    X = _vec.transform([body])
    if X.nnz == 0:
        return []
    contrib = X.data * _clf.coef_[0][X.indices]
    out = []
    for c, i in sorted(zip(contrib, X.indices), reverse=True):
        if c <= 0:
            break
        term = _names[i]
        if all(w in STOP for w in term.split()):
            continue
        out.append(term)
        if len(out) >= top:
            break
    return out


def detect_signals(body):
    low = tr_lower(body)
    found = {}
    for name, kws in SIGNALS.items():
        hits = [k for k in kws if k in low]
        if hits:
            found[name] = hits
    return found


def url_reasons(url):
    """Bağlantıdaki şüpheli özellikleri sade Türkçe cümlelere çevirir."""
    if is_trusted(url):
        return ["bilinen ve güvenilir bir kurumun resmi alan adı"]
    f = url_features(url)
    r = []
    if f["is_shortener"]:
        r.append("kısaltılmış bağlantı: gerçek adres gizleniyor")
    if f["brand_impersonation"]:
        r.append("adreste banka veya marka adı geçiyor ama adres o kuruma ait değil")
    if f["is_ip"]:
        r.append("alan adı yerine IP adresi kullanılıyor")
    if not f["has_https"]:
        r.append("güvenli bağlantı (https) kullanılmıyor")
    if f["num_hyphens"] >= 2:
        r.append("adreste çok sayıda tire var")
    if f["keyword_count"] >= 2:
        r.append("adreste giriş, doğrulama, güncelleme gibi şüpheli kelimeler var")
    if f["num_at"] > 0:
        r.append("adreste @ işareti var")
    if f["subdomain_count"] >= 3:
        r.append("çok sayıda alt alan adı var")
    if f["has_port"]:
        r.append("alışılmadık bir port numarası var")
    return r


def level_of(score):
    return "düşük" if score <= LOW_MAX else ("orta" if score <= MID_MAX else "yüksek")


def analyze(message):
    urls = extract_urls(message)
    body = strip_urls(message)
    has_text = len(body.split()) >= 4

    t = text_risk(body) if has_text else None
    url_scores = [(u, url_risk(u)) for u in urls]
    u = max((p for _, p in url_scores), default=None)
    link_only = t is None and u is not None

    if t is not None and u is not None:
        p = W_TEXT * t + W_URL * u
    elif t is not None:
        p = t
    elif u is not None:
        p = min(u, LINK_ONLY_MAX)       # tek modele dayanan sonuç 85'i geçmesin
    else:
        return None

    score = round(100 * p)
    level = level_of(score)
    return {
        "score": score,
        "level": level,
        "text_prob": t,
        "url_probs": url_scores,
        "link_only": link_only,
        "trusted_urls": [u_ for u_, _ in url_scores if is_trusted(u_)],
        "highlights": highlight_terms(body) if (has_text and level != "düşük") else [],
        "signals": detect_signals(body) if level != "düşük" else {},
        "url_reasons": {u_: url_reasons(u_) for u_, _ in url_scores},
        "body": body,
    }


if __name__ == "__main__":
    tests = [
        "Sayın müşterimiz, hesabınız güvenlik nedeniyle askıya alınacaktır. Bugün içinde şifrenizi "
        "doğrulamazsanız hesabınız kapatılacaktır. Hemen tıklayın: http://garanti-hesap-guncelle.top/giris",
        "Merhaba, yarınki toplantı saat 14:00'te yapılacaktır. Gündem maddeleri ekte yer almaktadır. "
        "Sorularınız için bizimle iletişime geçebilirsiniz. Saygılarımızla.",
        "https://bit.ly/3xYzAbc",
        "https://claude.ai/chat/b39875fb-2a98-4254-8e9f-0bc043d132cb",
    ]
    for m in tests:
        r = analyze(m)
        print("\nMESAJ:", m[:80], "...")
        print("PUAN:", r["score"], "->", r["level"], "| yalnızca link:", r["link_only"])
        print("metin olasılığı:", r["text_prob"], "| url:", r["url_probs"])
        print("işaretlenen ifadeler:", r["highlights"])
        print("sinyaller:", r["signals"])
        print("bağlantı bulguları:", r["url_reasons"])