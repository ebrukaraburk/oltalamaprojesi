"""PhishShield AI değerlendirme betiği.

Kullanım (proje klasöründen):
    .venv\\Scripts\\python.exe src\\05_evaluate.py
    .venv\\Scripts\\python.exe src\\05_evaluate.py --rag          # RAG: rerank'lı/rerank'sız (yavaş)
    .venv\\Scripts\\python.exe src\\05_evaluate.py --n 800 --esik 50

Çıktı ekrana yazılır ve data/processed/eval_sonuclari.md dosyasına kaydedilir.
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

sys.path.insert(0, ".")
from config import SEED, RERANK_TOP_N
from src.features import url_features
from src.scoring import analyze, extract_urls, strip_urls, text_risk
from src.url_model import _bundle, url_risk_raw, is_trusted, TRUSTED_MAX

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1500, help="uçtan uca testte kullanılacak en fazla e-posta sayısı")
ap.add_argument("--esik", type=int, default=50, help="bu puan ve üstü 'oltalama' sayılır (0-100)")
ap.add_argument("--rag", action="store_true", help="RAG karşılaştırmasını da çalıştır (yavaştır)")
ap.add_argument("--rag-n", type=int, default=100, help="RAG karşılaştırmasında örnek sayısı")
args = ap.parse_args()

OUT = []   # markdown çıktısı


def say(s=""):
    print(s)
    OUT.append(s)


def metrics(y, score):
    """score: 0-100 arası puan. args.esik ve üstü oltalama sayılır."""
    y = np.asarray(y)
    pred = (np.asarray(score) >= args.esik).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": len(y),
        "doğruluk": accuracy_score(y, pred),
        "kesinlik": precision_score(y, pred, zero_division=0),
        "geri çağırma": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "yanlış alarm": fp / max(fp + tn, 1),
    }


def tablo(rows):
    """rows: {ad: metrics sözlüğü} -> markdown tablo"""
    cols = ["n", "doğruluk", "kesinlik", "geri çağırma", "F1", "yanlış alarm"]
    lines = ["| Yöntem | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for ad, m in rows.items():
        vals = [str(m["n"])] + [f"{m[c]:.3f}" for c in cols[1:]]
        lines.append(f"| {ad} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


say(f"# PhishShield AI değerlendirme sonuçları\n\nKarar eşiği: {args.esik}/100 ve üstü oltalama sayılır.\n")

# ------------------------------------------------------------------
# A) URL modeli (PhiUSIIL test bölümü): ham model ve güvenilir alan adı katmanıyla
# ------------------------------------------------------------------
say("## A) URL modeli (PhiUSIIL test bölümü)\n")
t0 = time.time()
du = pd.read_csv("data/processed/urls.csv")
du = du[du["split"] == "test"].reset_index(drop=True)
X = pd.DataFrame([url_features(u) for u in du["url"]])[_bundle["features"]]
p_raw = _bundle["model"].predict_proba(X)[:, 1]
trusted = np.array([is_trusted(u) for u in du["url"]])
p_ok = np.where(trusted, np.minimum(p_raw, TRUSTED_MAX), p_raw)
rows = {
    "URL modeli (ham)": metrics(du["label"], 100 * p_raw),
    "URL modeli + güvenilir alan adı katmanı": metrics(du["label"], 100 * p_ok),
}
say(tablo(rows))
say(f"\nTest setinde güvenilir listede çıkan URL sayısı: {int(trusted.sum())} / {len(du)} "
    f"({time.time() - t0:.0f} sn)\n")
if trusted.sum() > 0:
    kacan = int(((du["label"].values == 1) & trusted).sum())
    say(f"Bunlardan oltalama etiketli olanlar: {kacan} (liste bunları kaçırır, sınırlılık olarak yazılabilir).\n")

# ------------------------------------------------------------------
# B) Uçtan uca: metin + URL (OLTA-TR test bölümü), ağırlık denemeleri
# ------------------------------------------------------------------
say("## B) Metin modeli ve uçtan uca birleşik skor (OLTA-TR test bölümü)\n")
tx = pd.read_csv("data/processed/texts.csv")
if "split" in tx.columns:
    tx = tx[tx["split"] == "test"]
tx = tx.reset_index(drop=True)


def find_raw_col(df):
    """Bağlantıları içeren (linksiz olmayan) ham mesaj kolonunu bulur."""
    best, best_key = None, (-1, -1)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) and c not in ("split", "sector", "label"):
            s = df[c].astype(str)
            key = (s.str.contains("http", case=False).mean(), s.str.len().mean())
            if key > best_key:
                best, best_key = c, key
    return best, best_key[0]


col, frac = find_raw_col(tx)
if col is None or "label" not in tx.columns:
    say(f"texts.csv kolonları: {list(tx.columns)}")
    sys.exit("Uygun metin/label kolonu bulunamadı. Kolon adlarını bana yaz.")
say(f"Kullanılan ham metin kolonu: `{col}` (içinde 'http' geçen satır oranı: {frac:.2f})\n")

if len(tx) > args.n:
    tx = tx.sample(args.n, random_state=SEED).reset_index(drop=True)

t0 = time.time()
recs = []
for msg, y in zip(tx[col].astype(str), tx["label"]):
    body = strip_urls(msg)
    t = text_risk(body) if len(body.split()) >= 4 else None
    urls = extract_urls(msg)
    raws = [url_risk_raw(u) for u in urls]
    u_raw = max(raws) if raws else None
    u_ok = max((min(r, TRUSTED_MAX) if is_trusted(u) else r) for u, r in zip(urls, raws)) if raws else None
    recs.append({"y": y, "t": t, "u_raw": u_raw, "u_ok": u_ok})
R = pd.DataFrame(recs)


def combine(row, w_t, ukey):
    t, u = row["t"], row[ukey]
    t = None if pd.isna(t) else t
    u = None if pd.isna(u) else u
    if t is not None and u is not None:
        return w_t * t + (1 - w_t) * u
    if t is not None:
        return t
    if u is not None:
        return u
    return 0.0


rows = {}
has_t = R["t"].notna()
rows["Yalnız metin modeli"] = metrics(R.loc[has_t, "y"], 100 * R.loc[has_t, "t"])
has_u = R["u_raw"].notna()
if has_u.any():
    rows["Yalnız URL modeli (ham)"] = metrics(R.loc[has_u, "y"], 100 * R.loc[has_u, "u_raw"])
    rows["Yalnız URL modeli + güvenilir liste"] = metrics(R.loc[has_u, "y"], 100 * R.loc[has_u, "u_ok"])
for w_t in (0.5, 0.7, 0.3):
    name = f"Birleşik {w_t:.1f} metin / {1 - w_t:.1f} URL (güvenilir liste var)"
    rows[name] = metrics(R["y"], [100 * combine(r, w_t, "u_ok") for _, r in R.iterrows()])
rows["Birleşik 0.5/0.5 (güvenilir liste yok)"] = metrics(
    R["y"], [100 * combine(r, 0.5, "u_raw") for _, r in R.iterrows()])
say(tablo(rows))
say(f"\n{len(R)} e-posta, {int(has_u.sum())} tanesinde bağlantı var ({time.time() - t0:.0f} sn).")
say("\nNot: OLTA-TR metinleri LLM ile üretilmiş sentetik veridir, bu skorlar gerçek dünyadan yüksek olabilir. "
    "URL modeli ise başka bir veri setiyle (PhiUSIIL) eğitildi; e-postalardaki bağlantılar sentetik olabilir.\n")

# ------------------------------------------------------------------
# C) Elle yazılmış küçük değerlendirme seti (gerçekçi örnekler, uçtan uca analyze())
# ------------------------------------------------------------------
say("## C) Elle yazılmış küçük değerlendirme seti (30 mesaj)\n")
ELLE = [
    # --- oltalama (1) ---
    (1, "Sayın müşterimiz, hesabınız güvenlik nedeniyle askıya alınacaktır. Bugün içinde şifrenizi doğrulamazsanız hesabınız kapatılacaktır. Hemen tıklayın: http://garanti-hesap-guncelle.top/giris"),
    (1, "PTT Kargo: Gönderiniz adresinize teslim edilemedi. Adres bilgilerinizi güncellemek için 24 saat içinde tıklayın: http://ptt-teslimat-bilgi.xyz/adres"),
    (1, "e-Devlet hesabınızda olağandışı giriş denemesi tespit edildi. Hesabınız bloke edilmeden önce kimlik bilgilerinizi doğrulayın: http://e-devlet-dogrulama.site/giris"),
    (1, "Tebrikler! Çekilişimizi kazandınız. Hediyenizi almak için kart bilgilerinizi girin: http://hediye-kazan-hemen.top/odul"),
    (1, "Ziraat Bankası: Kartınıza yasal işlem nedeniyle erişim kısıtlaması getirildi. Şifrenizi ve doğrulama kodunuzu paylaşın: http://ziraat-giris.xyz/dogrula"),
    (1, "Hesabınız için son uyarı: Faturanız ödenmediği için aboneliğiniz iptal edilecek. Hemen ödeme bilgilerinizi güncelleyin http://fatura-odeme-guncelle.icu/netflix"),
    (1, "Sayın kullanıcı, mobil bankacılık şifreniz 24 saat içinde sona erecektir. Yeni parola oluşturmak için tıklayın: http://192.168.44.12/login/verify.php"),
    (1, "Vergi iadeniz hazır. İade tutarınızı almak için IBAN ve kimlik bilgilerinizi bu formdan iletin: https://bit.ly/iade-onay-form"),
    (1, "Yurtiçi Kargo: Paketiniz gümrükte bekliyor. Ek ücreti ödemek için kart bilgilerinizi girin: http://kargo-gumruk-odeme.top/ode"),
    (1, "Hesabınıza yetkisiz erişim yapıldı. Hesabınız askıya alınmadan önce parolanızı doğrulayın: http://hesap-guvenlik-dogrula.online/giris"),
    (1, "Turkcell: 500 TL bonus hediyeniz hesabınıza tanımlandı. Kullanmak için hemen giriş yapın http://turkcell-bonus-kazan.xyz/giris"),
    (1, "Son gün! Kredi kartı limitiniz artırıldı, onaylamak için kart numaranızı ve cvv kodunuzu hemen yazın."),
    (1, "Banka hesabınız bloke edilmiştir. Çözmek için bizi arayın ve doğrulama kodunuzu söyleyin. Aksi halde hukuki işlem başlatılacaktır."),
    (1, "Microsoft hesabınız kapatılacak. Hemen şifrenizi doğrulayın: http://microsoft-account-verify.top/signin"),
    (1, "Sayın öğrencimiz, burs ödemeniz için IBAN ve kimlik bilgilerinizi 24 saat içinde aşağıdaki bağlantıdan güncelleyin: http://burs-bilgi-guncelle.xyz/form"),
    # --- meşru (0) ---
    (0, "Merhaba, yarınki toplantı saat 14:00'te yapılacaktır. Gündem maddeleri ekte yer almaktadır. Sorularınız için bizimle iletişime geçebilirsiniz. Saygılarımızla."),
    (0, "https://claude.ai/chat/b39875fb-2a98-4254-8e9f-0bc043d132cb"),
    (0, "https://www.google.com/search?q=e+devlet+kariyer+kapisi"),
    (0, "Merhaba Ayşe, ödev teslim tarihi cuma gününe uzatıldı. Dört ilanı karşılaştırmak istersen şuraya bak: https://www.sahibinden.com/ilan/12345678"),
    (0, "Siparişiniz kargoya verildi. Takip numaranız ile gönderinizi şu adresten izleyebilirsiniz: https://www.yurticikargo.com"),
    (0, "Randevunuz 28 Kasım Cuma günü saat 10:30'a alınmıştır. Değişiklik için kliniği arayabilirsiniz."),
    (0, "Yarın akşam yemeğe gelir misin? Saat 19:00 gibi buluşalım, ben yemek yaparım."),
    (0, "https://www.turkiye.gov.tr"),
    (0, "Merhaba, proje raporunun son halini inceleyip dönüş yapar mısın? Kodlar burada: https://github.com/anthropics"),
    (0, "Ziraat Bankası: Hesabınızdan 250 TL tutarında harcama yapılmıştır. Bilgi için www.ziraatbank.com.tr adresini ziyaret edebilirsiniz."),
    (0, "Akşam yemeği için sipariş verelim mi? Menüye buradan bakabilirsin: https://www.yemeksepeti.com"),
    (0, "Toplantı notları ekte. Cuma günü saat 15:00'te ekip toplantısı var, gündemi birlikte belirleyelim."),
    (0, "Kütüphane kitabınızın iade süresi 3 gün sonra doluyor. Süreyi uzatmak için kütüphane hesabınızdan işlem yapabilirsiniz."),
    (0, "https://www.hepsiburada.com/siparislerim"),
    (0, "Doğum günün kutlu olsun! Akşam hep birlikte dışarı çıkıyoruz, sen de gel."),
]
ys, scs, lvls = [], [], []
for y, msg in ELLE:
    r = analyze(msg)
    ys.append(y)
    scs.append(r["score"] if r else 0)
    lvls.append(r["level"] if r else "-")
say(tablo({"Elle yazılmış set (uçtan uca)": metrics(ys, scs)}))
yanlis = [(y, s, l, m) for (y, m), s, l in zip(ELLE, scs, lvls) if (s >= args.esik) != bool(y)]
say(f"\nYanlış sınıflananlar: {len(yanlis)} / {len(ELLE)}\n")
for y, s, l, m in yanlis:
    gercek = "oltalama" if y else "meşru"
    say(f"- gerçek: {gercek}, puan: {s} ({l}) → {m[:90]}")
say("\nNot: Bu 30 mesajı ben yazdım, sen de kendi gerçekçi örneklerini ekleyip kendi `ELLE` listene çevir. "
    "Bu sayılar küçük örnekleme dayanır, kesin ölçü değil, örnek olarak kullan.\n")

# ------------------------------------------------------------------
# D) RAG: rerank'lı ve rerank'sız (isteğe bağlı, --rag ile)
# ------------------------------------------------------------------
if args.rag:
    say("## D) RAG: rerank'lı ve rerank'sız karşılaştırma\n")
    from src.rag import search
    pool = tx.copy()
    pool["body"] = pool[col].astype(str).map(strip_urls)
    pool = pool[pool["body"].str.split().str.len() >= 4]
    pool = pool.sample(min(args.rag_n, len(pool)), random_state=SEED)
    say(f"{len(pool)} test e-postası için en yakın {RERANK_TOP_N} örnek getirildi.\n")
    out = {}
    for use_rerank in (False, True):
        votes, same = [], []
        t0 = time.time()
        for body, y in zip(pool["body"], pool["label"]):
            ex = search(body, use_rerank=use_rerank)[:RERANK_TOP_N]
            labs = [e["label"] for e in ex]
            if not labs:
                continue
            votes.append(int(np.mean(labs) >= 0.5) == int(y))
            same.append(np.mean([int(l) == int(y) for l in labs]))
        out["rerank'lı" if use_rerank else "rerank'sız"] = (np.mean(votes), np.mean(same), time.time() - t0)
    say("| Yöntem | çoğunluk oyu doğruluğu | örneklerin ortalama etiket uyumu | süre (sn) |\n|---|---|---|---|")
    for ad, (a, b, s) in out.items():
        say(f"| {ad} | {a:.3f} | {b:.3f} | {s:.0f} |")
    say("\nUyarı: 04_build_vectorstore.py bilgi tabanına test bölümündeki e-postaları da koyduysa bu sonuç "
        "şişer (aynı mesaj kendisini bulur). Yalnızca train bölümünden kurulduysa geçerlidir.\n")

os.makedirs("data/processed", exist_ok=True)
with open("data/processed/eval_sonuclari.md", "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
print("\nKaydedildi: data/processed/eval_sonuclari.md")
