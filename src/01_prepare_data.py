import glob, os, sys
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0, ".")
from config import SEED, TEST_SIZE

PHIUSIIL_LEGIT_IS_1 = True   # UCI sayfasından doğrula; ters ise False yap


def add_split(df):
    df = df.copy()
    df["split"] = "train"
    _, test_idx = train_test_split(df.index, test_size=TEST_SIZE,
                                   random_state=SEED, stratify=df["label"])
    df.loc[test_idx, "split"] = "test"
    return df


def clean(s):
    return s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


# ---------- METİN: OLTA-TR (Türkçe e-posta) ----------
texts = []
for name, lab in [("phishing", 1), ("legitimate", 0)]:
    f = f"data/raw/olta_tr/{name}.csv"
    if not os.path.exists(f):
        sys.exit(f"Bulunamadı: {f}  (önce 00_download_data.py çalıştır)")
    d = pd.read_csv(f)
    print(f"[OLTA-TR] {name}: {len(d)} satır, kolonlar: {d.columns.tolist()}")
    texts.append(pd.DataFrame({
        "text": d["email_with_url"],
        "text_nourl": d["email_without_url"],
        "label": lab,
        "source": "olta_tr",
        "lang": "tr",
        "sector": d["sector"],
    }))

# SMS (isteğe bağlı, İngilizce)
for f in glob.glob("data/raw/sms/*"):
    try:
        s = pd.read_csv(f, sep="\t", header=None, names=["lab", "text"])
        texts.append(pd.DataFrame({
            "text": s["text"],
            "text_nourl": s["text"],
            "label": (s["lab"] == "spam").astype(int),
            "source": "sms",
            "lang": "en",
            "sector": "sms",
        }))
        print("[SMS]", f, len(s))
    except Exception as e:
        print("SMS atlandı:", e)

T = pd.concat(texts, ignore_index=True)
T["text"] = clean(T["text"])
T["text_nourl"] = clean(T["text_nourl"])
before = len(T)
T = T[T["text"].str.len() > 20].drop_duplicates(subset="text").reset_index(drop=True)
print(f"\nTekrarlar atıldı: {before} -> {len(T)}")
T = add_split(T)
os.makedirs("data/processed", exist_ok=True)
T.to_csv("data/processed/texts.csv", index=False)
print("\nMETİN:\n", T.groupby(["source", "label", "split"]).size())

# ---------- URL: PhiUSIIL ----------
urls = []
for f in glob.glob("data/raw/phiusiil/*.csv"):
    u = pd.read_csv(f)
    print("\n[PhiUSIIL]", f, u.columns.tolist()[:8], "...")
    lab = (1 - u["label"]) if PHIUSIIL_LEGIT_IS_1 else u["label"]
    urls.append(pd.DataFrame({"url": u["URL"], "label": lab, "source": "phiusiil"}))
if urls:
    U = pd.concat(urls, ignore_index=True).dropna().drop_duplicates(subset="url").reset_index(drop=True)
    U = add_split(U)
    U.to_csv("data/processed/urls.csv", index=False)
    print("\nURL:\n", U.groupby(["source", "label", "split"]).size())
else:
    print("\nURL verisi yok: data/raw/phiusiil/ içine CSV koy.")