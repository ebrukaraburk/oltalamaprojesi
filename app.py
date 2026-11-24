import html, re, sys
sys.path.insert(0, ".")
import streamlit as st
from config import RETRIEVAL_K, RERANK_TOP_N, EXPLAIN_MODEL, LLM_OPTIONS
from src.explain import explain
from src.scoring import URL_RE, SIGNALS, tr_lower

st.set_page_config(page_title="PhishShield AI", page_icon="🛡️", layout="centered",
                   initial_sidebar_state="collapsed")

DENEME = {
    "🏦 Banka mesajı": "Sayın müşterimiz, hesabınız güvenlik nedeniyle askıya alınacaktır. Bugün içinde şifrenizi "
                       "doğrulamazsanız hesabınız kapatılacaktır. Hemen tıklayın: http://garanti-hesap-guncelle.top/giris",
    "🔗 Kısa link": "https://bit.ly/3xYzAbc",
    "📅 Toplantı mesajı": "Merhaba, yarınki toplantı saat 14:00'te yapılacaktır. Gündem maddeleri ekte yer almaktadır. "
                          "Sorularınız için bizimle iletişime geçebilirsiniz. Saygılarımızla.",
}

RED = "background:rgba(220,60,60,0.22);padding:1px 4px;border-radius:4px;"
AMBER = "background:rgba(240,170,30,0.28);padding:1px 4px;border-radius:4px;"
YELLOW = "background:rgba(250,220,80,0.30);padding:1px 4px;border-radius:4px;"

# seviye: (ana renk, zemin rengi, başlık, alt yazı, simge)
LEVELS = {
    "yüksek": ("#e5484d", "#fdecec", "YÜKSEK RİSK", "Bu mesaja güvenmeyin.", "🚨"),
    "orta": ("#e8930c", "#fff6e5", "ORTA RİSK", "Dikkatli olun.", "⚠️"),
    "düşük": ("#2fb36d", "#e9f8f0", "DÜŞÜK RİSK", "Belirgin bir tehlike bulunamadı.", "✅"),
}


# ---------------- Görünüm (CSS) ----------------
def stil_ekle():
    st.markdown("""
<style>
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {display:none !important;}
[data-testid="stHeader"] {background: transparent;}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #eef4ff 0%, #ffffff 45%, #f3f0ff 100%);
}
.block-container {max-width: 820px; padding-top: 1rem; padding-bottom: 3rem;}

.hero {text-align:center; padding: 1.6rem 1rem 1rem 1rem;}
.hero .logo {font-size: 3.4rem; line-height:1;}
.hero h1 {font-size: 2.7rem; margin: .3rem 0 .2rem 0; color:#0f2a5c; letter-spacing:-0.5px; padding:0;}
.hero p {color:#5a6b8c; font-size:1.1rem; margin:0 auto; max-width:560px;}
.rozetler {display:flex; gap:.5rem; justify-content:center; flex-wrap:wrap; margin-top:1.1rem;}
.rozet {background:#fff; border:1px solid #dbe5fb; color:#26437f; padding:.35rem .8rem;
        border-radius:999px; font-size:.85rem; box-shadow:0 1px 3px rgba(20,50,120,.06);}

.bolum {color:#0f2a5c; font-weight:700; font-size:1.15rem; margin:1.6rem 0 .6rem 0;}
.kucuk {color:#6b7a99; font-size:.85rem; margin-bottom:.3rem;}

textarea {
    border-radius: 14px !important; border: 1.5px solid #cfdcf7 !important;
    background:#fff !important; font-size:1rem !important;
    box-shadow: 0 4px 18px rgba(30,70,160,.07);
}
textarea:focus {border-color:#2f6fed !important;}

.stButton > button {
    border-radius: 12px; border:1px solid #cfdcf7; background:#fff; color:#26437f;
    padding:.55rem 1rem; transition: all .15s;
}
.stButton > button:hover {border-color:#2f6fed; color:#2f6fed; transform: translateY(-1px);}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#2f6fed,#5b3df5); color:#fff; border:none;
    padding:.7rem 1.8rem; font-weight:600; box-shadow:0 6px 16px rgba(60,80,240,.30);
}
.stButton > button[kind="primary"]:hover {color:#fff; filter:brightness(1.07);}

.kart {background:#fff; border:1px solid #e1e9fb; border-radius:16px; padding:1.1rem 1.3rem;
       box-shadow:0 4px 14px rgba(30,70,160,.06); color:#1d2b4a;}
.kart.vurgu {border-left:5px solid #2f6fed;}
.kart h4 {margin:0 0 .4rem 0; color:#0f2a5c; font-size:1rem;}

.sonuc {display:flex; align-items:center; gap:1.4rem; border:2px solid; border-radius:20px;
        padding:1.2rem 1.5rem; margin-top:.5rem;}
.gosterge {width:110px; height:110px; border-radius:50%; display:flex; align-items:center;
           justify-content:center; flex-shrink:0;}
.gosterge .ic {width:82px; height:82px; border-radius:50%; background:#fff; display:flex;
               flex-direction:column; align-items:center; justify-content:center; line-height:1;}
.gosterge .ic b {font-size:1.9rem; color:#0f2a5c;}
.gosterge .ic span {font-size:.75rem; color:#6b7a99; margin-top:3px;}
.sonuc .etiket {font-size:1.5rem; font-weight:800; letter-spacing:.3px;}
.sonuc .alt {font-size:1.05rem; color:#33415f; margin-top:.2rem;}

.mini {background:#fff; border:1px solid #e1e9fb; border-radius:14px; padding:.8rem 1rem;
       box-shadow:0 3px 10px rgba(30,70,160,.05); text-align:center;}
.mini .ad {color:#6b7a99; font-size:.8rem;}
.mini .deger {color:#0f2a5c; font-size:1.5rem; font-weight:700;}

.adim {text-align:center; padding:1.2rem .8rem;}
.adim .no {width:34px; height:34px; border-radius:50%; margin:0 auto .5rem auto;
           background:linear-gradient(135deg,#2f6fed,#5b3df5); color:#fff; font-weight:700;
           display:flex; align-items:center; justify-content:center;}
.adim b {color:#0f2a5c;}
.adim div.t {color:#5a6b8c; font-size:.9rem; margin-top:.2rem;}

.alt-yazi {text-align:center; color:#8793ad; font-size:.8rem; margin-top:2.5rem;}
[data-testid="stSidebar"] {background:#f6f8ff;}
[data-testid="stExpander"] {border-radius:14px; background:#fff;}
</style>
""", unsafe_allow_html=True)


def md_html(s):
    """HTML'i Markdown'ın kod bloğu sanmaması için girintileri ve boş satırları temizleyip basar."""
    temiz = "\n".join(l.strip() for l in s.splitlines() if l.strip())
    st.markdown(temiz, unsafe_allow_html=True)


def ust_alan():
    md_html("""
<div class="hero">
<div class="logo">🛡️</div>
<h1>PhishShield AI</h1>
<p>Şüpheli bir mesaj mı aldınız? Buraya yapıştırın, güvenli mi değil mi sade bir dille anlatalım.</p>
<div class="rozetler">
<span class="rozet">🔗 Bağlantı analizi</span>
<span class="rozet">✉️ Metin analizi</span>
<span class="rozet">🧠 Türkçe açıklama</span>
<span class="rozet">🔒 Mesajlar kaydedilmez</span>
</div>
</div>
""")


def nasil_calisir():
    md_html('<div class="bolum">Nasıl çalışır?</div>')
    c1, c2, c3 = st.columns(3)
    adimlar = [
        ("1", "Yapıştırın", "E-posta, SMS ya da bağlantıyı kutuya yapıştırın."),
        ("2", "Analiz edilir", "Bağlantı ve metin modelleri 0-100 arası risk puanı üretir."),
        ("3", "Açıklama alın", "Neden şüpheli olduğunu ve ne yapmanız gerektiğini okuyun."),
    ]
    for col, (no, baslik, yazi) in zip((c1, c2, c3), adimlar):
        with col:
            md_html(f'<div class="kart adim"><div class="no">{no}</div><b>{baslik}</b>'
                    f'<div class="t">{yazi}</div></div>')


def set_msg(text):
    st.session_state["msg"] = text


def highlight_html(message, a):
    """Şüpheli ifadeleri, sinyal kelimelerini ve bağlantıları işaretler."""
    colors = {}
    for cat, kws in SIGNALS.items():
        for kw in kws:
            colors[tr_lower(kw)] = AMBER if cat == "aciliyet" else RED
    for term in a["highlights"]:
        colors.setdefault(tr_lower(term), YELLOW)

    terms = sorted(colors.keys(), key=len, reverse=True)
    safe = html.escape(message)
    if not terms:
        pattern = re.compile(f"(?P<url>{URL_RE.pattern})", re.IGNORECASE)
    else:
        alt = "|".join(re.escape(t) for t in terms)
        pattern = re.compile(f"(?P<url>{URL_RE.pattern})|(?P<term>{alt})", re.IGNORECASE)

    def repl(m):
        if m.group("url"):
            return f'<mark style="{RED}">{m.group("url")}</mark>'
        t = m.group("term")
        return f'<mark style="{colors.get(tr_lower(t), YELLOW)}">{t}</mark>'

    return pattern.sub(repl, safe)


stil_ekle()

# ---------------- Kenar çubuğu: kullanılan ayarlar ----------------
with st.sidebar:
    st.header("Ayarlar")
    use_rerank = st.checkbox("Rerank kullan", value=True)
    st.caption(f"RAG: retrieval k={RETRIEVAL_K}, rerank top_n={RERANK_TOP_N}")
    st.caption(f"Açıklama modeli: {EXPLAIN_MODEL}")
    st.caption(f"temperature={LLM_OPTIONS['temperature']}, num_ctx={LLM_OPTIONS['num_ctx']}, "
               f"num_predict={LLM_OPTIONS['num_predict']}")
    st.divider()
    st.caption("Sistem yalnızca savunma amaçlıdır. Mesajlar kaydedilmez.")

# ---------------- Ana ekran ----------------
ust_alan()

md_html('<div class="kucuk">Hızlı deneme için bir örnek seçin:</div>')
cols = st.columns(len(DENEME))
for col, (ad, metin) in zip(cols, DENEME.items()):
    col.button(ad, on_click=set_msg, args=(metin,), use_container_width=True)

message = st.text_area("Mesaj", key="msg", height=150, label_visibility="collapsed",
                       placeholder="E-posta, SMS veya bağlantıyı buraya yapıştırın...")

if st.button("🔍 Analiz et", type="primary"):
    if not message.strip():
        st.warning("Önce bir mesaj ya da bağlantı yapıştırın.")
    else:
        with st.spinner("Analiz ediliyor, açıklama hazırlanıyor..."):
            st.session_state["result"] = explain(message, use_rerank=use_rerank)
            st.session_state["analyzed"] = message

res = st.session_state.get("result")
if res is None and "analyzed" in st.session_state:
    st.warning("Bu girdiden analiz edilecek bir metin ya da bağlantı çıkarılamadı.")

if not res:
    nasil_calisir()

if res:
    msg_shown = st.session_state["analyzed"]
    renk, zemin, baslik, alt, ikon = LEVELS[res["level"]]
    score = int(res["score"])

    md_html('<div class="bolum">Sonuç</div>')
    md_html(f"""
<div class="sonuc" style="border-color:{renk};background:{zemin}">
<div class="gosterge" style="background:conic-gradient({renk} {score * 3.6}deg,#e3eafa 0deg)">
<div class="ic"><b>{score}</b><span>/ 100</span></div>
</div>
<div>
<div class="etiket" style="color:{renk}">{ikon} {baslik}</div>
<div class="alt">{alt}</div>
</div>
</div>
""")

    text_txt = "yok" if res["text_prob"] is None else f"%{round(100 * res['text_prob'])}"
    url_max = max((p for _, p in res["url_probs"]), default=None)
    url_txt = "yok" if url_max is None else f"%{round(100 * url_max)}"
    st.write("")
    m1, m2 = st.columns(2)
    with m1:
        md_html(f'<div class="mini"><div class="ad">Metin modeli</div><div class="deger">{text_txt}</div></div>')
    with m2:
        md_html(f'<div class="mini"><div class="ad">Bağlantı modeli</div><div class="deger">{url_txt}</div></div>')

    if res["text_prob"] is None and url_max is not None:
        st.caption("Yalnızca bağlantı analizi yapıldı. Tek modele dayandığı için kesin hüküm sayılmamalıdır.")

    md_html('<div class="bolum">Bu mesaj neden şüpheli?</div>')
    md_html(f'<div class="kart vurgu">{html.escape(res["explanation"])}</div>')
    st.write("")
    md_html(f'<div class="kart" style="border-left:5px solid {renk}"><h4>Ne yapmalı?</h4>'
            f'{html.escape(res["advice"])}</div>')

    if res["level"] != "düşük":
        md_html('<div class="bolum">Şüpheli ifadeler işaretlendi</div>')
        chips = ""
        if res["signals"]:
            chips = "<div style='margin-bottom:.6rem'>" + "".join(
                f'<span style="{AMBER if k == "aciliyet" else RED}margin-right:8px;font-size:13px">{html.escape(k)}</span>'
                for k in res["signals"]) + "</div>"
        st.markdown(f'<div class="kart">{chips}<div style="line-height:2;font-size:15px">'
                    f'{highlight_html(msg_shown, res)}</div></div>', unsafe_allow_html=True)

    st.write("")
    with st.expander("Benzer bilinen örnekler (RAG)"):
        if not res["examples"]:
            st.write("Bu girdi için örnek getirilmedi (yalnızca bağlantı var).")
        for e in res["examples"]:
            tur = "oltalama" if e["label"] == 1 else "meşru"
            rr = f", rerank skoru {e['rerank_score']}" if "rerank_score" in e else ""
            st.markdown(f"**{tur}** · {e['sector']} · mesafe {e['distance']:.3f}{rr}")
            st.caption(e["text"][:300])

    with st.expander("Teknik ayrıntılar"):
        st.text(res["findings"])
        st.write("Bağlantı olasılıkları:", res["url_probs"])

md_html('<div class="alt-yazi">PhishShield AI · Yalnızca savunma amaçlıdır · Mesajlar kaydedilmez</div>')