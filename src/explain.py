import re, sys, requests
sys.path.insert(0, ".")
from config import *
from src.scoring import analyze
from src.rag import search

SYSTEM = (
    "Sen Türkçe yazan bir güvenlik yardımcısısın. Yalnızca Türkçe yaz, İngilizce kelime kullanma. "
    "Kısa, sade ve saygılı cümleler kur; yaşlı biri de anlayabilmeli. "
    "Yalnızca sana verilen bulguları kullan, yeni bilgi ekleme, tahmin yürütme. "
    "Kesin hüküm verme, 'şüpheli görünüyor' gibi ifadeler kullan."
)

ORNEK_BULGU = (
    "Risk seviyesi: yüksek\n"
    "- aciliyet baskısı: hemen, bugün içinde\n"
    "- tehdit: askıya, kapatıl\n"
    "- kişisel bilgi talebi: şifre\n"
    "- bağlantı: adreste banka adı geçiyor ama adres bankaya ait değil\n"
    "- Veri tabanındaki en benzer 4 örnekten 4 tanesi oltalama."
)
ORNEK_CIKTI = (
    "Mesaj sizi aceleye getirip hesabınızın kapatılacağını söyleyerek korkutuyor ve şifrenizi istiyor. "
    "Bağlantı adresi bir banka adı taşıyor ama bankaya ait görünmüyor. "
    "Veri tabanındaki benzer örneklerin tamamı da oltalama mesajlarıydı."
)

ADVICE = {
    "yüksek": "Ne yapmalı: Bağlantıya tıklamayın, şifre veya kart bilgisi girmeyin. "
              "Kurumu resmi uygulamasından ya da bilinen telefon numarasından arayarak doğrulayın. "
              "Gerekirse bir yakınınıza haber verin.",
    "orta": "Ne yapmalı: Bağlantıya tıklamadan önce göndereni başka bir yoldan doğrulayın "
            "ve kişisel bilgi paylaşmayın.",
    "düşük": "Belirgin bir risk bulunamadı. Yine de beklemediğiniz mesajlardaki bağlantılara dikkat edin.",
}

LOW_TEXT = "Mesajda belirgin bir oltalama işareti bulunamadı: aciliyet baskısı, tehdit, kişisel bilgi talebi ya da şüpheli bağlantı görünmüyor."


def build_findings(a, examples):
    lines = [f"Risk seviyesi: {a['level']}"]
    for k, v in a["signals"].items():
        lines.append(f"- {k}: {', '.join(v)}")
    for u, reasons in a["url_reasons"].items():
        if reasons:
            lines.append("- bağlantı: " + "; ".join(reasons))
    if a["highlights"]:
        lines.append("- modelin dikkat çektiği ifadeler: " + ", ".join(a["highlights"]))
    if examples:
        k = sum(1 for e in examples if e["label"] == 1)
        lines.append(f"- Veri tabanındaki en benzer {len(examples)} örnekten {k} tanesi oltalama.")
    if len(lines) == 1:
        lines.append("- belirgin şüpheli bulgu yok")
    return "\n".join(lines)


def fallback_why(a):
    parts = []
    if a["signals"]:
        parts.append("Mesajda " + ", ".join(a["signals"].keys()) + " işaretleri var.")
    for reasons in a["url_reasons"].values():
        if reasons:
            parts.append("Bağlantıda şu şüpheli özellikler var: " + "; ".join(reasons) + ".")
            break
    return " ".join(parts) or "Belirgin bir şüpheli işaret bulunamadı."


def llm_why(findings, message):
    user = (
        f"ÖRNEK\nBULGULAR:\n{ORNEK_BULGU}\nYAZI:\n{ORNEK_CIKTI}\n\n"
        f"ŞİMDİ SIRA SENDE\nİNCELENEN MESAJ:\n{message[:800]}\n"
        f"BULGULAR:\n{findings}\n\n"
        "GÖREV: Yukarıdaki örnek gibi, en fazla 3 cümleyle bu mesajın neden şüpheli (ya da neden "
        "şüpheli görünmediğini) açıkla. Puan yazma. Ne yapılması gerektiğini yazma. Sadece YAZI kısmını ver."
    )
    r = requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": EXPLAIN_MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": LLM_OPTIONS,
    }, timeout=900)
    r.raise_for_status()
    text = r.json()["message"]["content"]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return text


def explain(message, use_rerank=True):
    a = analyze(message)
    if a is None:
        return None
    a["advice"] = ADVICE[a["level"]]

    # düşük riskte RAG ve LLM çağrısı yapma: hızlı ve gereksiz model yüklemesi yok
    if a["level"] == "düşük":
        a["findings"] = build_findings(a, [])
        a["explanation"] = LOW_TEXT
        a["examples"] = []
        return a

    examples = []
    if len(a["body"].split()) >= 4:          # sadece link varsa RAG anlamsız
        examples = search(a["body"], use_rerank=use_rerank)
    findings = build_findings(a, examples)
    try:
        why = llm_why(findings, message)
    except Exception as e:
        print("LLM hatası, yedek metin kullanılıyor:", e)
        why = ""
    if len(why) < 20:
        why = fallback_why(a)
    a["findings"] = findings
    a["explanation"] = why
    a["examples"] = examples
    return a


if __name__ == "__main__":
    tests = [
        "Sayın müşterimiz, hesabınız güvenlik nedeniyle askıya alınacaktır. Bugün içinde şifrenizi "
        "doğrulamazsanız hesabınız kapatılacaktır. Hemen tıklayın: http://garanti-hesap-guncelle.top/giris",
        "https://bit.ly/3xYzAbc",
        "Merhaba, yarınki toplantı saat 14:00'te yapılacaktır. Gündem maddeleri ekte yer almaktadır. "
        "Sorularınız için bizimle iletişime geçebilirsiniz. Saygılarımızla.",
    ]
    for m in tests:
        res = explain(m)
        print("\n" + "=" * 60)
        print("PUAN:", res["score"], "->", res["level"])
        print("BULGULAR:\n" + res["findings"])
        print("\nAÇIKLAMA:", res["explanation"])
        print("\n" + res["advice"])