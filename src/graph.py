"""PhishShield AI akışı (LangGraph).

Akış:
  girdiyi sınıflandır -> analiz et (URL + metin, puan) -> koşullu yönlendirme:
      boş girdi           -> bitir
      düşük risk          -> hızlı yol (RAG ve LLM çağrılmaz)
      yalnızca bağlantı   -> açıklama (RAG atlanır)
      metin içeren risk   -> RAG ile benzer örnekler -> açıklama

Kullanım:
    from src.graph import run
    sonuc = run("mesaj metni", use_rerank=True)   # explain() ile aynı biçimde sözlük döner

Deneme:
    .venv\\Scripts\\python.exe src\\graph.py
"""
import sys
from typing import TypedDict

sys.path.insert(0, ".")
from langgraph.graph import StateGraph, START, END
from src.scoring import analyze, extract_urls, strip_urls
from src.rag import search
from src.explain import build_findings, fallback_why, llm_why, ADVICE, LOW_TEXT


class State(TypedDict, total=False):
    message: str          # kullanıcının yapıştırdığı girdi
    use_rerank: bool
    input_type: str       # "metin" | "bağlantı" | "karışık" | "boş"
    analysis: dict        # scoring.analyze() çıktısı
    examples: list        # RAG ile gelen benzer örnekler
    result: dict          # arayüze verilecek son sonuç


# ---------------- düğümler (node) ----------------
def classify_node(state: State):
    msg = state["message"]
    has_url = bool(extract_urls(msg))
    has_text = len(strip_urls(msg).split()) >= 4
    if has_url and has_text:
        kind = "karışık"
    elif has_url:
        kind = "bağlantı"
    elif has_text:
        kind = "metin"
    else:
        kind = "boş"
    return {"input_type": kind}


def analyze_node(state: State):
    return {"analysis": analyze(state["message"])}      # URL modeli + metin modeli + risk puanı


def empty_node(state: State):
    return {"result": None}


def low_risk_node(state: State):
    a = dict(state["analysis"])
    a["advice"] = ADVICE[a["level"]]
    a["findings"] = build_findings(a, [])
    a["explanation"] = LOW_TEXT
    a["examples"] = []
    return {"result": a}


def rag_node(state: State):
    a = state["analysis"]
    examples = search(a["body"], use_rerank=state.get("use_rerank", True))
    return {"examples": examples}


def explain_node(state: State):
    a = dict(state["analysis"])
    examples = state.get("examples", [])
    a["advice"] = ADVICE[a["level"]]
    findings = build_findings(a, examples)
    try:
        why = llm_why(findings, state["message"])
    except Exception as e:
        print("LLM hatası, yedek metin kullanılıyor:", e)
        why = ""
    if len(why) < 20:
        why = fallback_why(a)
    a["findings"] = findings
    a["explanation"] = why
    a["examples"] = examples
    return {"result": a}


# ---------------- koşullu yönlendirme (conditional routing) ----------------
def route_after_analysis(state: State):
    a = state["analysis"]
    if a is None:
        return "bos"
    if a["level"] == "düşük":
        return "dusuk"
    if state["input_type"] == "bağlantı":
        return "aciklama"          # sadece link varsa benzer örnek aramanın anlamı yok
    return "rag"


# ---------------- grafı kur ----------------
def build_graph():
    g = StateGraph(State)
    g.add_node("classify", classify_node)
    g.add_node("analyze", analyze_node)
    g.add_node("empty", empty_node)
    g.add_node("low_risk", low_risk_node)
    g.add_node("rag", rag_node)
    g.add_node("explain", explain_node)

    g.add_edge(START, "classify")
    g.add_edge("classify", "analyze")
    g.add_conditional_edges("analyze", route_after_analysis, {
        "bos": "empty",
        "dusuk": "low_risk",
        "rag": "rag",
        "aciklama": "explain",
    })
    g.add_edge("rag", "explain")
    g.add_edge("empty", END)
    g.add_edge("low_risk", END)
    g.add_edge("explain", END)
    return g.compile()


_app = build_graph()


def run(message, use_rerank=True):
    out = _app.invoke({"message": message, "use_rerank": use_rerank})
    return out["result"]


if __name__ == "__main__":
    try:
        print("--- Mermaid diyagramı (rapora yapıştırabilirsin) ---")
        print(_app.get_graph().draw_mermaid())
    except Exception as e:
        print("Diyagram çıkarılamadı:", e)

    tests = [
        "Sayın müşterimiz, hesabınız güvenlik nedeniyle askıya alınacaktır. Bugün içinde şifrenizi "
        "doğrulamazsanız hesabınız kapatılacaktır. Hemen tıklayın: http://garanti-hesap-guncelle.top/giris",
        "https://bit.ly/3xYzAbc",
        "Merhaba, yarınki toplantı saat 14:00'te yapılacaktır. Gündem maddeleri ekte yer almaktadır. "
        "Sorularınız için bizimle iletişime geçebilirsiniz. Saygılarımızla.",
    ]
    for m in tests:
        res = run(m)
        print("\n" + "=" * 60)
        print("PUAN:", res["score"], "->", res["level"])
        print("AÇIKLAMA:", res["explanation"])
        print(res["advice"])