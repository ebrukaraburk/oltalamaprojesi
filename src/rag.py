import sys, re, requests, chromadb
sys.path.insert(0, ".")
from config import *

_client = chromadb.PersistentClient(path=RAG_PATH)
_col = _client.get_collection(RAG_COLLECTION)


def embed(text):
    r = requests.post(f"{OLLAMA_URL}/api/embed",
                      json={"model": EMBED_MODEL, "input": [text]}, timeout=120)
    r.raise_for_status()
    return r.json()["embeddings"][0]


def retrieve(query, k=RETRIEVAL_K):
    """1. aşama: vektör araması, k aday getirir."""
    res = _col.query(query_embeddings=[embed(query)], n_results=k)
    return [{"text": d, "label": m["label"], "sector": m.get("sector", ""), "distance": dist}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]


def llm_score(query, passage):
    prompt = ("Aşağıdaki SORGU ile BELGE arasındaki ilişkiyi 0 ile 10 arasında puanla. "
              "10 = belge, sorguyla aynı türde ve doğrudan karşılaştırılabilir bir mesaj; "
              "0 = ilgisiz. Sadece tek bir tam sayı yaz.\n\n"
              f"SORGU: {query[:600]}\n\nBELGE: {passage[:600]}\n\nPuan:")
    r = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": LLM_MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 4, "seed": SEED}},
        timeout=120)
    r.raise_for_status()
    m = re.search(r"\d+", r.json()["response"])
    return min(int(m.group()), 10) if m else 0


def rerank(query, candidates, top_n=RERANK_TOP_N):
    """2. aşama: adayları sorguyla birlikte puanla, en iyi top_n'i seç."""
    for c in candidates:
        c["rerank_score"] = llm_score(query, c["text"])
    return sorted(candidates, key=lambda c: (-c["rerank_score"], c["distance"]))[:top_n]


def search(query, use_rerank=True):
    cands = retrieve(query)
    return rerank(query, cands) if use_rerank else cands[:RERANK_TOP_N]


if __name__ == "__main__":
    q = "Hesabınız askıya alındı, doğrulamak için hemen bağlantıya tıklayın"
    print("== REKRANK ÖNCESİ (ilk 4, yalnızca vektör araması) ==")
    for c in retrieve(q)[:RERANK_TOP_N]:
        print(f"[{c['distance']:.3f}] label={c['label']} {c['sector']:<20} {c['text'][:90]}")
    print("\n== RERANK SONRASI (top_n=4) ==")
    for c in search(q):
        print(f"[skor={c['rerank_score']} d={c['distance']:.3f}] label={c['label']} {c['sector']:<20} {c['text'][:90]}")