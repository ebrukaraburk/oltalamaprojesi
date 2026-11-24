import sys, requests, pandas as pd
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
sys.path.insert(0, ".")
from config import *

def embed(texts):
    r = requests.post(f"{OLLAMA_URL}/api/embed",
                      json={"model": EMBED_MODEL, "input": texts}, timeout=600)
    r.raise_for_status()
    return r.json()["embeddings"]

df = pd.read_csv("data/processed/texts.csv")
df = df[(df["lang"] == "tr") & (df["split"] == "train")]     # Türkçe + yalnızca train
parts = [g.sample(min(len(g), RAG_MAX_PER_LABEL), random_state=SEED) for _, g in df.groupby("label")]
df = pd.concat(parts)
print("İndekslenecek e-posta:", len(df))

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

ids, docs, metas = [], [], []
for i, row in enumerate(df.itertuples()):
    for j, chunk in enumerate(splitter.split_text(row.text)):
        ids.append(f"d{i}_c{j}")
        docs.append(chunk)
        metas.append({"label": int(row.label), "source": row.source,
                      "sector": str(row.sector), "doc": i, "chunk": j})
print("Toplam chunk:", len(docs))

client = chromadb.PersistentClient(path=RAG_PATH)
try:
    client.delete_collection(RAG_COLLECTION)
except Exception:
    pass
col = client.create_collection(RAG_COLLECTION, metadata={"hnsw:space": DISTANCE})

B = 32
for s in range(0, len(docs), B):
    col.add(ids=ids[s:s+B], documents=docs[s:s+B], metadatas=metas[s:s+B],
            embeddings=embed(docs[s:s+B]))
    if (s // B) % 20 == 0:
        print(f"{min(s+B, len(docs))}/{len(docs)}")

q = "Paketiniz teslim edilemedi, adres bilginizi güncellemek için bağlantıya tıklayın"
res = col.query(query_embeddings=embed([q]), n_results=RETRIEVAL_K)
for d, m, dist in zip(res["documents"][0][:5], res["metadatas"][0][:5], res["distances"][0][:5]):
    print(f"[{dist:.3f}] label={m['label']} {d[:100]}")
print("Bitti:", col.count())