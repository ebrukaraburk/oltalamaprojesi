
# ---- Açıklama modeli (rerank için LLM_MODEL = llama3.2 kalır) ----
EXPLAIN_MODEL = "qwen3:14b"

# ---- Veri ----
SEED = 42
TEST_SIZE = 0.2                 # test verisi RAG'e ve eğitime girmez

# ---- Chunking (dokümanın 4. bölümü) ----
CHUNK_SIZE = 1000               # karakter bazlı splitter
CHUNK_OVERLAP = 150

# ---- Retrieval ve rerank (dokümanın 2 ve 3. bölümleri) ----
RETRIEVAL_K = 12                # ilk aşama aday sayısı (candidate_count)
RERANK_TOP_N = 4                # LLM'e verilecek son parça sayısı
DISTANCE = "cosine"             # cosine distance: düşük = daha yakın

# ---- Embedding ----
EMBED_MODEL = "bge-m3"
OLLAMA_URL = "http://localhost:11434"

# ---- LLM üretim ayarları (dokümanın 6 ve 7. bölümleri) ----
LLM_MODEL = "llama3.2"
LLM_OPTIONS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 40,                # LLM top_k: token havuzu, retrieval k DEĞİL
    "num_ctx": 4096,
    "num_predict": 300,
    "repeat_penalty": 1.1,
    "seed": 42,
}

# ---- RAG indeksi ----
RAG_PATH = "rag_db"
RAG_COLLECTION = "phish_kb"
RAG_MAX_PER_LABEL = 500        # her sınıftan 1000 e-posta

