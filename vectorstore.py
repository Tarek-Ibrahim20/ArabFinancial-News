# vectorstore.py — AraFinNews RAG Pipeline
#
# Full pipeline: CSV → Documents → Chunks → Embeddings → Qdrant (hybrid search)
#
# Hybrid search combines:
#   dense  : semantic embeddings (Arabic-Triplet-Matryoshka-V2)
#   sparse : BM25 keyword vectors (Qdrant/bm25 via FastEmbed)
#
# Run once to build the store. Subsequent runs skip re-embedding
# if the collection already has the expected number of points.
#
# Config via .env:
#   csv_path         : preprocessed dataset
#   model_id         : HuggingFace dense embedding model
#   chunk_size       : max tokens per chunk
#   chunk_overlap    : overlap tokens between chunks
#   qdrant_path      : local directory for Qdrant persistence
#   qdrant_collection: Qdrant collection name

import os
import time

from tqdm import tqdm

import pandas as pd
from dotenv import dotenv_values
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from transformers import AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
config        = dotenv_values(".env")
MODEL_ID      = config["model_id"]
CSV_PATH      = config["csv_path"]
CHUNK_SIZE    = int(config["chunk_size"])
CHUNK_OVERLAP = int(config["chunk_overlap"])
QDRANT_PATH   = config["qdrant_path"]
COLLECTION    = config["qdrant_collection"]

# ── Tokenizer (token-based length for splitter) ───────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def token_len(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

# ── 1. Load → Documents ───────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", nrows=10000  )
df.dropna(subset=["text"], inplace=True)
df["date"] = pd.to_datetime(df["date"])

documents = [
    Document(
        page_content=row["text"],
        metadata={
            "id":      int(row["id"]),
            "title":   row["title"],
            "date":    row["date"].strftime("%Y-%m-%d"),
            "year":    int(row["date"].year),
            "month":   int(row["date"].month),
            "article": row["article"],
            "url":     row["url"],
        },
    )
    for _, row in df.iterrows()
]
print(f"[1] Documents loaded  : {len(documents):,}")

# ── 2. Chunk ──────────────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=token_len,
    separators=["\n\n", "\n", "."],
)
chunks = splitter.split_documents(documents)
print(f"[2] Chunks created    : {len(chunks):,}  (avg {len(chunks)/len(documents):.1f} per doc)")

# ── 3. Embedders ──────────────────────────────────────────────────────────────
print(f"[3] Loading dense embedder  : {MODEL_ID} on CPU ...")
dense_embedder  = HuggingFaceEmbeddings(
    model_name=MODEL_ID,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
sparse_embedder = FastEmbedSparse(model_name="Qdrant/bm25")
print(f"    Sparse embedder : Qdrant/bm25 (BM25 via FastEmbed)")

# ── 4. Qdrant — build or reuse ────────────────────────────────────────────────
# Use filesystem check to avoid opening two QdrantClient instances on the same
# path simultaneously (Windows file lock prevents concurrent local access).
collection_dir = os.path.join(QDRANT_PATH, "collection", COLLECTION)
already_built  = os.path.isdir(collection_dir) and bool(os.listdir(collection_dir))

if already_built:
    print(f"[4] Collection '{COLLECTION}' found on disk — loading existing store.\n")
    client = QdrantClient(path=QDRANT_PATH)
    existing = client.get_collection(COLLECTION).points_count
    print(f"    Existing points: {existing:,}")
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=dense_embedder,
        sparse_embedding=sparse_embedder,
        retrieval_mode=RetrievalMode.HYBRID,
    )
else:
    # Delete any stale lock/partial data before building fresh
    if os.path.exists(QDRANT_PATH):
        import shutil
        shutil.rmtree(QDRANT_PATH)

    BATCH = 60
    print(f"[4] Building Qdrant collection '{COLLECTION}' ...")
    print(f"    {len(chunks):,} chunks  |  batch={BATCH}  |  device=CPU\n")
    t0 = time.time()

    # Step A — first batch creates the collection; keeps its internal client open
    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks[:BATCH],
        embedding=dense_embedder,
        sparse_embedding=sparse_embedder,
        retrieval_mode=RetrievalMode.HYBRID,
        collection_name=COLLECTION,
        path=QDRANT_PATH,
    )

    # Step B — reuse vectorstore's internal client for remaining batches (no new client opened)
    remaining = chunks[BATCH:]
    with tqdm(total=len(chunks), initial=BATCH, desc="Indexing", unit="chunk", ncols=80) as pbar:
        for i in range(0, len(remaining), BATCH):
            batch = remaining[i : i + BATCH]
            vectorstore.add_documents(batch)
            pbar.update(len(batch))

    elapsed = time.time() - t0
    count   = vectorstore._client.get_collection(COLLECTION).points_count
    print(f"\n    Done in {elapsed:.1f}s  —  {count:,} points stored\n")
    client  = vectorstore._client  # alias for close() at end

# # ── 5. Smoke test — hybrid similarity search ──────────────────────────────────
# print("─" * 60)
# print("[5] Hybrid search smoke test")
# print("─" * 60)
# query = "ما هي أحدث أخبار البنوك السعودية؟"
# print(f"    Query : {query}\n")

# results = vectorstore.similarity_search(query, k=3)
# for i, doc in enumerate(results):
#     print(f"  [{i+1}]  id={doc.metadata['id']}  |  {doc.metadata['date']}  |  {doc.metadata['title'][:60]}")
#     print(f"       {doc.page_content[:200].strip()} ...")
#     print()

client.close()  # release lock so query.py can open cleanly
