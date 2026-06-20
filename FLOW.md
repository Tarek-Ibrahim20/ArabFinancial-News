# ArabFinancial News — End-to-End Flow

Arabic financial-news RAG pipeline from raw CSV to a cited answer served over HTTP.
Built on the [AraFinNews](https://github.com/ArabicNLP-UK/AraFinNews) dataset (~10k articles).

---

## 1. Top-level pipeline

Three phases: **build** runs once per dataset refresh, **serve** runs per user query, **eval** is an offline harness that reuses the serve path.

```mermaid
flowchart TB
    subgraph BUILD["OFFLINE — build once"]
        direction TB
        RAW["Remote CSV (AraFinNews)"]
        SD["sample_dataset.py"]
        CSV["sample_dataset.csv"]
        VS["vectorstore.py"]
        QD[("qdrant_db/")]
        RAW --> SD --> CSV --> VS --> QD
    end

    subgraph SERVE["ONLINE — per query"]
        direction LR
        Q["User query"] --> API["api.py"] --> RET["retrieval.py"] --> RAG["rag_chain.py"] --> OUT["Answer + sources"]
    end

    subgraph EVAL["OFFLINE — evaluation"]
        direction TB
        GOLD["golden_dataset.jsonl"]
        GEN["generate_outputs.py"]
        RMET["retrieval_metrics.py"]
        RAGAS["run_ragas.py"]
        GOLD --> GEN --> RMET & RAGAS
    end

    QD ==>|read| RET
    RAG -.answer_for_eval().-> GEN

    classDef store fill:#1f3a5f,stroke:#4a90d9,color:#fff
    classDef art fill:#2d4a22,stroke:#6aab4a,color:#fff
    class QD store
    class CSV,RAW art
```

The serve phase only *reads* `qdrant_db/` — it never re-embeds. Eval drives `answer_for_eval()` so scores reflect the exact production path.

---

## 2. Offline — Data → Vector store

### 2a. Data loading — `sample_dataset.py`

```mermaid
flowchart LR
    URL["filepathurl (.env)"] --> READ["pd.read_csv · 10k rows"]
    READ --> CLEAN["preprocess_text()\n· HTML unescape · strip tags\n· remove bidi/hidden Unicode\n· collapse whitespace"]
    CLEAN --> JOIN["text = title + ' | ' + article"]
    JOIN --> OUT["sample_dataset.csv"]

    classDef art fill:#2d4a22,stroke:#6aab4a,color:#fff
    class OUT art
```

### 2b. EDA — `exploratory_tokens.py`

Standalone token-length analysis (tiktoken `cl100k_base`) → `token_distribution.png`. Justifies `chunk_size=500 / overlap=50`. Does not feed the store.

### 2c. Ingest — `vectorstore.py`

```mermaid
flowchart TB
    CSV["sample_dataset.csv"] --> DOCS["LangChain Documents\nmetadata: id · title · date · year · month · url"]
    DOCS --> SPLIT["RecursiveCharacterTextSplitter\n500 tok / 50 overlap"]
    SPLIT --> CHUNKS["chunks"]

    CHUNKS --> DENSE["Arabic-Triplet-Matryoshka-V2\n(dense embeddings)"]
    CHUNKS --> SPARSE["Qdrant/bm25 FastEmbed\n(sparse BM25)"]

    DENSE --> QD[("Qdrant — HYBRID mode")]
    SPARSE --> QD

    QD --> CHECK{"collection\non disk?"}
    CHECK -->|yes| SKIP["reuse — skip re-embed"]
    CHECK -->|no| BUILD["build from_documents"]

    classDef store fill:#1f3a5f,stroke:#4a90d9,color:#fff
    class QD store
```

Metadata is nested under `"metadata."` in Qdrant payload — retrieval filters must prefix field names (e.g. `metadata.year`).

---

## 3. Online — Retrieval — `retrieval.py`

`retrieve(user_query, k_candidates=50) → list[Document]` (top 5, sorted by rerank score). All state initialized **once on import**.

```mermaid
flowchart TB
    Q["user_query (Arabic)"]

    Q --> C1["TemporalFilter chain\n→ year? month?"]
    Q --> C2["SemanticQuery chain\n→ MSA rewrite (date-free)\n+ BM25 keywords"]

    C1 --> FILT["Qdrant filter\nmetadata.year / month"]
    C2 --> KW["Arabic-only keywords\nappended to search_query"]

    FILT --> SEARCH["Qdrant hybrid search\ndense + BM25 → RRF · k=50"]
    KW --> SEARCH
    QD[("qdrant_db/")] ==> SEARCH

    SEARCH --> RERANK["FlashRank miniReranker_arabic_v1\nquery = date-free rewrite · top 5"]
    RERANK --> OUT["top 5 Documents + relevance_score"]

    C1 & C2 -.LLM fails.-> FB["Fallback: plain hybrid\nsearch on raw query"]
    FB --> RERANK

    classDef store fill:#1f3a5f,stroke:#4a90d9,color:#fff
    class QD store
```

Dates are stripped before reranking — the date constraint is already enforced by the Qdrant filter, so passing them dilutes the semantic signal.

---

## 4. Online — Augmentation — `rag_chain.py`

`answer(query) → {answer, sources}` and `answer_stream(query)` (token generator). Calls `retrieve()` — never loads the retrieval stack directly.

```mermaid
flowchart TB
    Q["user_query"] --> RET["retrieve(query) · top 5 docs"]
    RET --> EMPTY{"0 docs?"}
    EMPTY -->|yes| NODATA["Arabic no-data message\n(no LLM call)"]
    EMPTY -->|no| DEDUP["_deduplicate\ndrop identical chunks"]

    DEDUP --> USHAPE["_u_shape\nLost-in-the-Middle reorder\n[d1,d3,d5,d4,d2]"]
    USHAPE --> CTX["_format_context\nnumbered [N] blocks"]

    CTX --> LLM["qwen2.5:7b · temp=0 · num_ctx=5000\nEnglish rules · Arabic output · one-shot"]
    LLM --> BODY["answer body with inline [N] citations"]

    BODY --> REF["_format_references\nparse [N] → docs[n-1].metadata\n→ ## المراجع (deterministic)"]
    REF --> OUT["{answer, sources}"]

    classDef warn fill:#5f3a1f,stroke:#d99a4a,color:#fff
    class NODATA warn
```

`[N]` numbering is bound to the post-`_u_shape` order. `_format_references` renders the bibliography from metadata — URLs are exact and never hallucinated.

---

## 5. Online — Serving — `api.py`

FastAPI HTTP layer. Pipeline loads once at startup; all requests are serialized behind a lock.

```mermaid
flowchart TB
    subgraph STARTUP["Startup (once ~2-5s)"]
        S1["import rag_chain"] --> S2["embedder + Qdrant + reranker → RAM"]
        S2 --> S3["warm-up query → Ollama weights → RAM"]
        S3 --> S4["/health → 200 ready"]
    end

    subgraph REQUEST["Per request"]
        R1["POST /ask or /ask/stream"] --> R2{"pipeline\nready?"}
        R2 -->|no| E503["503 Not Ready"]
        R2 -->|yes| R3["asyncio.wait_for · 90s timeout"]
        R3 --> R4["threadpool + threading.Lock\n(one call at a time)"]
        R4 --> R5["answer() or answer_stream()"]
        R5 --> R6["_log_query → logs/rag_requests.jsonl"]
        R6 --> R7["JSON or plain-text stream"]
        R3 -.timeout.-> E504["504 Gateway Timeout"]
    end

    STARTUP --> REQUEST
```

`--workers 1` is required — multiple workers would each load the full pipeline and contend on the Qdrant file lock.

---

## 6. Offline — Evaluation — `eval/`

Generate once, score twice. `answer_for_eval()` returns the same generation plus `contexts` and `retrieved_ids` for metrics.

```mermaid
flowchart TB
    BG["build_golden.py\nqwen2.5:7b drafts\nstratified year/month"] --> GOLD["golden_dataset.jsonl\nquestion · type · ground_truth\nreference_article_ids · verified"]

    GOLD --> VER{"verified=true?"}
    VER -.gate.-> GEN

    GEN["generate_outputs.py\nanswer_for_eval() per Q\nRAG runs ONCE"] --> DUMP["pipeline_outputs_latest.jsonl\nanswer · contexts · retrieved_ids"]

    DUMP --> RMET["retrieval_metrics.py\ndeterministic · no LLM\nretrieved_ids vs reference_ids"]
    DUMP --> RAGAS["run_ragas.py\nqwen2.5:7b judge\nstrip المراجع first"]

    RMET --> RES["eval/results/\nHit@k · Recall@k · MRR"]
    RAGAS --> RES2["eval/results/\nfaithfulness · correctness"]

    classDef art fill:#2d4a22,stroke:#6aab4a,color:#fff
    classDef warn fill:#5f3a1f,stroke:#d99a4a,color:#fff
    class DUMP,GOLD art
    class VER warn
```

```bash
python eval/generate_outputs.py    # 1. slow — RAG once → shared dump
python eval/retrieval_metrics.py   # 2. instant — no model load
python eval/run_ragas.py           # 3. slow — Ragas judge only
```

Known finding: `comparison`/`multi_hop` bottleneck at Recall@5 ≈ 0.5 — single-query retrieval fetches one of two needed articles. Fix = query decomposition.

---

## 7. Module dependency graph

```mermaid
flowchart LR
    ENV[".env"]

    SD["sample_dataset.py"] --> CSV["sample_dataset.csv"]
    CSV --> VSP["vectorstore.py"] --> QD[("qdrant_db/")]

    QD --> RET["retrieval.py\nowns embedders + Qdrant + reranker"]
    RET --> RAGC["rag_chain.py"]
    RAGC --> API["api.py"]

    RAGC --> GEN["eval/generate_outputs.py"]
    BG["eval/build_golden.py"] --> GLD["golden_dataset.jsonl"]
    GLD --> GEN --> DUMP["pipeline_outputs_latest.jsonl"]
    DUMP --> RM["eval/retrieval_metrics.py"]
    DUMP --> RG["eval/run_ragas.py"]

    ENV -.config.-> SD & VSP & RET & RAGC

    classDef store fill:#1f3a5f,stroke:#4a90d9,color:#fff
    classDef art fill:#2d4a22,stroke:#6aab4a,color:#fff
    class QD store
    class CSV,GLD,DUMP art
```

`retrieval.py` is the only module that loads the embedders, Qdrant client, and reranker. Everything else goes through `retrieve()` or `answer*()`.

---

## Stage → file → artifact

| Stage | File | Reads | Produces |
|---|---|---|---|
| Data | `sample_dataset.py` | remote CSV | `sample_dataset.csv` |
| EDA | `exploratory_tokens.py` | `sample_dataset.csv` | `token_distribution.png` |
| Ingest | `vectorstore.py` | `sample_dataset.csv` | `qdrant_db/` |
| Retrieval | `retrieval.py` | `qdrant_db/` | `list[Document]` (top 5) |
| Augmentation | `rag_chain.py` | `retrieve()` | `{answer, sources}` |
| Serving | `api.py` | `answer()` / `answer_stream()` | HTTP response / stream |
| Golden set | `eval/build_golden.py` | `sample_dataset.csv` | `golden_dataset.jsonl` |
| Generate | `eval/generate_outputs.py` | golden + `answer_for_eval` | `pipeline_outputs_latest.jsonl` |
| Retrieval metrics | `eval/retrieval_metrics.py` | dump | `eval/results/` |
| Generation metrics | `eval/run_ragas.py` | dump | `eval/results/` |
