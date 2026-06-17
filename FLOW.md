# ArabFinancial News — End-to-End Flow

A walkthrough of the Arabic financial-news RAG pipeline, from raw CSV to a
cited Arabic answer and evaluation scores. Every stage below maps to a real
file in this repo (see the per-stage notes).

> Built on the [AraFinNews](https://github.com/ArabicNLP-UK/AraFinNews)
> dataset (~10k articles). Local-only stack: HuggingFace embeddings + Qdrant
> hybrid search + FlashRank Arabic reranker + `qwen2.5:7b` via Ollama.

---

## 1. Top-level pipeline

Two phases. The **offline / build** phase runs rarely (once per dataset
refresh); the **online / serve** phase runs per user query. Evaluation is an
offline harness that reuses the serve path.

```mermaid
flowchart TB
    subgraph BUILD["🔨 OFFLINE — build once"]
        direction TB
        RAW["Remote CSV<br/>(AraFinNews raw)"]
        SD["sample_dataset.py<br/>fetch + clean"]
        CSV["sample_dataset.csv<br/>10k rows · text column"]
        EDA["exploratory_tokens.py<br/>token analysis → chunk size"]
        VS["vectorstore.py<br/>chunk · embed · index"]
        QD[("qdrant_db/<br/>hybrid vector store")]

        RAW --> SD --> CSV
        CSV -.guides.-> EDA
        EDA -.chunk_size=500.-> VS
        CSV --> VS --> QD
    end

    subgraph SERVE["⚡ ONLINE — per query"]
        direction TB
        Q["User query (Arabic)"]
        RET["retrieval.py<br/>retrieve()"]
        RAG["rag_chain.py<br/>answer()"]
        OUT["Answer + المراجع<br/>+ sources[]"]

        Q --> RET --> RAG --> OUT
    end

    subgraph EVAL["📊 OFFLINE — evaluation"]
        direction TB
        GOLD["eval/build_golden.py<br/>→ golden_dataset.jsonl"]
        GEN["eval/generate_outputs.py<br/>run RAG once → dump"]
        RMET["eval/retrieval_metrics.py<br/>Hit/Recall/MRR"]
        RAGAS["eval/run_ragas.py<br/>faithfulness/correctness"]

        GOLD --> GEN
        GEN --> RMET
        GEN --> RAGAS
    end

    QD ==>|read| RET
    RAG -.answer_for_eval().-> GEN

    classDef store fill:#1f3a5f,stroke:#4a90d9,color:#fff
    classDef art fill:#2d4a22,stroke:#6aab4a,color:#fff
    class QD store
    class CSV,RAW art
```

**Key dependency:** the serve phase only *reads* `qdrant_db/`. It never
re-embeds or re-indexes. The eval phase drives the same `rag_chain` entry point
(`answer_for_eval`) so scores reflect the production path exactly.

---

## 2. Offline — Data → Vector store

### 2a. Data loading & preprocessing — `sample_dataset.py`

```mermaid
flowchart LR
    URL["filepathurl<br/>(.env)"] --> READ["pd.read_csv<br/>nrows=10000<br/>utf-8, replace errors<br/>parse_dates"]
    READ --> STRUCT["Structural clean<br/>· strip col names<br/>· drop empty rows<br/>· Int64 / string dtypes"]
    STRUCT --> CONTENT["preprocess_text()<br/>on title + article"]
    CONTENT --> DETAIL["· HTML unescape<br/>· strip tags<br/>· remove hidden/bidi Unicode<br/>· drop control chars<br/>· collapse whitespace"]
    DETAIL --> JOIN["text = title + ' | ' + article"]
    JOIN --> OUT["sample_dataset.csv<br/>utf-8-sig"]

    classDef art fill:#2d4a22,stroke:#6aab4a,color:#fff
    class OUT art
```

The `text` column (`title | article`) is what gets embedded. Cleaning targets
the noise specific to scraped Arabic web text: HTML entities, zero-width and
bidi-override characters, and control bytes.

### 2b. EDA (advisory) — `exploratory_tokens.py`

Standalone token-length analysis on the `text` column (tiktoken `cl100k_base`).
Prints percentile stats and saves `token_distribution.png`. It does **not**
feed the store — it justifies the `chunk_size=500 / overlap=50` choice.

### 2c. Ingest — `vectorstore.py`

```mermaid
flowchart TB
    CSV["sample_dataset.csv"] --> DOCS["LangChain Documents<br/>page_content = text<br/>metadata = id,title,date,<br/>year,month,article,url"]
    DOCS --> SPLIT["RecursiveCharacterTextSplitter<br/>500 tok / 50 overlap<br/>token_len via model tokenizer<br/>separators: ¶ · newline · period"]
    SPLIT --> CHUNKS["chunks (~N per doc)"]

    CHUNKS --> DENSE["Dense embedder<br/>Arabic-Triplet-Matryoshka-V2<br/>CPU · normalized"]
    CHUNKS --> SPARSE["Sparse embedder<br/>Qdrant/bm25 (FastEmbed)"]

    DENSE --> QD[("Qdrant collection<br/>HYBRID mode<br/>batched x60")]
    SPARSE --> QD

    QD --> REUSE{"collection<br/>on disk?"}
    REUSE -->|yes| LOAD["load existing<br/>(skip re-embed)"]
    REUSE -->|no| BUILD["build fresh<br/>from_documents"]

    classDef store fill:#1f3a5f,stroke:#4a90d9,color:#fff
    class QD store
```

Each chunk carries the full metadata payload. `langchain_qdrant` nests it under
a `"metadata."` payload key — which is why retrieval filters must prefix field
names (`metadata.year`). Idempotent: a populated collection on disk is reused,
not rebuilt.

---

## 3. Online — Retrieval — `retrieval.py`

Public API: `retrieve(user_query, k_candidates=50) -> list[Document]` (top 5,
sorted by rerank score). Module state (embedders, Qdrant client, reranker) is
initialized **once on import** and `_`-prefixed.

```mermaid
flowchart TB
    Q["user_query (Arabic)"]

    Q --> C1["Chain 1 — TemporalFilter<br/>LLM structured output<br/>→ year? month?"]
    Q --> C2["Chain 2 — SemanticQuery<br/>LLM structured output<br/>→ MSA rewrite (date-free)<br/>+ Arabic BM25 keywords"]

    C1 --> FILT["Build Qdrant filter<br/>FieldCondition metadata.year/month"]
    C2 --> KW["Keep Arabic-only keywords<br/>append to search_query"]

    FILT --> SEARCH
    KW --> SEARCH["Qdrant hybrid search<br/>dense + BM25 → RRF fusion<br/>k=50 candidates"]
    QD[("qdrant_db/")] ==> SEARCH

    SEARCH --> RERANK["FlashRank reranker<br/>miniReranker_arabic_v1<br/>query = date-free rewrite<br/>top_n=5 · score ≥ 0.5"]
    RERANK --> SORT["sort by relevance_score desc"]
    SORT --> OUT["top 5 Documents<br/>+ relevance_score in metadata"]

    C1 -.LLM fails.-> FB["Fallback:<br/>plain hybrid search<br/>on raw query"]
    C2 -.LLM fails.-> FB
    FB --> RERANK

    classDef store fill:#1f3a5f,stroke:#4a90d9,color:#fff
    class QD store
```

**Design notes baked into the code**
- **Dates stripped before reranking.** The reranker receives the date-free
  `search_query`; the date constraint is already enforced by the Qdrant filter,
  so passing dates would dilute the semantic signal.
- **Arabic reranker is mandatory.** English rerankers degrade on Arabic.
- **Self-query split into two focused chains** (temporal vs. semantic) — each is
  more reliable than one combined extraction on a moderate 7B.
- **Graceful fallback:** any LLM parse error drops to plain hybrid search.

---

## 4. Online — Augmentation / Generation — `rag_chain.py`

Public API: `answer(query) -> {answer, sources}`. Does **not** re-load the
retrieval stack — it calls `retrieve()`.

```mermaid
flowchart TB
    Q["user_query"] --> RET["retrieve(query)<br/>top 5 docs"]
    RET --> EMPTY{"0 docs?"}
    EMPTY -->|yes| NODATA["return fixed Arabic<br/>no-data message<br/>(no LLM call)"]
    EMPTY -->|no| DEDUP["_deduplicate<br/>drop identical page_content<br/>(RRF surfaces dup chunks)"]

    DEDUP --> USHAPE["_u_shape<br/>Lost-in-the-Middle reorder<br/>[d1,d3,d5,d4,d2]"]
    USHAPE --> CTX["_format_context<br/>numbered [N] blocks:<br/>(date) | url + content"]

    CTX --> LLM["ChatOllama qwen2.5:7b<br/>temp=0 · num_ctx=5000<br/>English RULES · Arabic out<br/>summarization + one-shot"]
    LLM --> BODY["answer body<br/>inline [N] citations"]

    BODY --> REF["_format_references<br/>parse cited [N] → look up<br/>docs[n-1].metadata<br/>render ## المراجع"]
    REF --> RESULT["{answer, sources}"]

    classDef warn fill:#5f3a1f,stroke:#d99a4a,color:#fff
    class NODATA warn
```

**Why it's built this way**
- **Citations split by responsibility.** The LLM emits the semantic `[N]`
  markers; `_format_references` renders the bibliography *deterministically*
  from metadata, so URLs are exact and never hallucinated.
- **Ordering is load-bearing.** `[N]` numbering binds to the post-`_u_shape`
  order — docs must not be reordered between `_format_context` and
  `_format_references`.
- **`num_ctx=5000`** is sized for 5 Arabic chunks; the default 2048 silently
  truncates and trips the no-data fallback.
- **Prompt diverged from the original Arabic-only spec:** English numbered
  rules + summarization framing + one-shot example produce more reliable inline
  citations from a 7B than Arabic instructions did.

---

## 5. Offline — Evaluation — `eval/`

The eval harness reuses the serve path via `answer_for_eval()`, which returns
the same generation plus the inputs metrics need (`contexts`, `retrieved_ids`).
The core principle: **generate once, score twice.**

```mermaid
flowchart TB
    BUILD["build_golden.py<br/>qwen2.5:7b drafts<br/>stratified by year/month<br/>keyword-paired for<br/>comparison/multi_hop"]
    BUILD --> GOLD["golden_dataset.jsonl<br/>question · type · ground_truth<br/>reference_article_ids · verified"]

    GOLD --> VERIFY{"human-verified?<br/>verified=true"}
    VERIFY -.gate.-> GEN

    GEN["generate_outputs.py<br/>answer_for_eval() per Q<br/>RAG runs ONCE"]
    GEN --> DUMP["pipeline_outputs_latest.jsonl<br/>answer · contexts<br/>retrieved_ids · sources"]

    DUMP --> RMET["retrieval_metrics.py<br/>deterministic · no LLM<br/>retrieved_ids vs reference_ids"]
    DUMP --> RAGAS["run_ragas.py<br/>qwen2.5:7b judge<br/>strip المراجع first"]

    RMET --> R1["Hit@k · Recall@k<br/>Precision@k · MRR<br/>by question_type"]
    RAGAS --> R2["faithfulness · answer_correctness<br/>context_precision · context_recall"]

    R1 --> RES["eval/results/"]
    R2 --> RES

    classDef art fill:#2d4a22,stroke:#6aab4a,color:#fff
    classDef warn fill:#5f3a1f,stroke:#d99a4a,color:#fff
    class DUMP,GOLD art
    class VERIFY warn
```

**Run order**
```bash
python eval/generate_outputs.py    # 1. slow — RAG once → shared dump
python eval/retrieval_metrics.py   # 2. instant — no model load
python eval/run_ragas.py           # 3. slow — Ragas judge only
```

**Caveats that shape the design**
- **One dump, two consumers** — both metric scripts read
  `pipeline_outputs_latest.jsonl`, so reports never drift and the pipeline runs
  once per eval.
- **Judge == generator** (`qwen2.5:7b`): self-grading, so trust by-type deltas
  and run-over-run trends, not absolute numbers.
- **Strip `## المراجع` before judging** — it's code-rendered metadata, not LLM
  claims; scoring it unfairly tanks faithfulness/correctness.
- **`keep_alive="1h"` on the judge is mandatory** — Ragas' serial gaps exceed
  Ollama's default 5-min keep-alive, causing reload-from-disk timeouts.
- **`answer_relevancy` off by default** — its reverse-question step is low-trust
  on Arabic (~0.31).
- **Known finding:** `comparison`/`multi_hop` bottleneck at Recall@5 ≈ 0.5 —
  single-query retrieval fetches one of two needed articles. Fix path = query
  decomposition, not prompt tuning.

---

## 6. Module dependency graph

Who imports whom. Note the single ownership of the retrieval stack.

```mermaid
flowchart LR
    ENV[".env"]

    SD["sample_dataset.py"] --> CSV["sample_dataset.csv"]
    CSV --> VSP["vectorstore.py"]
    VSP --> QD[("qdrant_db/")]

    RET["retrieval.py<br/>owns embedders+Qdrant+reranker"]
    QD --> RET
    RET --> RAGC["rag_chain.py"]

    RAGC --> GEN["eval/generate_outputs.py"]
    BG["eval/build_golden.py"] --> GLD["golden_dataset.jsonl"]
    GLD --> GEN
    GEN --> DUMP["pipeline_outputs_latest.jsonl"]
    DUMP --> RM["eval/retrieval_metrics.py"]
    DUMP --> RG["eval/run_ragas.py"]

    ENV -.config.-> SD
    ENV -.config.-> VSP
    ENV -.config.-> RET
    ENV -.config.-> RAGC

    classDef store fill:#1f3a5f,stroke:#4a90d9,color:#fff
    classDef art fill:#2d4a22,stroke:#6aab4a,color:#fff
    class QD store
    class CSV,GLD,DUMP art
```

**Single-ownership rule:** `retrieval.py` is the *only* module that loads the
embedders, Qdrant client, and reranker. `rag_chain.py` and the eval scripts go
through `retrieve()` / `answer*()` — never the heavy stack directly. Everything
is configured from `.env` and has no import-time side effects beyond
`retrieval.py`'s one-time stack init.

---

## Stage → file → artifact summary

| Stage | File | Reads | Produces |
|---|---|---|---|
| Data | `sample_dataset.py` | remote CSV (`.env`) | `sample_dataset.csv` |
| EDA | `exploratory_tokens.py` | `sample_dataset.csv` | `token_distribution.png` |
| Ingest | `vectorstore.py` | `sample_dataset.csv` | `qdrant_db/` |
| Retrieval | `retrieval.py` | `qdrant_db/` | `list[Document]` (top 5) |
| Augmentation | `rag_chain.py` | `retrieve()` | `{answer, sources}` |
| Golden set | `eval/build_golden.py` | `sample_dataset.csv` | `golden_dataset.jsonl` |
| Generate | `eval/generate_outputs.py` | golden + `answer_for_eval` | `pipeline_outputs_latest.jsonl` |
| Retrieval metrics | `eval/retrieval_metrics.py` | dump | `eval/results/` |
| Generation metrics | `eval/run_ragas.py` | dump | `eval/results/` |
