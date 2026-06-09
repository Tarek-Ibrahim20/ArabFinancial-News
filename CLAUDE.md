# ArabFinancial News — RAG Pipeline

Production Arabic financial news RAG app built on the
[AraFinNews](https://github.com/ArabicNLP-UK/AraFinNews) dataset (~10 k articles).

## Stack
| Layer | Choice |
|---|---|
| Embedder | `Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2` (HuggingFace, CPU) |
| Vector store | Qdrant (hybrid search — dense + sparse BM25) |
| Reranker | `miniReranker_arabic_v1` via FlashRank (local, top 5) |
| Splitter | `RecursiveCharacterTextSplitter` — 500 tokens, 50 overlap |
| LLM (local) | `qwen2.5:3b` via Ollama |
| Framework | LangChain |

## Config — `.env` keys
```
filepathurl        remote CSV source (AraFinNews GitHub raw)
csv_path           local preprocessed CSV
model_id           HuggingFace embedding model ID
chunk_size         tokens per chunk (500)
chunk_overlap      overlap tokens (50)
qdrant_path        local directory for Qdrant persistence
qdrant_collection  Qdrant collection name
LLM_model          Ollama model tag
```

## Pipeline status
- [x] **Data** — fetch & clean (`sample_dataset.py`)
- [x] **EDA** — token distribution analysis (`exploratory_tokens.py`)
- [x] **Ingest** — chunk, embed, store (`vectorstore.py`)
- [x] **Retrieval** — hybrid self-query + Arabic reranking (`retrieval.py`)
- [ ] **Augmentation** — prompt construction + LLM generation (`rag_chain.py`) ← **current**
- [ ] **Evaluation** — retrieval and generation quality metrics

## Files

| File | Stage | Description |
|---|---|---|
| `sample_dataset.py` | Data | Fetches 10 k rows from AraFinNews remote CSV. Cleans HTML entities, hidden Unicode/bidi chars, control chars, and normalises whitespace. Outputs `sample_dataset.csv` with a `text` column (`title \| article`) ready for embedding. |
| `exploratory_tokens.py` | EDA | Token-length analysis on the `text` column using tiktoken `cl100k_base`. Prints descriptive stats, percentile table, and model-fit summary. Saves `token_distribution.png` histogram to guide chunk-size selection. |
| `vectorstore.py` | Ingest | Full ingest pipeline: CSV → LangChain `Document` objects → recursive chunks (500 tok / 50 overlap) → `Arabic-Triplet-Matryoshka-V2` embeddings → Qdrant. Run once to build the store. |
| `retrieval.py` | **Retrieval** | **Production retrieval module.** Import and call `retrieve(user_query)`. Pipeline: LLM parses Arabic query → semantic text + year/month filters → Qdrant hybrid search (k=50 candidates) → FlashRank Arabic reranker → top 5 sorted by relevance score. All module-level state is private (`_`-prefixed). No side effects on import. |
| `query.py` | Retrieval experiment | Test/debug script. Mirrors `retrieval.py` logic but prints results to stdout. Use to manually inspect retrieval quality. Not imported by other modules. |
| `gg.py` | Retrieval experiment | Earlier scratch file. Superseded by `query.py`. |
| `rag_chain.py` | **Augmentation** | **Production RAG module (to build).** Import and call `answer(query)`. Pipeline: `retrieve(query)` → format Arabic context → Ollama LLM → return answer + sources. |
| `sample_dataset.csv` | Artifact | 10 k-row preprocessed dataset. Source of truth for `vectorstore.py`. |
| `token_distribution.png` | Artifact | Histogram of article token lengths with chunk-size reference lines. |
| `requirements.txt` | Config | Python package dependencies (includes `flashrank`). |
| `.env` | Config | All runtime config. Never commit. |
| `qdrant_db/` | Artifact | Persisted Qdrant vector store. Rebuilt by `vectorstore.py`. |

## Metadata schema (Qdrant payload)
Every chunk carries:
```
id      int     unique article ID
title   str     Arabic headline
date    str     YYYY-MM-DD  (for display)
year    int     publication year  — use for payload filtering
month   int     publication month — use for payload filtering
article str     full Arabic body text
url     str     argaam.com source URL (provenance)
```
Filter syntax: `FieldCondition(key="metadata.year", match=MatchValue(value=y))`
— `langchain_qdrant` nests metadata under a `"metadata"` payload key, so always
prefix field names with `"metadata."` in filter conditions.

## Retrieval design notes
- **Strip dates before reranking**: `parsed.search_query` (date-free) is passed to the
  reranker because the date constraint is already enforced by the Qdrant filter. Passing
  the full query would dilute the reranker's semantic signal.
- **English rerankers degrade quality on Arabic**: always use an Arabic or multilingual
  reranker model. Default FlashRank (`ms-marco-MiniLM-L-12-v2`) is English-only.
- **Public API**: `from retrieval import retrieve` — returns `list[Document]`, each with
  `relevance_score` added to metadata alongside the original fields.

## Hybrid search approach
- **Dense**: `Arabic-Triplet-Matryoshka-V2` embeddings (semantic similarity)
- **Sparse**: BM25 via FastEmbed (keyword/lexical match)
- **Fusion**: Reciprocal Rank Fusion (RRF) inside Qdrant

## Augmentation spec — `rag_chain.py`

### Public API
```python
from rag_chain import answer

result = answer("ما أخبار أسعار النفط في 2020؟")
# result = {"answer": "...", "sources": [{"title": ..., "date": ..., "url": ...}, ...]}
```

### Pipeline
```
user_query
  → retrieve(user_query)          # list[Document], top 5, sorted by relevance_score
  → format_context(docs)          # Arabic context block: title + date + content per doc
  → RAG prompt (system + context + question)
  → ChatOllama(model=LLM_model, temperature=0)
  → parse response → {"answer": str, "sources": list[dict]}
```

### Prompt template
```
System:
  أنت مساعد متخصص في الأخبار المالية العربية.
  أجب على السؤال بناءً على السياق المقدم فقط.
  إذا لم يكن الجواب في السياق، قل: "لا تتوفر معلومات كافية في قاعدة البيانات."
  أجب بالعربية الفصحى دائماً.

Context block (one entry per retrieved doc):
  المصدر {i}: {title} ({date})
  {page_content}
  ---
```

### Design notes
- `rag_chain.py` must **not** re-load embedder/Qdrant/reranker — call `retrieve()` from `retrieval.py`, which owns those.
- `LLM_model` comes from `.env` (same key used in `retrieval.py`).
- Use `temperature=0` for deterministic, factual answers.
- If `retrieve()` returns 0 docs, return a fixed Arabic no-data message without calling the LLM.
- Sources list = `[{"title": d.metadata["title"], "date": d.metadata["date"], "url": d.metadata["url"]} for d in docs]`.
- Keep all module-level state `_`-prefixed; no side effects on import.

## Next steps
1. **Augmentation** (`rag_chain.py`) — implement per spec above. ← **current**
2. **Evaluation** — retrieval metrics (MRR, Recall@k) and generation metrics (faithfulness, answer relevance via `ragas` or manual).
