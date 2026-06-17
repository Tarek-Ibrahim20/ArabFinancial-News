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
| LLM (local) | `qwen2.5:7b` via Ollama |
| Framework | LangChain (v1.x) |
| Eval | Ragas 0.4.3 (local `qwen2.5:7b` judge) + deterministic retrieval metrics |

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
- [x] **Augmentation** — dedup → U-shape → inline-citation generation (`rag_chain.py`)
- [ ] **Evaluation** — golden set + retrieval & generation metrics (`eval/`) ← **current**

## Files

| File | Stage | Description |
|---|---|---|
| `sample_dataset.py` | Data | Fetches 10 k rows from AraFinNews remote CSV. Cleans HTML entities, hidden Unicode/bidi chars, control chars, and normalises whitespace. Outputs `sample_dataset.csv` with a `text` column (`title \| article`) ready for embedding. |
| `exploratory_tokens.py` | EDA | Token-length analysis on the `text` column using tiktoken `cl100k_base`. Prints descriptive stats, percentile table, and model-fit summary. Saves `token_distribution.png` histogram to guide chunk-size selection. |
| `vectorstore.py` | Ingest | Full ingest pipeline: CSV → LangChain `Document` objects → recursive chunks (500 tok / 50 overlap) → `Arabic-Triplet-Matryoshka-V2` embeddings → Qdrant. Run once to build the store. |
| `retrieval.py` | **Retrieval** | **Production retrieval module.** Import and call `retrieve(user_query)`. Pipeline: LLM parses Arabic query → semantic text + year/month filters → Qdrant hybrid search (k=50 candidates) → FlashRank Arabic reranker → top 5 sorted by relevance score. All module-level state is private (`_`-prefixed). No side effects on import. |
| `query.py` | Retrieval experiment | Test/debug script. Mirrors `retrieval.py` logic but prints results to stdout. Use to manually inspect retrieval quality. Not imported by other modules. |
| `gg.py` | Retrieval experiment | Earlier scratch file. Superseded by `query.py`. |
| `rag_chain.py` | **Augmentation** | **Production RAG module.** `answer(query) → {answer, sources}`. Pipeline: `retrieve` → `_deduplicate` (content-hash) → `_u_shape` (Lost-in-the-Middle reorder) → `_format_context` (numbered source blocks) → Ollama LLM (inline `[N]` citations) → `_format_references` (code-rendered `## المراجع`). Also exposes `answer_for_eval(query) → {answer, contexts, sources, retrieved_ids}` for the eval harness — `retrieved_ids` is the **raw `retrieve()` ranking** (captured before dedup/U-shape), `contexts` are post-U-shape. |
| `eval/build_golden.py` | **Eval** | Generates the golden dataset (drafts) via `qwen2.5:7b`. Stratified (year/month) article sampling; keyword-pairs articles for `comparison`/`multi_hop`. Writes `eval/golden_dataset.jsonl`. Drafts must be **human-verified** (`"verified": true`) before trustworthy use. |
| `eval/generate_outputs.py` | **Eval** | **Single generation step — run FIRST.** Calls `answer_for_eval` once per golden question and writes the shared dump (`pipeline_outputs_<stamp>.jsonl` + stable `pipeline_outputs_latest.jsonl`). Both metric scripts read this dump, so the RAG pipeline runs **once** for the whole eval. Config via module-level constants (`_VERIFIED_ONLY`, `_LIMIT`). |
| `eval/retrieval_metrics.py` | **Eval** | Deterministic retrieval scoring (no LLM judge). **Reads the shared dump** (`pipeline_outputs_latest.jsonl`), compares each record's `retrieved_ids` to `reference_article_ids` → Hit@k, Recall@k, Precision@k, MRR, broken down by `question_type`. Does **not** import the retrieval stack. Errors if the dump is missing. Report → `eval/results/`. |
| `eval/run_ragas.py` | **Eval** | Generation-quality scoring via Ragas. **Reads the shared dump** (no regeneration) and scores faithfulness / context_precision / context_recall / answer_correctness with the local `qwen2.5:7b` judge (`answer_relevancy` available but off by default — low-trust on Arabic). Errors if the dump is missing. Report → `eval/results/`. |
| `sample_dataset.csv` | Artifact | 10 k-row preprocessed dataset. Source of truth for `vectorstore.py`. |
| `token_distribution.png` | Artifact | Histogram of article token lengths with chunk-size reference lines. |
| `requirements.txt` | Config | Python package dependencies (includes `flashrank`, `ragas`, `datasets`). |
| `.env` | Config | All runtime config. Never commit. |
| `qdrant_db/` | Artifact | Persisted Qdrant vector store. Rebuilt by `vectorstore.py`. |
| `eval/golden_dataset.jsonl` | Artifact | Golden eval set. One JSON record/line: `question`, `question_type`, `ground_truth`, `reference_article_ids`, `reference_contexts`, `verified`. |
| `eval/results/` | Artifact | Timestamped metric reports + pipeline-output dumps. `pipeline_outputs_latest.jsonl` is the stable shared dump both metric scripts read. |

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
from rag_chain import answer, answer_for_eval

answer("ما أخبار أسعار النفط في 2020؟")
# {"answer": "... [1] ... [2]\n\n## المراجع\n1. url (date)", "sources": [{"title","date","url"}, ...]}

answer_for_eval("...")  # same generation + eval inputs:
# {"answer", "contexts": [str], "sources": [dict], "retrieved_ids": [int]}
```

### Pipeline (as built)
```
user_query
  → retrieve(user_query)          # list[Document], top 5, sorted by relevance_score
  → _deduplicate(docs)            # drop exact dup chunks (hash of page_content, not id)
  → _u_shape(docs)                # Lost-in-the-Middle reorder if >2: [d1,d3,d5,d4,d2]
  → _format_context(docs)         # numbered source blocks:  [i]: (date) | url \n content
  → ChatOllama(LLM_model, temperature=0, num_ctx=5000)
  → body with inline [N] citations
  → _format_references(body,docs) # code-rendered ## المراجع from cited [N] (no hallucination)
  → {"answer", "sources"}
```

### Prompt design (as built — diverged from the original Arabic-only spec)
- **System = English numbered RULES, Arabic output.** A moderate 7B follows explicit
  English instructions more reliably than Arabic ones, while still answering in فصحى.
- **Summarization framing**, not QA ("Summarize what the sources say...") — QA framing made
  the model over-trigger the no-data fallback.
- **One-shot example** (human sources + ai answer) teaches the exact inline-`[N]` format.
- Context lives in the **human** turn, not the system message.

### Design notes
- `rag_chain.py` must **not** re-load embedder/Qdrant/reranker — call `retrieve()` from `retrieval.py`, which owns those.
- **Citations are split by responsibility**: the LLM emits inline `[N]` markers (semantic
  judgment); `_format_references` renders the `## المراجع` bibliography deterministically by
  looking up `docs[n-1].metadata` — so URLs are exact and never hallucinated. The `[N]`
  numbering is bound to the post-`_u_shape` order; never reorder `docs` between
  `_format_context` and `_format_references`.
- `num_ctx` must be large enough for 5 Arabic chunks (default 2048 silently truncates → fallback).
- If `retrieve()` returns 0 docs, return the fixed Arabic no-data message without calling the LLM.
- Keep all module-level state `_`-prefixed; no side effects on import.

## Evaluation — `eval/`

### Golden dataset (`eval/golden_dataset.jsonl`)
One JSON record per line:
```jsonc
{
  "question": "...", "question_type": "factual|comparison|explanation|multi_hop",
  "ground_truth": "...", "reference_article_ids": [int], "reference_contexts": [str],
  "verified": false   // human-review gate — flip to true before trusting scores
}
```
Built by `build_golden.py` (LLM-drafted via `qwen2.5:7b`), then **human-verified**.

### Workflow — generate once, then score
The RAG pipeline runs **once** for the whole eval. `generate_outputs.py` calls
`answer_for_eval` per question and writes a shared dump; both metric scripts read it (no
regeneration, no drift between reports):
```
python eval/generate_outputs.py    # 1. run RAG once -> pipeline_outputs_latest.jsonl  (slow)
python eval/retrieval_metrics.py   # 2. instant: reads dump, no model load
python eval/run_ragas.py           # 3. slow: Ragas judge only, no regeneration
```
Set `_VERIFIED_ONLY` / `_LIMIT` in `generate_outputs.py` to control which records enter the
dump. The consumers score whatever is in the dump.

### Two metric families
- **Retrieval (deterministic, no judge)** — `retrieval_metrics.py`: Hit@k, Recall@k,
  Precision@k, MRR vs `reference_article_ids` (from the dump's raw `retrieved_ids`), broken
  down by `question_type`.
- **Generation (LLM-judged via Ragas)** — `run_ragas.py`: faithfulness,
  context_precision, context_recall, answer_correctness (answer_relevancy off by default).

### Eval notes / caveats
- **Ragas ↔ langchain-1.x shim**: ragas 0.4.3 imports the removed
  `langchain_community.chat_models.vertexai`; `run_ragas.py` stubs that module before
  `import ragas`. Unused (we judge with Ollama, not VertexAI).
- **Judge == generator (`qwen2.5:7b`)** — cheap but self-grading; treat absolute numbers as
  directional, trust by-type deltas and run-over-run trends. Swap via the `_JUDGE_MODEL`
  constant in `run_ragas.py` (the eval scripts are config-by-constant, no CLI args).
- **`keep_alive="1h"` on the judge is mandatory.** Ragas' serial calls gap longer than
  Ollama's default 5-min keep_alive, so the 7B gets evicted and reloaded from disk every
  call → a 2-question run ballooned to 8h of all-NaN timeouts. Pinning it = ~12 min/question.
- **Strip the `## المراجع` block before scoring** (`_strip_references`). The bibliography is
  code-rendered metadata, not LLM claims; judging it tanks faithfulness/correctness unfairly
  (answer_correctness 0.67 → 0.84 once stripped).
- **`answer_relevancy` is low-trust on Arabic** (~0.31 across runs, unaffected by fixes). Its
  reverse-question-generation step degrades on Arabic; rely on faithfulness + correctness. It
  is **off by default** — add it back to `_METRIC_NAMES` in `run_ragas.py` if needed.
- **Speed**: ~12 min/question with a local 7B judge (serial). The default 4 metrics over the
  full 12-record set runs for hours — run in the background. The slow part is `run_ragas.py`;
  generation (`generate_outputs.py`) happens once and `retrieval_metrics.py` is instant.
- **Known finding**: multi-source questions (`comparison`/`multi_hop`) bottleneck at
  Recall@5 ≈ 0.5 — single-query retrieval fetches one of two needed articles. Fix path =
  query decomposition, not prompt tuning.

## Next steps
1. **Human-verify** the golden drafts (`"verified": true`), then set `_VERIFIED_ONLY = True` in
   `generate_outputs.py`, regenerate the dump, and re-run both metric scripts.
2. Iterate on retrieval (query decomposition for multi-hop) and chunking/prompt based on scores.
