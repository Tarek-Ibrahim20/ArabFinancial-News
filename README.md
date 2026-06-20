# ArabFinancial News — RAG Pipeline

A production Arabic financial-news **Retrieval-Augmented Generation** app built on the
[AraFinNews](https://github.com/ArabicNLP-UK/AraFinNews) dataset (~10 k articles). It
answers Arabic questions about financial news, grounded in retrieved source articles, with
inline `[N]` citations and a `## المراجع` bibliography — served over HTTP via FastAPI.

---

## Architecture

```
question
  → retrieve()        hybrid search (dense + BM25) → Arabic rerank → top 5
  → deduplicate       drop exact duplicate chunks
  → U-shape reorder   mitigate "Lost in the Middle"
  → format context    numbered source blocks
  → qwen2.5:7b (Ollama)  Arabic answer with inline [N] citations
  → references         deterministic ## المراجع from cited sources
  → {answer, sources}
```

### Stack

| Layer | Choice |
|---|---|
| Embedder | `Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2` (HuggingFace, CPU) |
| Vector store | Qdrant (hybrid — dense + sparse BM25, RRF fusion) |
| Reranker | `miniReranker_arabic_v1` via FlashRank (local, top 5) |
| Splitter | `RecursiveCharacterTextSplitter` — 500 tokens, 50 overlap |
| LLM | `qwen2.5:7b` via Ollama |
| Framework | LangChain (v1.x) |
| Serving | FastAPI + Uvicorn |
| Eval | Ragas 0.4.3 (local `qwen2.5:7b` judge) + deterministic retrieval metrics |

---

## Project layout

| File | Stage | Description |
|---|---|---|
| `sample_dataset.py` | Data | Fetches & cleans 10 k rows → `sample_dataset.csv`. |
| `exploratory_tokens.py` | EDA | Token-length analysis → `token_distribution.png`. |
| `vectorstore.py` | Ingest | CSV → chunks → embeddings → Qdrant. Run once to build the store. |
| `retrieval.py` | Retrieval | `retrieve(query)` — hybrid self-query search + Arabic reranking. |
| `rag_chain.py` | Augmentation | `answer(query) → {answer, sources}` — full RAG generation. |
| `api.py` | **Serving** | FastAPI HTTP layer (`POST /ask`, `GET /health`). |
| `eval/` | Evaluation | Golden dataset + retrieval & Ragas generation metrics. |

See [`CLAUDE.md`](./CLAUDE.md) for the full design notes, metadata schema, and eval caveats.

---

## Setup

### Prerequisites
- **Python 3.10+**
- **[Ollama](https://ollama.com)** running locally, with the model pulled:
  ```bash
  ollama pull qwen2.5:7b
  ```
- The **Qdrant store built** (`qdrant_db/`). If it doesn't exist yet:
  ```bash
  python vectorstore.py
  ```

### Install
```bash
pip install -r requirements.txt
```

### Configure — `.env`
All runtime config lives in `.env` (never committed):

```
filepathurl        remote CSV source (AraFinNews GitHub raw)
csv_path           local preprocessed CSV
model_id           HuggingFace embedding model ID
chunk_size         tokens per chunk (500)
chunk_overlap      overlap tokens (50)
qdrant_path        local directory for Qdrant persistence
qdrant_collection  Qdrant collection name
LLM_model          Ollama model tag (qwen2.5:7b)
```

---

## Running the API server (Uvicorn)

The whole RAG pipeline (embedder, Qdrant, reranker, Ollama client) loads **once** at server
startup — the first ~2–5 s. After that it stays in memory and is reused for every request, so
asking a question hours apart does **not** reload the pipeline.

### Start the server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```

- Wait for the log line **`RAG pipeline ready.`** before sending requests. Until then,
  `GET /health` returns `503`.
- **`--workers 1` is required.** Each worker is a separate process that would load the full
  pipeline (~500–700 MB) and contend on the local Qdrant file lock. Concurrent requests are
  already serialized in-process behind a lock; scale with a request queue or a dedicated
  inference service, not more workers.
- Add `--reload` during development to auto-restart on code changes (do **not** use in
  production — it re-triggers the slow startup load on every edit).

### Stop the server

- **Foreground (running in your terminal):** press **`Ctrl + C`** in that terminal. Uvicorn
  runs the `lifespan` shutdown, closes the Qdrant client cleanly, and exits.
- **If it's stuck or running in the background**, find and kill the process by port:

  **Windows (PowerShell):**
  ```powershell
  # find the PID listening on 8000
  Get-NetTCPConnection -LocalPort 8000 | Select-Object OwningProcess
  # stop it
  Stop-Process -Id <PID> -Force
  ```

  **macOS / Linux:**
  ```bash
  lsof -i :8000        # find the PID
  kill <PID>           # or: kill -9 <PID> if it won't stop
  ```

> Note: the Qdrant store uses a file lock. If the server didn't shut down cleanly, you may
> need to ensure no leftover process holds `qdrant_db/` before restarting.

---

## Using the API

### Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/health` | — | `{"status":"ok","ready":true}` (503 until loaded) |
| `POST` | `/ask` | `{"query": "..."}` | `{"answer": "...", "sources": [...]}` |
| `GET` | `/docs` | — | Interactive Swagger UI |

> ⚠️ `/ask` is **POST only** — sending a `GET` (e.g. typing the URL in a browser) returns
> **405 Method Not Allowed**.

### Example — curl
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "ما أخبار أسعار النفط في 2020؟"}'
```

### Example — Postman
1. Method **POST**, URL `http://localhost:8000/ask`
2. **Body → raw → JSON**
3. ```json
   { "query": "ما أخبار أسعار النفط في 2020؟" }
   ```
4. **Send** (first answer may take 2–7 s).

Or just open **`http://localhost:8000/docs`** and use the **"Try it out"** button — no client setup needed.

### Example response
```json
{
  "answer": "... النص العربي مع اقتباسات [1] [2] ...\n\n## المراجع\n1. https://... (2020-01-07)",
  "sources": [
    { "title": "...", "date": "2020-01-07", "url": "https://..." }
  ]
}
```

---

## Evaluation

The pipeline is evaluated in two families — deterministic retrieval metrics and Ragas-judged
generation quality. Generate once, then score:

```bash
python eval/generate_outputs.py    # run RAG once → shared dump (slow)
python eval/retrieval_metrics.py   # Hit@k, Recall@k, Precision@k, MRR (instant)
python eval/run_ragas.py           # faithfulness / context / correctness (slow, LLM judge)
```

See the **Evaluation** section of [`CLAUDE.md`](./CLAUDE.md) for the golden-dataset format,
judge caveats, and known findings.

---

## Notes

- **Ollama model eviction:** the FastAPI pipeline stays loaded, but Ollama unloads model
  weights after ~5 min idle (default `keep_alive`), so the first query after a long gap pays a
  brief reload-from-disk cost. Pin `keep_alive` on the `ChatOllama` instances (or set
  `OLLAMA_KEEP_ALIVE`) to avoid it.
- **`.env` is never committed** — it holds all runtime config.
