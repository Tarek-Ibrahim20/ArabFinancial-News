"""FastAPI serving layer for the ArabFinancial News RAG pipeline.

Thin HTTP wrapper around `rag_chain.answer`. The whole pipeline (embedder,
Qdrant, reranker, Ollama) is loaded **once** at startup — never per request —
because import alone costs ~2-5 s. Requests are serialized behind a single lock:
the local Qdrant store uses a file lock and Ollama is a single shared
connection, so concurrent pipeline calls are unsafe.

Run (single worker is required — see module docstring / CLAUDE.md):

    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
"""

import asyncio
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("rag_api")

# ── Shared state ───────────────────────────────────────────────────────────────
# `_answer` holds rag_chain.answer once startup loads it. `_lock` serializes
# pipeline calls so only one runs at a time (Qdrant file lock + single Ollama conn).
_answer        = None
_answer_stream = None
_lock          = threading.Lock()
_TIMEOUT       = 600.0  # seconds before a hung pipeline returns 504
_QUERY_LOG     = Path("logs/rag_requests.jsonl")


def _log_query(endpoint: str, query: str, result: dict | None,
               latency_s: float, status: str) -> None:
    _QUERY_LOG.parent.mkdir(exist_ok=True)
    record = {
        "ts":          datetime.now(timezone.utc).isoformat(),
        "endpoint":    endpoint,
        "query":       query,
        "answer":      result.get("answer", "") if result else "",
        "sources":     result.get("sources", []) if result else [],
        "num_sources": len(result.get("sources", [])) if result else 0,
        "latency_s":   round(latency_s, 2),
        "status":      status,
    }
    with open(_QUERY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG pipeline once at startup; fail fast if it can't load."""
    global _answer, _answer_stream
    _log.info("Loading RAG pipeline (embedder, Qdrant, reranker, Ollama)...")
    try:
        from rag_chain import answer, answer_stream  # heavy import: ~2-5 s

        _answer        = answer
        _answer_stream = answer_stream
    except Exception:
        _log.exception("Failed to load RAG pipeline at startup")
        raise
    _log.info("RAG pipeline ready. Warming up Ollama model weights...")
    _t = time.perf_counter()
    try:
        _answer("مرحبا")
        _log.info("Warm-up done in %.1fs — pipeline hot.", time.perf_counter() - _t)
    except Exception:
        _log.warning("Warm-up query failed — first real request may be slow.", exc_info=True)
    yield


app = FastAPI(
    title="ArabFinancial News RAG API",
    description="Arabic financial-news question answering over the AraFinNews corpus.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ──────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Arabic financial-news question.")


class Source(BaseModel):
    title: str
    date: str
    url: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class HealthResponse(BaseModel):
    status: str
    ready: bool


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    """Readiness probe. 503 until the pipeline has finished loading."""
    ready = _answer is not None
    if not ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    return HealthResponse(status="ok", ready=True)


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """Answer an Arabic financial-news question, grounded in retrieved sources."""
    if _answer is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    def _run():
        # Serialize: only one pipeline call at a time. Held in the worker thread
        # so the event loop (and /health) stays responsive during the 2-7 s call.
        with _lock:
            return _answer(req.query)

    t0 = time.perf_counter()
    try:
        result = await asyncio.wait_for(run_in_threadpool(_run), timeout=_TIMEOUT)
        _log_query("/ask", req.query, result, time.perf_counter() - t0, "ok")
        return result
    except asyncio.TimeoutError:
        _log_query("/ask", req.query, None, time.perf_counter() - t0, "timeout")
        _log.error("Pipeline timed out after %.0fs for query: %r", _TIMEOUT, req.query)
        raise HTTPException(status_code=504, detail="Request timed out")
    except Exception:
        _log_query("/ask", req.query, None, time.perf_counter() - t0, "error")
        _log.exception("Pipeline failed for query: %r", req.query)
        raise HTTPException(status_code=500, detail="Internal error answering query")


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """Stream the Arabic answer as plain text — readable in Postman and Swagger.

    Tokens arrive as raw Arabic characters. After generation ends, a `---` separator
    is appended followed by a numbered source list.
    """
    if _answer_stream is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    def _gen():
        t0      = time.perf_counter()
        sources = []
        tokens  = []
        try:
            with _lock:
                for event in _answer_stream(req.query):
                    if event["type"] == "token":
                        tokens.append(event["text"])
                        yield event["text"]
                    elif event["type"] == "sources":
                        sources = event["sources"]
            if sources:
                yield "\n\n---\n"
                for i, s in enumerate(sources, 1):
                    yield f"{i}. {s['title']} ({s['date']})\n   {s['url']}\n"
            _log_query("/ask/stream",
                       req.query,
                       {"answer": "".join(tokens), "sources": sources},
                       time.perf_counter() - t0, "ok")
        except Exception:
            _log_query("/ask/stream", req.query, None,
                       time.perf_counter() - t0, "error")
            raise

    async def _timed_stream():
        ait = iterate_in_threadpool(_gen())
        while True:
            try:
                chunk = await asyncio.wait_for(ait.__anext__(), timeout=_TIMEOUT)
                yield chunk
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                _log.error("Streaming pipeline timed out for query: %r", req.query)
                yield "\n\n[انتهت مهلة الطلب]"  # "Request timed out"
                return

    return StreamingResponse(
        _timed_stream(),
        media_type="text/plain; charset=utf-8",
    )
