# eval/run_ragas.py — generation-quality evaluation via Ragas (local qwen2.5:7b judge)
#
# Pipeline:
#   pipeline_outputs_latest.jsonl       # written by eval/generate_outputs.py (run it FIRST)
#     → Ragas evaluate() with metrics:
#         faithfulness        answer grounded in retrieved contexts (no hallucination)
#         context_precision   retrieved contexts are relevant (uses reference)
#         context_recall      contexts cover what the reference answer needs
#         answer_correctness  answer matches the reference answer
#         (answer_relevancy   available but off by default: low-trust on Arabic)
#     → aggregate overall + by question_type → report (json + csv)
#
# This script does NOT generate — it scores the shared dump so the RAG runs once for
# both eval scripts. Run eval/generate_outputs.py first.
#
# Both the system-under-test generator AND the judge are qwen2.5:7b. That is cheap
# but NOT fully credible (a model grading itself). Treat absolute numbers as
# directional; the by-type deltas and run-over-run trends are what to trust. Swap
# the judge to a stronger/different model via _JUDGE_MODEL once one is available.
#
# Usage:
#   python eval/run_ragas.py        # score the shared dump
#
# Behaviour is controlled by the default constants below (no CLI args).

# ── ragas/langchain-1.x compat shim ───────────────────────────────────────────
# ragas 0.4.3 imports langchain_community.chat_models.vertexai.ChatVertexAI at
# module load, but that path was removed in the (sunset) langchain-community 0.4.x
# this project runs on. We never use VertexAI — stub the symbol so ragas imports.
# MUST run before any `import ragas`.
import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

import json
import warnings
from datetime import datetime
from pathlib import Path

from dotenv import dotenv_values

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))  # make root modules (rag_chain, retrieval) importable

warnings.filterwarnings("ignore")  # silence ragas' "use ragas.metrics.collections" notices

from langchain_ollama import ChatOllama  # noqa: E402
from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig  # noqa: E402

_config    = dotenv_values(_ROOT / ".env")
_LLM_MODEL = _config["LLM_model"]

_METRICS = {
    "faithfulness":       faithfulness,
    "answer_relevancy":   answer_relevancy,
    "context_precision":  context_precision,
    "context_recall":     context_recall,
    "answer_correctness": answer_correctness,
}

# ── Defaults ────────────────────────────────────────────────────────────────────
# answer_relevancy is intentionally dropped: low-trust on Arabic (~0.31, see CLAUDE.md)
# and slow. Add it back to _METRIC_NAMES if you want it.
_OUT_DIR      = _ROOT / "eval" / "results"
_DUMP_PATH    = _OUT_DIR / "pipeline_outputs_20260616_212850.jsonl"  # written by generate_outputs.py
_JUDGE_MODEL  = _LLM_MODEL
_METRIC_NAMES = ["faithfulness", "context_precision", "context_recall", "answer_correctness"]
_TIMEOUT      = 600            # per-call judge timeout (seconds)


def _load_dump(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[run_ragas] dump not found: {path}", file=sys.stderr)
        print("[run_ragas] run `python eval/generate_outputs.py` first.", file=sys.stderr)
        sys.exit(1)
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    print(f"[run_ragas] loaded {len(rows)} records from {path.name}", file=sys.stderr)
    return rows


def _strip_references(answer: str) -> str:
    # Score only the LLM-written body. The "## المراجع" block is appended by code
    # (deterministic urls/dates), not generated claims — judging it against the
    # retrieved context would unfairly penalise faithfulness/correctness.
    return answer.split("## المراجع", 1)[0].strip()


def _build_dataset(records: list[dict]) -> EvaluationDataset:
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=_strip_references(r["answer"]),
            retrieved_contexts=r["contexts"] or [""],  # ragas needs a non-empty list
            reference=r["ground_truth"],
        )
        for r in records
    ]
    return EvaluationDataset(samples=samples)


def main() -> None:
    out_dir = _OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 1. Load the shared pipeline dump (written by generate_outputs.py) ────────
    records = _load_dump(_DUMP_PATH)

    # ── 2. Judge LLM + embeddings (reuse the already-loaded Arabic embedder) ────
    # keep_alive="1h": pin the model in memory for the whole run. Without this, Ollama
    # evicts it after ~5 min idle and reloads from disk on the next (gappy, serial) ragas
    # call — that reload thrashing is what blew a 2-question run out to 8h of all-NaN timeouts.
    judge = LangchainLLMWrapper(
        ChatOllama(model=_JUDGE_MODEL, temperature=0, num_ctx=8192, keep_alive="1h")
    )
    # Reuse the project's Arabic embedder for embedding-based metric components
    # (e.g. answer_correctness). Importing retrieval.py loads the embedder once.
    from retrieval import _dense_embedder  # noqa: E402
    embeddings = LangchainEmbeddingsWrapper(_dense_embedder)

    metrics = [_METRICS[m] for m in _METRIC_NAMES]
    dataset = _build_dataset(records)

    print(f"[run_ragas] scoring {len(records)} samples x {len(metrics)} metrics "
          f"with judge={_JUDGE_MODEL} (this is slow on a local model)...", file=sys.stderr)

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge,
        embeddings=embeddings,
        # max_workers=1: a single local Ollama model can't truly parallelise — concurrent
        # calls just contend and trip per-call timeouts. Serial is as fast and more stable.
        run_config=RunConfig(timeout=_TIMEOUT, max_workers=1),
        raise_exceptions=False,  # a 7B judge will sometimes fail to parse -> NaN, keep going
        show_progress=True,
    )

    # ── 3. Aggregate: overall + by question_type ────────────────────────────────
    df = result.to_pandas()
    df["question_type"] = [r["question_type"] for r in records]
    metric_cols = [m for m in _METRIC_NAMES if m in df.columns]

    overall = {m: float(df[m].mean(skipna=True)) for m in metric_cols}
    by_type = {
        t: {m: float(g[m].mean(skipna=True)) for m in metric_cols}
        for t, g in df.groupby("question_type")
    }

    def _fmt(d: dict) -> str:
        return "  ".join(f"{m}={d[m]:.2f}" for m in metric_cols)

    print(f"\nOVERALL (n={len(df)})\n  {_fmt(overall)}")
    for t in sorted(by_type):
        n_t = int((df["question_type"] == t).sum())
        print(f"{t} (n={n_t})\n  {_fmt(by_type[t])}")

    # ── 4. Persist ──────────────────────────────────────────────────────────────
    csv_path  = out_dir / f"ragas_per_question_{stamp}.csv"
    json_path = out_dir / f"ragas_summary_{stamp}.json"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(
        json.dumps({
            "timestamp":    stamp,
            "judge_model":  _JUDGE_MODEL,
            "metrics":      metric_cols,
            "n_questions":  len(df),
            "overall":      overall,
            "by_type":      by_type,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[run_ragas] per-question -> {csv_path}", file=sys.stderr)
    print(f"[run_ragas] summary      -> {json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
