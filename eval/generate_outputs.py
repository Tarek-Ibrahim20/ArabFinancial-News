# eval/generate_outputs.py — run the RAG pipeline ONCE, dump outputs for both eval scripts
#
# This is the single source of generated RAG outputs. It calls answer_for_eval(question)
# once per golden question and persists everything both metric families need:
#   - retrieval_metrics.py  uses  retrieved_ids   (raw retrieve ranking)
#   - run_ragas.py          uses  answer + contexts
# so the (slow) pipeline never runs twice and the two reports can't drift apart.
#
# Run this FIRST, then run eval/retrieval_metrics.py and eval/run_ragas.py — both just
# read the dump it writes (pipeline_outputs_latest.jsonl).
#
# Usage:
#   python eval/generate_outputs.py        # generate over the golden set
#
# Behaviour is controlled by the default constants below (no CLI args).

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))  # make root modules (rag_chain, retrieval) importable

# ── Defaults ────────────────────────────────────────────────────────────────────
_GOLDEN_PATH   = _ROOT / "eval" / "golden_dataset.jsonl"
_OUT_DIR       = _ROOT / "eval" / "results"
_LATEST_NAME   = "pipeline_outputs_latest.jsonl"  # stable copy both consumers read
_LIMIT         = None          # cap number of questions, or None for all
_VERIFIED_ONLY = False         # only generate for human-verified golden records


def _load_golden(path: Path, verified_only: bool, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    n_verified = sum(1 for r in rows if r.get("verified"))
    print(f"[generate_outputs] golden: {len(rows)} records, {n_verified} verified", file=sys.stderr)
    if verified_only:
        rows = [r for r in rows if r.get("verified")]
        print(f"[generate_outputs] verified-only -> {len(rows)} records", file=sys.stderr)
    elif n_verified < len(rows):
        print(f"[generate_outputs] WARNING: including {len(rows) - n_verified} unverified DRAFT records.",
              file=sys.stderr)
    if limit:
        rows = rows[:limit]
    return rows


def main() -> None:
    rows = _load_golden(_GOLDEN_PATH, _VERIFIED_ONLY, _LIMIT)
    if not rows:
        print("[generate_outputs] no records to generate.", file=sys.stderr)
        sys.exit(1)

    from rag_chain import answer_for_eval  # heavy import (loads retrieval + LLM stack)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    dump_path = _OUT_DIR / f"pipeline_outputs_{stamp}.jsonl"

    with dump_path.open("w", encoding="utf-8") as f:
        for n, r in enumerate(rows, 1):
            res = answer_for_eval(r["question"])
            rec = {
                "question":              r["question"],
                "question_type":         r["question_type"],
                "ground_truth":          r["ground_truth"],
                "reference_article_ids": r["reference_article_ids"],
                "answer":                res["answer"],
                "contexts":              res["contexts"],
                "retrieved_ids":         res["retrieved_ids"],
                "sources":               res["sources"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[generate_outputs] {n}/{len(rows)} ({r['question_type']}) "
                  f"retrieved_ids={res['retrieved_ids']}", file=sys.stderr)

    # Stable copy that both consumers default to reading.
    latest_path = _OUT_DIR / _LATEST_NAME
    shutil.copyfile(dump_path, latest_path)

    print(f"\n[generate_outputs] wrote {len(rows)} records", file=sys.stderr)
    print(f"[generate_outputs] dump   -> {dump_path}", file=sys.stderr)
    print(f"[generate_outputs] latest -> {latest_path}", file=sys.stderr)
    print("[generate_outputs] NEXT: run eval/retrieval_metrics.py and eval/run_ragas.py", file=sys.stderr)


if __name__ == "__main__":
    main()
