# eval/retrieval_metrics.py — deterministic retrieval evaluation (no LLM judge)
#
# Scores the retriever by comparing the article ids it returned to each question's
# reference_article_ids. Pure set/rank math — fast, reproducible, no LLM.
#
# Reads the shared pipeline dump written by eval/generate_outputs.py (so the RAG runs
# once for both eval scripts). Run generate_outputs.py FIRST. This script does not
# import the retrieval stack — it only reads JSON.
#
# Metrics (computed at k = 1, 3, 5):
#   Hit@k        did any relevant article appear in the top k?            (0/1, then averaged)
#   Recall@k     fraction of a question's relevant articles found in top k
#   Precision@k  fraction of the top k that are relevant
#   MRR (Mean Reciprocal Rank)         1 / rank of the first relevant article (0 if none in top k_max)
#
# Why these: Hit@k and MRR answer "can the generator even see the right source,
# and how high is it?"; Recall@k matters for multi_hop/comparison where the answer
# needs more than one article. Results are broken down by question_type because
# multi-source questions are expected to be the hard case.
#
# Usage:
#   python eval/retrieval_metrics.py

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# ── Defaults ────────────────────────────────────────────────────────────────────
_OUT_DIR   = _ROOT / "eval" / "results"
_DUMP_PATH = _OUT_DIR / "pipeline_outputs_20260616_212850.jsonl"  # written by generate_outputs.py
_KS        = [1, 3, 5]


def _load_dump(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[retrieval_metrics] dump not found: {path}", file=sys.stderr)
        print("[retrieval_metrics] run `python eval/generate_outputs.py` first.", file=sys.stderr)
        sys.exit(1)
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    print(f"[retrieval_metrics] loaded {len(rows)} records from {path.name}", file=sys.stderr)
    return rows


def _score(retrieved_ids: list[int], relevant_ids: list[int], ks: list[int]) -> dict:
    # retrieved_ids: ranked best->worst. relevant_ids: ground-truth set for this question.
    relevant = set(relevant_ids)
    out: dict = {}
    for k in ks:
        topk = retrieved_ids[:k]
        n_hit = sum(1 for i in topk if i in relevant)
        out[f"hit@{k}"]       = 1.0 if n_hit > 0 else 0.0
        out[f"recall@{k}"]    = n_hit / len(relevant) if relevant else 0.0
        out[f"precision@{k}"] = n_hit / k if k else 0.0
    # MRR over the full returned list (reciprocal rank of first relevant hit).
    rr = 0.0
    for rank, i in enumerate(retrieved_ids, 1):
        if i in relevant:
            rr = 1.0 / rank
            break
    out["mrr"] = rr
    return out


def _mean(dicts: list[dict]) -> dict:
    if not dicts:
        return {}
    keys = dicts[0].keys()
    return {k: sum(d[k] for d in dicts) / len(dicts) for k in keys}


def _print_table(title: str, m: dict, ks: list[int]) -> None:
    print(f"\n{title}")
    print("  " + "  ".join(f"hit@{k}" for k in ks) + "   " +
          "  ".join(f"rec@{k}" for k in ks) + "   MRR")
    row = "  ".join(f" {m[f'hit@{k}']:.2f} " for k in ks) + "   " + \
          "  ".join(f"{m[f'recall@{k}']:.2f}" for k in ks) + f"   {m['mrr']:.2f}"
    print("  " + row)


def main() -> None:
    ks = sorted(_KS)
    rows = _load_dump(_DUMP_PATH)
    if not rows:
        print("[retrieval_metrics] dump is empty.", file=sys.stderr)
        sys.exit(1)

    per_question: list[dict] = []
    by_type: dict[str, list[dict]] = defaultdict(list)

    for n, r in enumerate(rows, 1):
        retrieved_ids = [int(i) for i in r["retrieved_ids"]]
        scores = _score(retrieved_ids, r["reference_article_ids"], ks)
        by_type[r["question_type"]].append(scores)
        per_question.append({
            "question_type":         r["question_type"],
            "reference_article_ids": r["reference_article_ids"],
            "retrieved_ids":         retrieved_ids,
            "scores":                scores,
        })
        print(f"[retrieval_metrics] {n}/{len(rows)} {r['question_type']:<11} "
              f"hit@{ks[-1]}={scores[f'hit@{ks[-1]}']:.0f} mrr={scores['mrr']:.2f} "
              f"ref={r['reference_article_ids']} got={retrieved_ids}", file=sys.stderr)

    overall = _mean([p["scores"] for p in per_question])
    type_means = {t: _mean(s) for t, s in by_type.items()}

    _print_table(f"OVERALL  (n={len(per_question)})", overall, ks)
    for t in sorted(type_means):
        _print_table(f"{t}  (n={len(by_type[t])})", type_means[t], ks)

    out_dir = _OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp":     stamp,
        "dump":          str(_DUMP_PATH),
        "ks":            ks,
        "n_questions":   len(per_question),
        "overall":       overall,
        "by_type":       type_means,
        "per_question":  per_question,
    }
    out_path = out_dir / f"retrieval_metrics_{stamp}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[retrieval_metrics] report -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
