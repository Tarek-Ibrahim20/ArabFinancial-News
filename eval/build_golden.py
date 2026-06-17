# eval/build_golden.py — build a golden evaluation dataset for the RAG pipeline
#
# Samples articles from sample_dataset.csv and asks the local LLM (qwen2.5:7b via
# Ollama) to draft an Arabic question + reference answer grounded ONLY in those
# article(s). Output is a JSONL file ready for Ragas + retrieval metrics.
#
# IMPORTANT: this produces DRAFTS. Every record must be human-verified before use —
# a moderate 7B model will occasionally hallucinate, mislabel, or write a question
# the source can't actually answer. The schema reserves a "verified" flag for that.
#
# Usage:
#   python eval/build_golden.py                       # 30 questions, default mix
#   python eval/build_golden.py --factual 12 --explanation 8 --comparison 5 --multi_hop 5
#   python eval/build_golden.py --out eval/golden_dataset.jsonl --seed 42
#
# Record schema (one JSON object per line):
#   {
#     "question":              str,        # Arabic question
#     "question_type":         str,        # factual | comparison | explanation | multi_hop
#     "ground_truth":          str,        # Arabic reference answer
#     "reference_article_ids": [int],      # source article id(s) — for retrieval metrics
#     "reference_contexts":    [str],      # source article text(s) — for context_recall
#     "verified":              false       # flip to true after human review
#   }

import argparse
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values
from langchain_ollama import ChatOllama

# ── Config ────────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).resolve().parent.parent
_config    = dotenv_values(_ROOT / ".env")
_CSV_PATH  = _ROOT / _config["csv_path"]
_LLM_MODEL = _config["LLM_model"]

# Single-article types need 1 source; multi-article types need 2.
_MULTI_TYPES  = {"comparison", "multi_hop"}
_SINGLE_TYPES = {"factual", "explanation"}

# format="json" makes Ollama emit a parseable JSON object instead of free text.
_llm = ChatOllama(model=_LLM_MODEL, temperature=0.2, num_ctx=8192, format="json")

# ── Per-type drafting instructions (Arabic question, Arabic answer) ────────────
_TYPE_INSTRUCTIONS = {
    "factual": (
        "اكتب سؤالاً مباشراً عن حقيقة أو رقم أو تاريخ ورد صراحةً في المقال، "
        "بحيث تكون إجابته جملة قصيرة دقيقة مأخوذة من النص."
    ),
    "explanation": (
        "اكتب سؤالاً بصيغة «لماذا» أو «كيف» يتطلب شرحاً من المقال، "
        "بحيث تكون الإجابة فقرة قصيرة تشرح السبب أو الآلية كما وردت في النص."
    ),
    "comparison": (
        "اكتب سؤالاً واحداً يقارن بين معطيين أو رقمين أو موقفين وردا في المصدرين، "
        "بحيث لا يمكن الإجابة عنه إلا بالرجوع إلى كلا المصدرين معاً."
    ),
    "multi_hop": (
        "اكتب سؤالاً واحداً يتطلب الربط بين معلومة من المصدر الأول ومعلومة من المصدر الثاني "
        "للوصول إلى الإجابة، بحيث لا يكفي مصدر واحد للإجابة عنه."
    ),
}

_SYSTEM = (
    "أنت خبير في إعداد بيانات تقييم لنظام أسئلة وأجوبة مالية باللغة العربية.\n"
    "مهمتك: توليد سؤال وإجابة مرجعية اعتماداً على المصدر/المصادر المقدمة فقط.\n"
    "قواعد صارمة:\n"
    "- لا تستخدم أي معلومة من خارج النص المقدم.\n"
    "- السؤال والإجابة بالعربية الفصحى.\n"
    "- يجب أن تكون الإجابة مذكورة فعلاً في النص، لا استنتاجاً خارجياً.\n"
    "- يجب أن يبدو السؤال طبيعياً كأن مستخدماً حقيقياً يطرحه؛ "
    "لا تُشِر داخل نص السؤال إلى «المصدر الأول» أو «المصدر الثاني» أو إلى وجود مصادر أصلاً.\n"
    'أعد النتيجة بصيغة JSON فقط بهذا الشكل: {"question": "...", "ground_truth": "..."}'
)


def _year_month(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["date"], errors="coerce")
    df = df.copy()
    df["_year"]  = dt.dt.year
    df["_month"] = dt.dt.month
    return df.dropna(subset=["_year", "_month"])


def _stratified_sample(df: pd.DataFrame, n: int, rng: random.Random) -> list[int]:
    # Spread the sample across (year, month) buckets so date-filtered retrieval
    # gets exercised, then top up randomly if buckets run dry.
    buckets = [idx.tolist() for _, idx in df.groupby(["_year", "_month"]).groups.items()]
    rng.shuffle(buckets)
    for b in buckets:
        rng.shuffle(b)
    picked: list[int] = []
    while len(picked) < n and any(buckets):
        for b in buckets:
            if b:
                picked.append(b.pop())
                if len(picked) >= n:
                    break
    return picked[:n]


_STOPWORDS = {"في", "من", "على", "عن", "إلى", "مع", "أن", "إن", "the", "of", "and"}


def _title_tokens(title: str) -> set[str]:
    return {t for t in re.findall(r"[؀-ۿ]{4,}", str(title)) if t not in _STOPWORDS}


def _find_partner(df: pd.DataFrame, seed_pos: int, rng: random.Random) -> int | None:
    # Pair the seed with another article sharing a meaningful title token (likely
    # same topic/entity), so a 2-source question is actually answerable. Fall back
    # to a same-year article if no keyword overlap is found.
    seed = df.iloc[seed_pos]
    seed_tokens = _title_tokens(seed["title"])
    candidates = []
    for pos in range(len(df)):
        if pos == seed_pos:
            continue
        row = df.iloc[pos]
        if _title_tokens(row["title"]) & seed_tokens:
            candidates.append(pos)
    if candidates:
        return rng.choice(candidates)
    same_year = [p for p in range(len(df)) if p != seed_pos and df.iloc[p]["_year"] == seed["_year"]]
    return rng.choice(same_year) if same_year else None


def _build_human_message(qtype: str, rows: list[pd.Series]) -> str:
    parts = [f"المطلوب: {_TYPE_INSTRUCTIONS[qtype]}", ""]
    for i, row in enumerate(rows, 1):
        parts.append(f"المصدر {i} ({row['date']}): {row['title']}")
        parts.append(str(row["article"]))
        parts.append("---")
    return "\n".join(parts)


def _draft(qtype: str, rows: list[pd.Series]) -> dict | None:
    msg = [("system", _SYSTEM), ("human", _build_human_message(qtype, rows))]
    try:
        raw = _llm.invoke(msg).content
        data = json.loads(raw)
    except (json.JSONDecodeError, KeyError, Exception):  # noqa: BLE001 — drafting is best-effort
        return None
    q, gt = data.get("question", "").strip(), data.get("ground_truth", "").strip()
    if not q or not gt:
        return None
    return {
        "question":              q,
        "question_type":         qtype,
        "ground_truth":          gt,
        "reference_article_ids": [int(r["id"]) for r in rows],
        "reference_contexts":    [str(r["article"]) for r in rows],
        "verified":              False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a golden eval dataset (drafts) via qwen2.5:7b.")
    ap.add_argument("--factual",     type=int, default=12)
    ap.add_argument("--explanation", type=int, default=8)
    ap.add_argument("--comparison",  type=int, default=5)
    ap.add_argument("--multi_hop",   type=int, default=5)
    ap.add_argument("--out",  type=str, default=str(_ROOT / "eval" / "golden_dataset.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    df = _year_month(pd.read_csv(_CSV_PATH)).reset_index(drop=True)

    plan = {
        "factual":     args.factual,
        "explanation": args.explanation,
        "comparison":  args.comparison,
        "multi_hop":   args.multi_hop,
    }
    total = sum(plan.values())
    print(f"[build_golden] model={_LLM_MODEL}  target={total}  plan={plan}", file=sys.stderr)

    # Pre-pick seed positions, oversampling so failed drafts can be retried.
    seeds = _stratified_sample(df, min(total * 2, len(df)), rng)
    seed_iter = iter(seeds)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for qtype, count in plan.items():
            made = 0
            while made < count:
                try:
                    pos = next(seed_iter)
                except StopIteration:
                    print(f"[build_golden] ran out of seeds at {qtype} ({made}/{count})", file=sys.stderr)
                    break
                rows = [df.iloc[pos]]
                if qtype in _MULTI_TYPES:
                    partner = _find_partner(df, pos, rng)
                    if partner is None:
                        continue
                    rows.append(df.iloc[partner])
                rec = _draft(qtype, rows)
                if rec is None:
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                made += 1
                written += 1
                print(f"[build_golden] {qtype} {made}/{count}  ids={rec['reference_article_ids']}", file=sys.stderr)

    print(f"[build_golden] wrote {written} drafts -> {out_path}", file=sys.stderr)
    print("[build_golden] NEXT: human-verify each record and set \"verified\": true", file=sys.stderr)


if __name__ == "__main__":
    main()
