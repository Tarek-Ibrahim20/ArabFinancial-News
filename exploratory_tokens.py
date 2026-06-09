# Token Exploratory Analysis — AraFinNews `text` column
#
# Purpose:
#   Count tokens in the `text` column (title + article) using the tiktoken
#   cl100k_base encoder (used by OpenAI text-embedding-3-small/large and ada-002).
#   The distribution guides the choice of embedding model max-token limit
#   and the chunking strategy for the RAG pipeline.
#
# Outputs:
#   - Console: descriptive stats + percentile table + model fit summary
#   - token_distribution.png: histogram of token lengths

import pandas as pd
import tiktoken
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display required)

CSV_PATH   = "sample_dataset.csv"
ENCODING   = "cl100k_base"   # tokenizer for text-embedding-3-* and ada-002

# Common embedding model context windows to benchmark against
MODEL_LIMITS = {
    "text-embedding-3-small / large  (OpenAI)": 8191,
    "text-embedding-ada-002          (OpenAI)": 8191,
    "embed-multilingual-v3.0         (Cohere)": 512,
    "AraBERT / CAMeLBERT             (HF)":     512,
    "multilingual-e5-large           (HF)":     512,
}

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", usecols=["text"])
df.dropna(subset=["text"], inplace=True)
df["text"] = df["text"].astype(str)

print(f"Loaded {len(df):,} rows from '{CSV_PATH}'\n")

# ── Tokenise ──────────────────────────────────────────────────────────────────
enc = tiktoken.get_encoding(ENCODING)

print(f"Counting tokens with encoder: {ENCODING} ...")
df["token_count"] = df["text"].apply(lambda t: len(enc.encode(t)))
print("Done.\n")

# ── Descriptive statistics ────────────────────────────────────────────────────
stats = df["token_count"].describe(percentiles=[.25, .5, .75, .90, .95, .99])
print("=" * 50)
print("Token count — descriptive statistics")
print("=" * 50)
print(stats.to_string())
print()

# Explicit percentile table
percentiles = [50, 75, 90, 95, 99, 100]
print(f"{'Percentile':>12}  {'Token count':>12}")
print("-" * 28)
for p in percentiles:
    val = df["token_count"].quantile(p / 100)
    print(f"{p:>11}%  {int(val):>12,}")
print()

# ── Model fit summary ─────────────────────────────────────────────────────────
print("=" * 60)
print("% of articles that fit within each model's context window")
print("=" * 60)
total = len(df)
for model, limit in MODEL_LIMITS.items():
    fits     = (df["token_count"] <= limit).sum()
    too_long = total - fits
    print(f"  {model}")
    print(f"    limit={limit:,} tokens  |  fits={fits:,} ({fits/total*100:.1f}%)  |  exceeds={too_long:,} ({too_long/total*100:.1f}%)")
    print()

# ── Chunking recommendation ───────────────────────────────────────────────────
median_tok  = int(df["token_count"].median())
p95_tok     = int(df["token_count"].quantile(0.95))
p99_tok     = int(df["token_count"].quantile(0.99))

print("=" * 60)
print("Chunking recommendation")
print("=" * 60)
print(f"  Median text length : {median_tok:,} tokens")
print(f"  95th percentile    : {p95_tok:,} tokens")
print(f"  99th percentile    : {p99_tok:,} tokens")
print()
print("  Suggested chunk_size values to evaluate:")
for cs in [256, 512, 1024, 2048]:
    pct_fit = (df["token_count"] <= cs).mean() * 100
    print(f"    chunk_size={cs:>5}  →  {pct_fit:.1f}% of texts fit in a single chunk (no splitting)")
print()

# ── Histogram ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(df["token_count"], bins=80, color="#4C72B0", edgecolor="white", linewidth=0.4)

# Mark common chunk sizes
for label, limit in [("512", 512), ("1024", 1024), ("2048", 2048), ("8191", 8191)]:
    ax.axvline(limit, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(limit + 30, ax.get_ylim()[1] * 0.95, label,
            color="red", fontsize=8, va="top")

ax.set_title(f"Token length distribution — {len(df):,} articles  ({ENCODING})", fontsize=13)
ax.set_xlabel("Token count per article (title + body)")
ax.set_ylabel("Number of articles")
plt.tight_layout()
plt.savefig("token_distribution.png", dpi=150)
print("Histogram saved → token_distribution.png")
