# Sample Dataset Builder — AraFinNews
# Fetches the first 10,000 rows of the AraFinNews Arabic financial news CSV from the remote URL
# defined in .env (filepathurl).
#
# Structural cleaning (pd.read_csv):
#   - Arabic UTF-8 encoding enforced; stray bytes replaced instead of crashing
#   - Article column correctly unquoted (fields contain commas)
#   - id → nullable Int64, text columns → StringDtype, date → datetime64
#   - Malformed rows skipped with a warning (on_bad_lines="warn")
#   - Column names whitespace-stripped; fully empty rows dropped
#
# Content cleaning (preprocess_text UDF — applied to title & article):
#   - HTML entities decoded  (&quot; → "  &amp; → &  etc.)
#   - Residual HTML tags stripped
#   - Hidden/bidi-override Unicode characters removed (U+202D, U+200B, etc.)
#   - ASCII control characters removed
#   - Whitespace normalized (multiple spaces/tabs → single space)
#
# Output:
#   - title and article stored cleaned
#   - text = title + "\n\n" + article  →  feed this column to the embedder

import html
import re

import pandas as pd
from dotenv import dotenv_values

_HIDDEN_CHARS = (
    '​',  # ZERO WIDTH SPACE
    '‌',  # ZERO WIDTH NON-JOINER
    '‍',  # ZERO WIDTH JOINER
    '‎',  # LEFT-TO-RIGHT MARK
    '‏',  # RIGHT-TO-LEFT MARK
    '‭',  # LEFT-TO-RIGHT OVERRIDE  (U+202D)
    '‮',  # RIGHT-TO-LEFT OVERRIDE
    '﻿',  # ZERO WIDTH NO-BREAK SPACE / BOM
    '\xa0',    # NON-BREAKING SPACE
)

def preprocess_text(s) -> str:
    if not isinstance(s, str) or not s.strip():
        return ""
    s = html.unescape(s)                              # &quot; &amp; &#39; etc.
    s = re.sub(r'<[^>]+>', '', s)                     # residual HTML tags
    for ch in _HIDDEN_CHARS:
        s = s.replace(ch, '')                         # hidden / bidi-override chars
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', s)  # ASCII control chars
    s = re.sub(r'[ \t]+', ' ', s)                    # collapse spaces/tabs
    s = re.sub(r'\n{3,}', '\n\n', s)                 # collapse excess newlines
    return s.strip()


config = dotenv_values(".env")

raw = config.get("filepathurl", "")
url = re.sub(r'^"|"$', "", raw.strip())  # strip surrounding quotes if present

if not url:
    raise ValueError("filepathurl not found in .env — check format is key=value")

df = pd.read_csv(
    url,
    nrows=10000,
    encoding="utf-8",
    encoding_errors="replace",
    quotechar='"',
    quoting=0,
    dtype={
        "id":      "Int64",
        "title":   "string",
        "article": "string",
        "url":     "string",
    },
    parse_dates=["date"],
    date_format="%m/%d/%Y",
    on_bad_lines="warn",
)

df.columns = df.columns.str.strip()
df.dropna(how="all", inplace=True)

# Content cleaning
df['title']   = df['title'].apply(preprocess_text)
df['article'] = df['article'].apply(preprocess_text)

# Join for embedding
df['text'] = df['title'] + ' | ' + df['article']

df.to_csv("sample_dataset.csv", index=False, encoding="utf-8-sig")

print(f"Saved {len(df):,} rows  |  columns: {list(df.columns)}")
print(f"Date range : {df['date'].min().date()} -> {df['date'].max().date()}")
print(f"Null counts:\n{df.isnull().sum()}")
print(df.head(3).to_string())
