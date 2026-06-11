# rag_chain.py — AraFinNews production augmentation module
#
# Exposes a single public function:
#   answer(user_query) -> {"answer": str, "sources": list[dict]}
#
# Pipeline:
#   retrieve(query) → deduplicate → U-shape reorder → format context → Ollama LLM → answer + sources
#
# Import and call from any downstream module (API, evaluation, etc.):
#   from rag_chain import answer
#   result = answer("ما أخبار أسعار النفط في 2020؟")

import re

from dotenv import dotenv_values
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI
from retrieval import retrieve

# ── Config ────────────────────────────────────────────────────────────────────
_config    = dotenv_values(".env")
_LLM_MODEL = _config["LLM_model"]
# _GENERATION_MODEL = _config["generation_model"]

# ── LLM (generation only — retrieval stack lives in retrieval.py) ─────────────
_llm = ChatOllama(model=_LLM_MODEL, temperature=0, num_ctx=5000 )
# _gen_llm = ChatOpenAI(model=_GENERATION_MODEL, temperature=0)

# ── RAG prompt ────────────────────────────────────────────────────────────────
_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an Arabic financial news assistant. Rules:\n"
        "1. Summarize what the provided sources say about the topic. No outside knowledge.\n"
        "2. Cite every fact inline with a bracketed number matching the source: [1], [2], etc.\n"
        "3. Write in Arabic (فصحى), 5-7 sentences. No intro, no conclusion.\n"
        "4. ONLY if the sources contain zero relevant content, say exactly: "
        "لا تتوفر معلومات كافية في المصادر المقدمة.\n"
        "5. Do NOT write a references list — it is appended automatically. "
        "Just write the summary with inline [N] markers."
    )),
    # ── One-shot example ─────────────────────────────────────────────────────
    ("human", (
        " [1]: (2020-01-07) | https://www.argaam.com/ar/article/articledetail/id/1340083\n"
        "تراجعت أسعار النفط الخام اليوم مع بقاء المستثمرين حذرين إزاء التطورات في الشرق الأوسط، وتراجع خام برنت 0.4% ليصل إلى 68.3 دولار للبرميل.\n"
        "---\n"
        " [2]: (2020-01-08) | https://www.argaam.com/ar/article/articledetail/id/1340311\n"
        "تحولت أسعار النفط إلى الانخفاض وتداول خام برنت دون مستوى 69 دولاراً للبرميل في أعقاب الهجمات على القواعد الأمريكية في العراق.\n"
        "---\n\n"
        "السؤال: ما أخبار أسعار النفط في مطلع يناير 2020؟"
    )),
    ("ai", (
        "تراجعت أسعار النفط الخام في مطلع يناير 2020 مع حذر المستثمرين من التوترات الجيوسياسية، "
        "إذ انخفض خام برنت 0.4% ليبلغ 68.3 دولار للبرميل [1]. "
        "وفي اليوم التالي تحولت الأسعار إلى مزيد من الانخفاض وتداول برنت دون مستوى 69 دولاراً "
        "في أعقاب الهجمات على القواعد الأمريكية بالعراق [2]."
    )),
    # ── Actual query ─────────────────────────────────────────────────────────
    ("human", "المصادر:\n{context}\n\nالسؤال: {question}"),
])


# ── Helpers (private) ─────────────────────────────────────────────────────────

def _deduplicate(docs: list[Document]) -> list[Document]:
    # Drop chunks with identical content. Uses page_content hash, not article id,
    # so two genuinely different chunks from the same article are both kept.
    # Exact duplicates arise when RRF fusion surfaces the same chunk from both
    # the dense and sparse retrieval paths.
    seen: set = set()
    result: list[Document] = []
    for doc in docs:
        h = hash(doc.page_content)
        if h not in seen:
            seen.add(h)
            result.append(doc)
    return result


def _u_shape(docs: list[Document]) -> list[Document]:
    # Reorder to counter Lost-in-the-Middle: most relevant at start and end,
    # least relevant in the middle.
    # For 5 docs [d1..d5] sorted best→worst: result = [d1, d3, d5, d4, d2]
    if len(docs) <= 2:
        return docs
    evens = docs[::2]
    odds  = docs[1::2]
    return evens + odds[::-1]


def _format_context(docs: list[Document]) -> str:
    # Wrap each doc in a numbered source block so the LLM can cite sources.
    parts = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        parts.append(
            f" [{i}]: ({m['date']}) | {m['url']}\n"
            f"{doc.page_content}\n"
            f"---"
        )
    return "\n".join(parts)


def _format_references(body: str, docs: list[Document]) -> str:
    # Build the ## المراجع section deterministically from docs.
    # Lists only the sources the model actually cited (parsed from [N] markers),
    # so URLs are exact and never hallucinated.
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", body)})
    lines = ["## المراجع", ""]
    for n in cited:
        if 1 <= n <= len(docs):
            m = docs[n - 1].metadata
            lines.append(f"{n}. {m['url']} ({m['date']})")
    return "\n ".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def answer(user_query: str) -> dict:
    """Full RAG pipeline: retrieve → deduplicate → U-shape → generate.

    Returns:
        {
            "answer":  str,              # Arabic answer grounded in retrieved context
            "sources": list[dict],       # [{"title": ..., "date": ..., "url": ...}, ...]
        }
    """
    docs = retrieve(user_query)
    if not docs:
        return {"answer": "لا تتوفر معلومات كافية في قاعدة البيانات.", "sources": []}

    docs    = _deduplicate(docs)
    docs    = _u_shape(docs)
    context = _format_context(docs)

    print("=== RAG CONTEXT START ===")
    print(context)
    print("=== RAG CONTEXT END ===")

    response = _llm.invoke(
        _RAG_PROMPT.format_messages(context=context, question=user_query)
    )

    body = response.content.strip()
    cited = re.findall(r"\[(\d+)\]", body)
    if cited:
        body = f"{body}\n\n{_format_references(body, docs)}"

    sources = [
        {"title": d.metadata["title"], "date": d.metadata["date"], "url": d.metadata["url"]}
        for d in docs
    ]
    return {"answer": body.strip()}

