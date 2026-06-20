# retrieval.py — AraFinNews production retrieval module
#
# Exposes a single public function:
#   retrieve(user_query, k_candidates) -> list[Document]
#
# Pipeline:
#   1. LLM (Ollama) parses Arabic query → semantic text + optional year/month filters
#   2. Qdrant hybrid search (dense + BM25) on filtered collection (k_candidates results)
#   3. FlashRank Arabic reranker rescores candidates → top 5, sorted by relevance score
#
# Import and call from any downstream module (rag_chain, evaluation, etc.):
#   from retrieval import retrieve
#   docs = retrieve("أخبار البنوك العربية في يونيو 2019")

import atexit
import re
from typing import Optional

from dotenv import dotenv_values
from flashrank import Ranker
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

# ── Config ────────────────────────────────────────────────────────────────────
_config      = dotenv_values(".env")
_MODEL_ID    = _config["model_id"]
_QDRANT_PATH = _config["qdrant_path"]
_COLLECTION  = _config["qdrant_collection"]
_LLM_MODEL   = _config["LLM_model"]
_KEEP_ALIVE  = _config.get("keep_alive", "30m")

# ── Vectorstore (initialised once on import) ──────────────────────────────────
_dense_embedder = HuggingFaceEmbeddings(
    model_name=_MODEL_ID,
    model_kwargs={"device": "cpu", "local_files_only": True},
    encode_kwargs={"normalize_embeddings": True},
)
_sparse_embedder = FastEmbedSparse(model_name="Qdrant/bm25")

_client = QdrantClient(path=_QDRANT_PATH)
atexit.register(_client.close)

_vectorstore = QdrantVectorStore(
    client=_client,
    collection_name=_COLLECTION,
    embedding=_dense_embedder,
    sparse_embedding=_sparse_embedder,
    retrieval_mode=RetrievalMode.HYBRID,
)

# ── Reranker ──────────────────────────────────────────────────────────────────
_compressor = FlashrankRerank(client=Ranker(model_name="miniReranker_arabic_v1"), top_n=5 , score_threshold= 0.5)

# ── Chain 1 — Temporal extraction (year / month only) ────────────────────────
class TemporalFilter(BaseModel):
    year:  Optional[int] = Field(None)
    month: Optional[int] = Field(None)

_temporal_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "استخرج السنة والشهر من النص العربي فقط. أعد null إن لم يُذكرا.\n"
        "تاريخ النظام الحالي: يونيو 2026.\n\n"
        "'لعام 2020'          → year=2020, month=null\n"
        "'في يونيو 2019'      → year=2019, month=6\n"
        "'مايو 2025'          → year=2025, month=5\n"
        "'لسنة 2022'          → year=2022, month=null\n"
        "'الشهر الماضي'       → year=2026, month=5\n"
        "'هذا العام'          → year=2026, month=null\n"
        "'أخبار البنوك'       → year=null, month=null\n"
        "'استراتيجية النفط'   → year=null, month=null\n"
    )),
    ("human", "{user_input}"),
])

# ── Chain 2 — Semantic rewriting + BM25 keywords ─────────────────────────────
class SemanticQuery(BaseModel):
    search_query:      str       = Field(description="MSA rewrite, no dates or years.")
    keyword_variations: list[str] = Field(default_factory=list)

_semantic_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "أعد صياغة الاستعلام العربي بالفصحى الرسمية.\n"
        "احذف: التواريخ، السنوات، الشهور، العامية، التحيات.\n"
        "أضف 2-3 مصطلحات مالية عربية مرادفة. حروف عربية فقط.\n\n"
        "'اسعار النفط العالمي لعام 2020'              → search_query='أسعار النفط العالمية',           keywords=['خام برنت', 'أسواق الطاقة', 'تذبذب النفط']\n"
        "'بالله ابحثلي عن اخبار البنوك في يونيو 2019' → search_query='أخبار البنوك',                  keywords=['مصارف', 'قطاع مصرفي', 'سيولة مالية']\n"
        "'تقارير أرباح الشركات الربع سنوية مايو 2025' → search_query='تقارير أرباح الشركات الربعية', keywords=['عائدات', 'قوائم مالية', 'نمو الشركات']\n"
        "'استراتيجيه شركات النفط السعودية'             → search_query='استراتيجية شركات النفط السعودية', keywords=['أرامكو', 'قطاع الطاقة', 'تنويع النفط']\n"
    )),
    ("human", "{user_input}"),
])

_llm = ChatOllama(model=_LLM_MODEL, temperature=0, keep_alive=_KEEP_ALIVE)
_temporal_chain = _temporal_prompt | _llm.with_structured_output(TemporalFilter)
_semantic_chain = _semantic_prompt | _llm.with_structured_output(SemanticQuery)


# ── Public API ────────────────────────────────────────────────────────────────
def retrieve(user_query: str, k_candidates: int = 50) -> list[Document]:
    """Hybrid self-query retrieval with Arabic reranking.

    Returns up to 5 Document objects sorted by rerank score (highest first).
    Each Document carries full metadata: id, title, date, year, month, article, url.
    Falls back to plain hybrid search if LLM query parsing fails.
    """
    try:
        print(f"\n[Retrieval] Processing query: {user_query}")

        # Chain 1 — temporal filter (focused, high accuracy)
        temporal: TemporalFilter = _temporal_chain.invoke({"user_input": user_query})
        print(f"  [Temporal]  year={temporal.year}  month={temporal.month}")

        # Chain 2 — semantic rewrite + BM25 keywords
        semantic: SemanticQuery = _semantic_chain.invoke({"user_input": user_query})
        print(f"  [Semantic]  search_query='{semantic.search_query}'")

        # langchain_qdrant nests metadata under a "metadata" payload key
        conditions = []
        if temporal.year:
            conditions.append(FieldCondition(key="metadata.year",  match=MatchValue(value=temporal.year)))
        if temporal.month:
            conditions.append(FieldCondition(key="metadata.month", match=MatchValue(value=temporal.month)))
        qdrant_filter = Filter(must=conditions) if conditions else None

        # Reject keywords containing any non-Arabic characters
        arabic_keywords = [
            k for k in semantic.keyword_variations
            if k.strip() and all(c == ' ' or '؀' <= c <= 'ۿ' for c in k)
        ]

        search_text = semantic.search_query
        if arabic_keywords:
            search_text += " " + " ".join(arabic_keywords)
        print(f"  [Search]    {search_text}")
        print(f"  [Filter]    {qdrant_filter or 'none'}")

        candidates = _vectorstore.similarity_search(
            query=search_text,
            k=k_candidates,
            filter=qdrant_filter,
        )
        reranked = _compressor.compress_documents(candidates, semantic.search_query)

    except Exception:
        candidates = _vectorstore.similarity_search(user_query, k=k_candidates)
        reranked = _compressor.compress_documents(candidates, user_query)

    return sorted(reranked, key=lambda d: d.metadata.get("relevance_score", 0), reverse=True)
