# query.py — manual retrieval test script
#
# Imports the production retrieve() function from retrieval.py and
# prints results to stdout for inspection. Not imported by other modules.

from rag_chain import answer

# ── Helper ────────────────────────────────────────────────────────────────────
# def print_results(query: str, results: list):
#     print(f"Query : {query}")
#     print(f"Found : {len(results)} chunk(s)\n")
#     for i, doc in enumerate(results):
#         m = doc.metadata
#         print(f"  [{i+1}]  id={m['id']}  |  {m['date']}  |  score={m.get('relevance_score', 'n/a'):.3f}  |  {m['title'][:60]}")
#         print(f"       {doc.page_content[:300].strip()} ...")
#         print(f"       url: {m.get('url', 'n/a')}")
#         print()

# ── Test queries ──────────────────────────────────────────────────────────────
q = " اسعار النفط في عام 2020"

result = answer(q)
print(result)

