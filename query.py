# query.py — manual retrieval test script
#
# Imports the production retrieve() function from retrieval.py and
# prints results to stdout for inspection. Not imported by other modules.

from rag_chain import answer , answer_stream
from retrieval import retrieve

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
q = "ما هي الفرق بين خطط خدمة البث التدفقي لأبل وديزني+؟"

full_answer =[]
for chunk in answer_stream(q):
    if chunk['type'] == 'token':
            full_answer.append(chunk['text'])
print("".join(full_answer))