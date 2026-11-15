from search import search

query = "Могу ли я запустить БП Командировку, если не заложил средства в бюджетную сделку"
hits = search(query, top_k=5)

print(f"Вопрос: {query}\n")
for i, h in enumerate(hits, 1):
    print(f"{i}. score={h['score']:.4f} | {h['source']} | {h['text'][:120]}...")
    print()