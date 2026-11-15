import sys

from search import (
    CONFIDENCE_THRESHOLD,
    SOFT_MARGIN,
    _expand_abbr,
    _key_words,
    accepts,
    detect_system,
    search,
    select_hits,
)

sys.stdout.reconfigure(encoding="utf-8")


def read_query() -> str:
    raw = open("q.txt", "rb").read()
    try:
        return raw.decode("utf-8-sig").strip()
    except UnicodeDecodeError:
        return raw.decode("cp1251").strip()


query = _expand_abbr(read_query())
qwords = _key_words(query)

print("query      :", query)
print("system     :", detect_system(query))
print("key words  :", sorted(qwords))
print("threshold  :", CONFIDENCE_THRESHOLD, "soft zone from", round(CONFIDENCE_THRESHOLD - SOFT_MARGIN, 4))

for title, ctype in (("ALL TYPES", None), ("TEXT ONLY", "text")):
    hits = search(query, top_k=8, chunk_type=ctype)
    print("\n==", title)
    for h in hits:
        mark = "OK " if accepts(h, qwords) else "-- "
        print(f"{mark}{h['score']:.4f} {h['type']:5} p{h['page']} | {h['section']} | {h['text'][:110]!r}")

print("\naccepted by select_hits:", len(select_hits(search(query, top_k=8), query)))