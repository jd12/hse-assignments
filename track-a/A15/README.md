# A15 · Semantic Search and the Embedding Space

**Meetings:** D30 · **Points:** 15 pts

**Watch**
Thu Nov 5 — [Boot.dev, *Learn Retrieval Augmented Generation*](https://www.boot.dev/courses/learn-retrieval-augmented-generation) ch. 4 Semantic Search, about 25 minutes of lessons

**During the video**

Same folder, `~/version_control/bootdev-rag/`. Type every solution, `bootdev run <lesson-id>` before `-s`.

**Ch. 4 · write down the model.** When the chapter loads an embedding model, write its name and the dimension of the vectors it produces into `scratch/a15-models.md` in your rag repo. Under it, write the model you have used since A05 (`text-embedding-3-small`) and its dimension from your A05 `FINDINGS.md`. They will not match. Step 2 is about why that is fine and what would make it not fine.

**Ch. 4 · find the normalize.** Somewhere in the chapter the vectors are normalized, or the model is asked to return them normalized. Write down the line that does it. If you cannot find one, write that down too, and check whether the chapter's similarity scores stay between -1 and 1.

**Notes**

You built cosine search by hand in A06. This is where the same idea becomes a bug source in a real pipeline, and three things break it.

**Mixing models.** Every embedding model outputs vectors of one fixed length. Embed passages with one model and queries with another and you get either a shape error or, if the lengths happen to match, meaningless numbers that rank confidently. The walkthrough stores the model name next to the vectors and refuses to load vectors made by a different model or from different text.

**Skipping normalization.** Cosine similarity is the dot product of unit vectors. Skip the normalize and you rank by raw dot product, which rewards long vectors. A similarity of `47.3` means you did not normalize. Every score this walkthrough prints is between -1 and 1.

**Re-embedding.** Embed the corpus once, save it, load it after that. The walkthrough caches every query vector too, because you will run the same ten queries dozens of times this month. Your key has a hard cap and it does not warn you.

**Walkthrough — semantic search on your corpus**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch main && git pull
git switch dev/keyword && git pull      # skip this line if A14 has merged
git switch -c dev/semantic
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Boot.dev ch. 4 Semantic Search, every lesson green
- [ ] `scratch/a15-models.md`: both models, both dimensions, the normalize line
- [ ] `rag/semantic.py`: embed once, cached, guarded (Steps 2–3)
- [ ] Self-retrieval check on three passages (Step 4)
- [ ] Semantic column on the scoreboard (Step 5)
- [ ] Extension: hits vs misses by score, three failures re-run
- [ ] Push, PR, sign off
```

**Step 2. `rag/semantic.py`.**

The `embed` function is your A05 one with a loop around it: the endpoint takes a list, so passages go up in batches of a hundred rather than one request each.

```python
# rag/semantic.py
import hashlib
import json
import os
import sys
import urllib.request

import numpy as np

from rag.corpus import ROOT, load_passages

MODEL = "text-embedding-3-small"
DATA = ROOT / "data"
QCACHE = DATA / "query_cache.json"

def embed(texts, batch=100):
    out = []
    for start in range(0, len(texts), batch):
        body = json.dumps({"model": MODEL, "input": texts[start:start + batch]}).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings", data=body,
            headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            items = json.load(r)["data"]
        items.sort(key=lambda it: it["index"])
        out.extend(it["embedding"] for it in items)
        print(f"  embedded {min(start + batch, len(texts))}/{len(texts)}", file=sys.stderr)
    return np.array(out, dtype=np.float32)

def normalize(V):
    return V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-10)

class Semantic:
    def __init__(self, units, name):
        self.units = units
        fingerprint = hashlib.sha256("\x00".join(u["text"] for u in units).encode()).hexdigest()
        vec_path, meta_path = DATA / f"emb_{name}.npy", DATA / f"emb_{name}.json"
        meta = {"model": MODEL, "count": len(units), "sha256": fingerprint}
        if vec_path.exists() and meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) == meta:
            V = np.load(vec_path)
        else:
            print(f"embedding {len(units)} units with {MODEL} (this costs money)", file=sys.stderr)
            V = embed([u["text"] for u in units])
            np.save(vec_path, V)
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
        assert V.shape[0] == len(units), f"{V.shape[0]} vectors for {len(units)} units"
        self.Vn = normalize(V)

    def query_vector(self, query):
        cache = json.loads(QCACHE.read_text(encoding="utf-8")) if QCACHE.exists() else {}
        key = f"{MODEL}::{query}"
        if key not in cache:
            cache[key] = embed([query])[0].tolist()
            QCACHE.write_text(json.dumps(cache), encoding="utf-8")
        return normalize(np.array(cache[key], dtype=np.float32))

    def search(self, query, k=5):
        sims = self.Vn @ self.query_vector(query)
        top = np.argsort(-sims, kind="stable")[:k]
        return [dict(self.units[i], score=float(sims[i])) for i in top]

if __name__ == "__main__":
    sem = Semantic(load_passages(), "passages")
    print("matrix:", sem.Vn.shape)
    for r in sem.search(" ".join(sys.argv[1:])):
        print(f"{r['score']:.3f}  {r['id']}  {r['text'][:70]}")
```

Read `__init__` before you run it. The vectors are saved under `data/`, which is git-ignored, next to a small JSON file holding the model name, the passage count, and a hash of every passage's text. The vectors are only loaded if all three match. Change the model, change `MIN_CHARS`, or change the corpus, and the hash changes and it re-embeds rather than silently pairing new text with old vectors. `np.argsort(-sims)` is descending; `kind="stable"` breaks ties by position, the same rule as A14.

**Step 3. Embed once.**

```bash
uv run python -m rag.semantic where does the story begin
```

*You should see* the `embedding ... (this costs money)` line and a progress count the first time, then `matrix: (<your passage count>, 1536)`, then five results with scores between 0 and 1. Run the same command again. *You should see* no `embedding ... (this costs money)` line and no `embedded .../<your passage count>` count: it loaded from disk. (A new query still prints `embedded 1/1`, for the query itself.)

*If it broke:* `KeyError: 'OPENAI_API_KEY'` means this terminal did not load your shell profile. `HTTP Error 400` on the first batch usually means a passage is enormous; your longest passage from `rag.corpus` will tell you. `HTTP Error 429` is rate limiting; wait a minute and re-run, since finished batches are not saved and it will start again.

**Step 4. Self-retrieval, on three passages.**

The A06 check, because it is the one that catches row misalignment:

```bash
uv run python -c "
from rag.corpus import load_passages
from rag.semantic import Semantic
ps = load_passages()
sem = Semantic(ps, 'passages')
for i in (0, len(ps) // 2, len(ps) - 1):
    top = sem.search(ps[i]['text'], k=2)
    print(ps[i]['id'], '->', top[0]['id'], round(top[0]['score'], 4), '| next', round(top[1]['score'], 4))
"
```

*You should see* each id map to itself at rank 1 with a score of 0.999 or higher, and a second-best score clearly below it. The first, the middle and the last, because an off-by-one only shows at an end.

*If it broke:* a different id at rank 1 means `V[i]` is not the embedding of `units[i]`. The usual cause is removing the `items.sort` line. Delete `data/emb_passages.*` after you fix it, so the bad vectors are not loaded again.

**Step 5. Add semantic search to the scoreboard.**

In `rag/scoreboard.py`, import `Semantic` and add one entry to the dictionary `methods()` returns:

```python
from rag.semantic import Semantic
```

```python
    sem = Semantic(units, "passages")
    return {
        "tfidf": lambda q: kw.search(q, 5, method="tfidf"),
        "bm25": lambda q: kw.search(q, 5),
        "semantic": lambda q: sem.search(q, 5),
    }
```

Before you run it, write in `scratch/a15-models.md` the query ids you expect semantic search to find in the top 3 and the ones you expect it to miss. Commit that file first; I will check your commit timestamps.

```bash
git add scratch/a15-models.md && git commit -m "A15: semantic predictions before the run"
uv run python -m rag.scoreboard
```

*You should see* the A14 columns unchanged, which proves the queries and needles did not move, and a third column. Semantic search should find at least one of your three `paraphrase` queries that keyword search could not, because it is comparing meanings rather than stems. If it finds none of them, look at the rank it did give each; a `4` is a different story from a `-`.

```bash
git add rag/semantic.py rag/scoreboard.py
git commit -m "A15: cached semantic search, guarded by model and text hash"
git push -u origin dev/semantic
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**.

**Extension — where the scores sit, and what semantic search fixed (ASSIGNED)**

**Part 1. The scoreboard.** Paste the new table into `rag/SCOREBOARD.md` under `## A15`. Under it, one line per method: hit@3 out of 10.

**Part 2. Scores on hits and misses.** For each of your ten queries, print the top semantic score, and mark whether the answer passage was in the top 3:

```bash
uv run python -c "
from rag.corpus import load_passages
from rag.semantic import Semantic
from rag.scoreboard import QUERIES, first_hit
sem = Semantic(load_passages(), 'passages')
for q in QUERIES:
    res = sem.search(q['query'], 5)
    r = first_hit(res, q['needle'])
    print(q['id'], q['kind'].ljust(12), 'hit' if r and r <= 3 else 'miss', round(res[0]['score'], 3))
"
```

*You should see* every top score well above zero, misses included. That is the floor you measured in A05: `text-embedding-3-small` puts even unrelated English text at a clearly positive similarity, so a miss does not score near 0. Put your A05 anisotropy number next to this table in `SCOREBOARD.md`. The question the table answers: is there a score that separates your hits from your misses? Draw the line if there is one; say there is not if the ranges overlap, and by how much.

**Part 3. Re-run three failures.** Take three queries from `rag/failures_wk7.md` and run each through `uv run python -m rag.semantic <query>`. Append a section `## A15 re-run` to that file: for each, the top semantic result (id and first 100 characters), whether it answers the query, and one sentence on why semantic search fixed it or did not.

**Part 4. Where it got worse.** Your two `rare-name` queries: compare the semantic rank to the BM25 rank. A name that appears in one or two passages is a single rare token to BM25 and a small part of a 1,536-number summary to an embedding. If semantic search lost the name on either query, paste the passage it put at rank 1 instead and say what that passage has in common with the query besides the name. If it lost neither, write one new rare-name query from your corpus where it does, and add it to the section (not to `queries.json`, which stays at ten).

```bash
git add rag/SCOREBOARD.md rag/failures_wk7.md evidence/a15-bootdev-ch4.png
git commit -m "A15: semantic column, score ranges, failures re-run"
git push
```

Then close the log: `bash scripts/sign-off.sh`, and `git add logs && git commit && git push` in the log repo.

**Deliverable**
`rag/semantic.py` + `rag/scoreboard.py` with three methods + `rag/SCOREBOARD.md` (A15 section with the hit/miss score table) + `rag/failures_wk7.md` with the A15 re-run + `scratch/a15-models.md` + `evidence/a15-bootdev-ch4.png`.

**Reflection Questions**

1. Paste your three self-retrieval lines from Step 4. What was the gap between the rank-1 score and the next score on each? Then embed the same passage with one character changed (add a period at the end) by searching for that text, and paste the new rank-1 score. What does the size of that change tell you about how a 0.97 in A06's check should be read?
2. Paste your Part 2 table and your A05 anisotropy number. Where did you draw the hit/miss line, or how much do the two ranges overlap? If you used your line as a "no good answer" cutoff on these ten queries, how many right answers would it throw away and how many wrong ones would it keep? Count them.
3. Paste the prediction you committed in `scratch/a15-models.md` before Step 5, then the scoreboard row for each query where it was wrong. Pick one of them and explain, from the passage it returned, what the embedding matched on.
