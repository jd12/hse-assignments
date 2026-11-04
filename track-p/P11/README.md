# P11 · Hybrid Search on Your Chunks

**Meetings:** D34 · **Points:** 10 pts

**Watch — none**

[Boot.dev, *Learn Retrieval Augmented Generation*](https://www.boot.dev/courses/learn-retrieval-augmented-generation) · chapter 6 Hybrid Search, lessons 4 and 5, about 20 minutes.

**Notes**

**Rank fusion throws the scores away.** Each engine hands over an ordered list; a passage at rank `r` in a list gets `1 / (k + r)` from it, and the fused score is the sum. No normalization, no `alpha`, nothing that depends on the scale of either engine. The chapter's `k` is 60, and the largest fused score two lists can give is `2 / 61`, about 0.0328, for a chunk at rank 1 in both.

**From here on, everything runs on chunks.** Passages are finished. `rag/search.py` builds both engines over the chunks from P09, with your frozen `SIZE` and `OVERLAP`, and the scoreboard moves to chunks today. Your needles survive the move because the overlap is longer than any of them.

**The chunk vectors are a second embedding run.** `rag/search.py` names the vector file after the configuration, `data/emb_chunks_200_50.npy` for the defaults, and P08's guard re-embeds if the chunks change. The first call embeds your chunks once; on Homer that is 1,882 texts, about the same work as the passages.

**Walkthrough — the chapter, then fusion on your chunks**

**Step 1. Branch, log.**
Same branch as P10:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch dev/hybrid && git pull
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Chapter 6, lessons 4 and 5, every lesson green; the chapter page saved
- [ ] rag/hybrid.py and rag/search.py: the scale check on one query (Steps 3 and 4)
- [ ] Scoreboard on chunks, three methods, my P10 predictions checked (Step 5)
- [ ] Fill every slot in evidence/P11.md (check_evidence.py: all slots filled); push, sign off
```

**Step 2. Finish chapter 6: lessons 4 and 5.**

In `~/version_control/rag-search-engine/`. When the chapter page shows every lesson complete, save it as `evidence/p11-bootdev-ch6.png` in the rag repo.

**Lesson 4: the weighted combination.** Both engines run, both score lists are normalized with lesson 3's function, and each movie's hybrid score is `alpha * bm25 + (1 - alpha) * semantic`. `alpha = 1` is keyword search; `alpha = 0` is semantic.

1. In `cli/lib/hybrid_search.py`, above the class:

```python
# In: a normalized BM25 score, a normalized semantic score, and alpha.  Out: their weighted mix.
def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score
```

2. Replace the `weighted_search` stub. Each engine is asked for 500 times the limit, so a movie near the top of one list and far down the other is still in both. The dictionary collects every movie either engine returned, with `0.0` for the engine that did not return it:

```python
    # In: a query, alpha, how many results.  Out: every movie either engine returned, with both normalized scores and the mix, best mix first.
    def weighted_search(self, query, alpha, limit=5):
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        bm25_scores = normalize_scores([score for movie, score in bm25_results])
        semantic_scores = normalize_scores([r["score"] for r in semantic_results])
        combined = {}
        for (movie, score), norm in zip(bm25_results, bm25_scores):
            combined[movie["id"]] = {"doc": movie, "bm25": norm, "semantic": 0.0}
        for r, norm in zip(semantic_results, semantic_scores):
            entry = combined.setdefault(r["id"], {"doc": self.semantic_search.document_map[r["id"]], "bm25": 0.0, "semantic": 0.0})
            entry["semantic"] = norm
        for entry in combined.values():
            entry["hybrid"] = hybrid_score(entry["bm25"], entry["semantic"], alpha)
        return sorted(combined.values(), key=lambda e: e["hybrid"], reverse=True)
```

`setdefault` here is the same trick as the inverted index: it returns the entry already there, or stores and returns a new one with both scores at `0.0`.

3. In the CLI, `HybridSearch` and `load_movies` imported, a `weighted-search` subcommand with `query`, `--alpha` (default 0.5) and `--limit` (default 5), and a case that prints the top `limit` the chapter's way:

```python
from lib.hybrid_search import HybridSearch, normalize_scores
from lib.search_utils import load_movies
```

```python
    weighted_parser = subparsers.add_parser("weighted-search", help="Hybrid search with a weighted score")
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument("--alpha", type=float, default=0.5, help="Weight of the keyword score, 0 to 1")
    weighted_parser.add_argument("--limit", type=int, default=5, help="How many results")
```

```python
        case "weighted-search":
            searcher = HybridSearch(load_movies())
            for i, entry in enumerate(searcher.weighted_search(args.query, args.alpha, args.limit)[:args.limit], start=1):
                print(f"{i}. {entry['doc']['title']}")
                print(f"   Hybrid Score: {entry['hybrid']:.3f}")
                print(f"   BM25: {entry['bm25']:.3f}, Semantic: {entry['semantic']:.3f}")
                print(f"   {entry['doc']['description'][:100]}...")
```

4. Run, then submit:

```bash
uv run cli/hybrid_search_cli.py weighted-search "British Bear" --alpha 0.5 --limit 25
uv run cli/hybrid_search_cli.py weighted-search "British Bear" --alpha 0.2 --limit 25
uv run cli/hybrid_search_cli.py weighted-search "British Bear" --alpha 0.8 --limit 25
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* `Paddington` first in all three, at `1.000` for both scores when it tops both lists as the chapter's example shows, and the rest of the list reshuffled as `alpha` moves: `The Country Bears` and `Legends of the Fall` in the 25 at `0.5`, `The Edge` appearing at `0.2`, `The Duchess` and `The Great Bear` at `0.8`; those are the checks. *If it broke* with `KeyError` in the semantic loop, `document_map` was not filled: `load_or_create_chunk_embeddings` has to set it, which lesson 4 of chapter 5 did.

**Lesson 5: reciprocal rank fusion.** No scores, no normalization. Each list votes `1 / (k + rank)` for every movie it ranked, and the votes are added.

1. Above the class:

```python
# In: a rank (1 is first) and k.  Out: that rank's vote.
def rrf_score(rank, k=60):
    return 1 / (k + rank)
```

2. Replace the `rrf_search` stub. The same shape as `weighted_search`, with ranks where the scores were:

```python
    # In: a query, k, how many results.  Out: every movie either engine returned, with both ranks and the summed vote, best first.
    def rrf_search(self, query, k, limit=10):
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        combined = {}
        for rank, (movie, score) in enumerate(bm25_results, start=1):
            combined[movie["id"]] = {"doc": movie, "bm25_rank": rank, "semantic_rank": None, "rrf": rrf_score(rank, k)}
        for rank, r in enumerate(semantic_results, start=1):
            entry = combined.setdefault(r["id"], {"doc": self.semantic_search.document_map[r["id"]], "bm25_rank": None, "semantic_rank": None, "rrf": 0.0})
            entry["semantic_rank"] = rank
            entry["rrf"] += rrf_score(rank, k)
        return sorted(combined.values(), key=lambda e: e["rrf"], reverse=True)
```

3. In the CLI, an `rrf-search` subcommand with `query`, `-k` (default 60) and `--limit` (default 5), and its case. A rank of `None` prints as `-`:

```python
    rrf_parser = subparsers.add_parser("rrf-search", help="Hybrid search with reciprocal rank fusion")
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("-k", type=int, default=60, help="The RRF constant")
    rrf_parser.add_argument("--limit", type=int, default=5, help="How many results")
```

```python
        case "rrf-search":
            searcher = HybridSearch(load_movies())
            for i, entry in enumerate(searcher.rrf_search(args.query, args.k, args.limit)[:args.limit], start=1):
                print(f"{i}. {entry['doc']['title']}")
                print(f"   RRF Score: {entry['rrf']:.3f}")
                print(f"   BM25 Rank: {entry['bm25_rank'] or '-'}, Semantic Rank: {entry['semantic_rank'] or '-'}")
                print(f"   {entry['doc']['description'][:100]}...")
```

4. Run, then submit:

```bash
uv run cli/hybrid_search_cli.py rrf-search "family fighting movie" --limit 25
uv run cli/hybrid_search_cli.py rrf-search "family fighting movie" --limit 25 -k 5
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* `Anjali`, `The Spy Next Door` and `Kung Pow: Enter the Fist` in the first 25; that is the check. A movie at rank 1 in both lists shows `0.033`, which is `2 / 61`. The second run, with a small `k`, makes rank 1 worth far more than rank 10, and the top of the list changes; write what moved in `scratch/p10-hybrid.md`.

**Step 3. `rag/hybrid.py` and `rag/search.py`.**

Open `rag/hybrid.py`: `rrf` is lesson 5 on any number of ranked lists of ids, with the P06 tie rule, and `K` is the chapter's constant; change it if yours differs. Open `rag/search.py`: `engines` builds the chunks from `rag/chunking.py` with your frozen `SIZE` and `OVERLAP`, a `Keyword` over them and a `Semantic` over them, once per run; `keyword`, `semantic` and `hybrid` are the three searches, and `hybrid` asks each engine for `POOL` results and fuses the two id lists. Then:

```bash
uv run python -m rag.search <one of your five queries>
```

*You should see* `embedding <your chunk count> units with all-MiniLM-L6-v2` and a progress bar the first time, then three lines: BM25 ids with scores in the single or double digits, semantic ids with scores below 1, and hybrid ids with no scores at all. Run it again: no bar. The hybrid ids are drawn from both lists; a chunk high in both is first. Paste the three lines under `## Step 3` in `evidence/P11.md`. On Homer, `how was the Cyclops blinded`:

```text
bm25     [('c00308', 10.04), ('c00023', 7.91), ('c00305', 6.71), ('c00823', 6.63), ('c00280', 6.45)]
semantic [('c00300', 0.609), ('c00341', 0.56), ('c00292', 0.548), ('c00659', 0.543), ('c00307', 0.54)]
hybrid   ['c00300', 'c00308', 'c00307', 'c00305', 'c00301']
```

The two engines disagree on rank 1 and the fused list keeps both, `c00300` then `c00308`, with the chunks they share next.

*If it broke* with `AssertionError: overlap ... must be smaller than size`, `rag/chunking.py`'s frozen values are wrong. *If it broke* with the bar running on every call, the chunk text changed between calls, which it cannot unless `SIZE` or `OVERLAP` did.

**Step 4. The scale problem, settled.**

The same query, the two engines' raw scores and what `rrf` does instead:

```bash
uv run python -c "
from rag import search
from rag.hybrid import rrf, rrf_score
q = '<the same query>'
kw = [r['id'] for r in search.keyword(q, 10)]
se = [r['id'] for r in search.semantic(q, 10)]
fused = rrf([kw, se], top=5)
print('keyword ', kw)
print('semantic', se)
print('fused   ', fused)
for cid in fused[:3]:
    votes = [rrf_score(ranking.index(cid) + 1) for ranking in (kw, se) if cid in ranking]
    print(cid, 'ranks', [ranking.index(cid) + 1 if cid in ranking else '-' for ranking in (kw, se)], 'score', round(sum(votes), 4))
"
```

*You should see* the two id lists, the fused top five, and for the first three fused ids their rank in each list and the summed vote. A chunk at rank 1 and rank 1 shows `0.0328`, and a chunk in only one list shows one vote. Paste it under `## Step 4`. On Homer:

```text
keyword  ['c00308', 'c00023', 'c00305', 'c00823', 'c00280', 'c00307', 'c00301', 'c00300', 'c00828', 'c00306']
semantic ['c00300', 'c00341', 'c00292', 'c00659', 'c00307', 'c00306', 'c00301', 'c00308', 'c00305', 'c01431']
fused    ['c00300', 'c00308', 'c00307', 'c00305', 'c00301']
c00300 ranks [8, 1] score 0.0311
c00308 ranks [1, 8] score 0.0311
c00307 ranks [6, 5] score 0.0305
```

`c00300` and `c00308` are mirror images, rank 1 in one list and rank 8 in the other, so their votes are the same `1/61 + 1/68`. The tie goes to the one that comes first in the engines' order, the P06 rule, which is why `c00300` leads. Nothing in a raw score of `10.04` against `0.609` could have produced that comparison. *If it broke* with `ValueError: ... is not in list` the `if cid in ranking` guard was edited out.

**Step 5. The scoreboard on chunks.**

Replace `methods()` and the top of `rag/scoreboard.py`. Passages are finished; from here every method runs on chunks:

```python
from rag import search
from rag.chunking import OVERLAP, SIZE, load_chunks
```

```python
def methods():
    units = load_chunks(SIZE, OVERLAP)
    for q in QUERIES:
        assert any(norm(q["needle"]) in norm(u["text"]) for u in units), f"{q['id']}: needle not in any chunk"
    return {
        "bm25": lambda q: search.keyword(q, 5),
        "semantic": lambda q: search.semantic(q, 5),
        "hybrid": lambda q: search.hybrid(q, 5),
    }
```

Change `from rag.corpus import ROOT, load_passages` to `from rag.corpus import ROOT`, and delete the `from rag.keyword import Keyword` and `from rag.semantic import Semantic` lines; `__main__` stays as it was. Your predictions are already under `## Predictions` in `evidence/P10.md`, committed. Then:

```bash
uv run python -m rag.scoreboard
```

*You should see* three columns on chunks, and every needle found in some chunk, or the `assert` would have stopped it. The ranks will not match P08's: the units changed. Expect hybrid to equal the better of its two inputs on most queries and to beat both on at most one or two; expect at least one query where it is worse than one input, because a chunk only one engine ranked gets one vote. Paste the table under `## Step 5` and check your five predictions, one line each. On Homer:

```text
query      bm25 semantic   hybrid
q01           -        -        -
q02           -        5        -
q03           -        1        1
q04           1        2        1
q05           -        -        3
hit@3       1/5      2/5      3/5
```

Compare this with P08's table. `q03`, outside the top 30 on passages, is rank 1 on chunks: the needle that sat past the model's window at word 206 of a 375-word passage now sits inside a 200-word chunk the model reads whole. That is the chapter 5 argument in one row. `q05` is the case where hybrid beat both inputs: BM25 had it outside the top 5 and semantic had it at rank 8, and two weak votes outranked every chunk that only one engine liked.

**Where it got worse.** Find the query where hybrid ranked the answer lower than the better of BM25 and semantic did. Paste the three ranks, and say what RRF did to a chunk only one list found. On Homer it is `q02`: BM25 `-`, semantic 5, hybrid `-`. The answer chunk had one vote, `1/65` from semantic, for `0.0154`. The fused rank 1, `c00542`, was rank 15 for BM25 and rank 10 for semantic, two middling votes for `0.0276`, and the whole fused top five is chunks like it, ranked somewhere in both of the 50-deep pools. Neither engine liked any of them more than semantic liked the answer, and the answer still lost, because one good vote is worth less than two mediocre ones.

**Commit, push, sign off.** Fill `## Reflection` in `evidence/P11.md` (the questions are below) and run `uv run python scripts/check_evidence.py P11` until it says `all slots filled`. Then:

```bash
git add rag/scoreboard.py scratch/ evidence/
git commit -m "P11: RRF hybrid search over chunks; scoreboard on chunks"
git push
```

The P10 pull request is already open; these commits land on it. Edit its body to name the query where fusion lost.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`evidence/P11.md` with every slot filled + `rag/scoreboard.py` on chunks with three methods + `scratch/p10-hybrid.md` + `evidence/p11-bootdev-ch6.png`.

**Reflection Questions** (answer under `## Reflection` in `evidence/P11.md`)

1. Paste your Step 4 output. For the first fused chunk, show the arithmetic of its score from its two ranks and `k = 60`, and say what its score would be with `k = 5`, and whether it would still be first.

   *How to get it:* `1 / (60 + r1) + 1 / (60 + r2)`; then the same with 5. With a small `k`, a rank-1 vote is worth about three rank-3 votes; with 60, they are nearly equal.

2. Paste your scoreboard and your P10 predictions. For the "where it got worse" query, paste the three ranks and say which engine's list did not have the answer chunk, and what a second engine would have needed to rank it for fusion to help.

   *How to get it:* `uv run python -m rag.search <that query>` shows the two lists; the answer chunk's id is in one and not the other, or low in both.

3. Lesson 4's weighted search and lesson 5's RRF are two answers to lesson 3's problem. Say, with your Step 4 numbers, what RRF never has to know about the two engines that weighted search does, and name one thing weighted search can express that RRF cannot.

   *How to get it:* RRF reads only positions, so it never sees a 14.9 or a 0.41 and never needs them normalized. Weighted search has `alpha`, which can say "trust keyword more today"; RRF treats both lists as equal voters.
