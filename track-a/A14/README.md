# A14 · Keyword Search and Its Ceiling

**Meetings:** D29 · **Points:** 15 pts

**Watch**
Wed Nov 4 — [Boot.dev, *Learn Retrieval Augmented Generation*](https://www.boot.dev/courses/learn-retrieval-augmented-generation) ch. 3 Keyword Search, about 25 minutes of lessons

**During the video**

Same folder as A13, `~/version_control/bootdev-rag/`. Type every solution yourself, run each lesson with `bootdev run <lesson-id>` before you submit it with `-s`.

**Ch. 3 · write the constants down.** The chapter builds BM25, and BM25 has two tuning constants, `k1` and `b`. When the chapter gives you their values, write them into `scratch/a14-bm25.md` in your rag repo, along with one sentence for each saying what it controls in the chapter's own words. Step 3 uses those values.

**Ch. 3 · predict one ranking.** Before the lesson that first ranks documents, pick a query of two words, one common and one rare, and write in `scratch/a14-bm25.md` which word you think will decide the ranking. Check it when the lesson runs.

**Notes**

**Sort direction.** `sorted(scores)` is ascending. It hands you the *worst* k documents, and everything downstream looks mysteriously stupid with no error anywhere. Negate the score in the sort key.

**Ties must break the same way every time.** Two passages with identical scores have to come back in a fixed order, or a test that passes on your machine fails on the grader. The walkthrough breaks ties by passage position, `(-score, i)`. Python's sort is stable, and relying on that by accident is how a tie breaks differently after an unrelated change.

**The query goes through the same `preprocess` as the passages.** Lowercase the documents and not the query, and every capitalized query term scores zero. This is the most common RAG bug there is and it produces no error message. The walkthrough calls one function for both, with the same arguments, stored on the index.

**A query made of stopwords is an empty list of terms.** Every passage scores zero. Return nothing rather than five arbitrary passages with a score of 0.0.

**Walkthrough — BM25 on your corpus**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. A13 is not due until tomorrow morning, so it has almost certainly not merged. Branch from it:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch main && git pull
git switch dev/tfidf && git pull        # skip this line if A13 has merged
git switch -c dev/keyword
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Boot.dev ch. 3 Keyword Search, every lesson green
- [ ] `scratch/a14-bm25.md`: k1, b, and my two-word prediction
- [ ] `rag/keyword.py` running, three invariants checked (Steps 2–5)
- [ ] `rag/queries.json` written and committed BEFORE the scoreboard runs
- [ ] `rag/scoreboard.py`: tfidf vs bm25, hit@3
- [ ] `rag/failures_wk7.md`: five failures, each with its mechanism
- [ ] Push, PR, sign off
```

**Step 2. `rag/keyword.py`.**

Two scorers over the A13 index. `tfidf_score` is the obvious one: add up the tf-idf of each query term in the passage. `bm25` is the chapter's: term frequency saturates (the tenth occurrence of a word adds much less than the first) and long passages are penalized for being long.

```python
# rag/keyword.py
import math
import sys

from rag.corpus import load_passages
from rag.preprocess import preprocess
from rag.tfidf import TfidfIndex

class Keyword:
    def __init__(self, units, k1=1.5, b=0.75, **pp):
        self.ix = TfidfIndex(units, **pp)
        self.units = units
        self.k1, self.b = k1, b
        self.avgdl = sum(self.ix.lengths) / self.ix.N

    def bm25_idf(self, term):
        n, N = self.ix.df[term], self.ix.N
        return math.log((N - n + 0.5) / (n + 0.5) + 1)

    def bm25(self, terms, i):
        doc, dl = self.ix.docs[i], self.ix.lengths[i]
        score = 0.0
        for t in terms:
            f = doc[t]
            if f:
                norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += self.bm25_idf(t) * f * (self.k1 + 1) / (f + norm)
        return score

    def tfidf_score(self, terms, i):
        return sum(self.ix.tfidf(t, i) for t in terms)

    def search(self, query, k=5, method="bm25"):
        terms = preprocess(query, **self.ix.pp)
        score = self.bm25 if method == "bm25" else self.tfidf_score
        scored = [(score(terms, i), i) for i in range(self.ix.N)]
        scored = [(s, i) for s, i in scored if s > 0]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [dict(self.units[i], score=s) for s, i in scored[:k]]

if __name__ == "__main__":
    kw = Keyword(load_passages())
    query = " ".join(sys.argv[1:])
    print("terms:", preprocess(query))
    for method in ("tfidf", "bm25"):
        print(f"--- {method}")
        for r in kw.search(query, method=method):
            print(f"{r['score']:7.3f}  {r['id']}  {r['text'][:70]}")
```

Replace `1.5` and `0.75` with the chapter's `k1` and `b` if they differ, and `bm25_idf` with the chapter's BM25 idf if it differs. `**pp` is how the A13 extension's corpus stopwords reach the query: whatever you pass to `Keyword(...)` is stored on the index and used for both sides.

**Step 3. Invariant one: a rare name finds its passage.**

Find three words that appear in exactly one passage of your corpus:

```bash
uv run python -c "
from rag.corpus import load_passages
from rag.keyword import Keyword
kw = Keyword(load_passages())
rare = [t for t, n in kw.ix.df.items() if n == 1 and t.isalpha() and len(t) > 6][:3]
for t in rare:
    hits = kw.search(t)
    print(t, len(hits), hits[0]['id'], round(hits[0]['score'], 2))
"
```

*You should see* each term with a hit count of exactly `1`. A term in one passage can only score in that passage, so it is rank 1 and nothing else scores at all. These are stems, so `t` may look clipped. Now search a real rare proper noun from your corpus, as you would type it:

```bash
uv run python -m rag.keyword <a name that appears once in your corpus>
```

*You should see* the passage containing it at rank 1 under both methods. If it is not, your query did not preprocess the same way as the passages; the `terms:` line tells you what your query became.

**Step 4. Invariant two: stopwords score nothing.**

```bash
uv run python -m rag.keyword the and of
```

*You should see* `terms: []` and two empty result lists. If you see five passages with a score of `0.000`, you did not drop zero scores.

**Step 5. Invariant three: the two methods disagree in a particular way.**

Run a two-word query with one common word from your corpus and one rare one, the pair you predicted during the chapter. Compare the two lists.

*You should see* BM25 put the rare word's passage at rank 1. Plain tf-idf often does not: its `tf` divides by passage length, so a short passage that repeats the common word can outscore the long passage that holds the rare word once. That difference is the whole point of the chapter's two constants. Below rank 1 the orders differ too; where they do, BM25 prefers the shorter passage and the one where the common word does not repeat ten times. Check your prediction in `scratch/a14-bm25.md` and write one line under it: right or wrong.

**Step 6. Commit the engine.**

```bash
git add rag/keyword.py scratch/a14-bm25.md
git commit -m "A14: BM25 and tf-idf search over passages"
git push -u origin dev/keyword
```

Open the pull request now: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**.

**Extension — ten queries and a scoreboard (ASSIGNED)**

You are writing the ten queries that every retrieval method from here to A17 is scored on. They go in a file and get committed **before** anything scores them. I will check your commit timestamps.

**Part 1. Write `rag/queries.json`.** Ten entries, this shape:

```json
[
  {"id": "q01", "query": "what does the hero call himself to the one-eyed giant",
   "needle": "my name is Noman", "kind": "paraphrase"}
]
```

`needle` is four to ten consecutive words copied exactly from the passage that answers the query, all from within one paragraph. It is how the scoreboard knows a result is right without you marking it by hand, and it keeps working when A16 replaces passages with chunks. The mix is fixed:

| How many | `kind` | What it is |
|---|---|---|
| 3 | `paraphrase` | the answer passage shares **no content words** with the query. Check it: run `preprocess` on both and confirm the lists do not overlap. |
| 2 | `rare-name` | the query names a person, place or thing that appears in only a few passages. |
| 5 | your choice | anything a real reader of your corpus would ask. Say the kind in one word. |

The example above is an Odyssey query. Yours are about your corpus.

```bash
git add rag/queries.json
git commit -m "A14: ten scoring queries, written before any run"
git push
```

**Part 2. `rag/scoreboard.py`.**

```python
# rag/scoreboard.py
import json

from rag.corpus import ROOT, load_passages
from rag.keyword import Keyword

QUERIES = json.loads((ROOT / "rag" / "queries.json").read_text(encoding="utf-8"))

def norm(s):
    return " ".join(s.lower().split())

def first_hit(results, needle):
    for rank, r in enumerate(results, start=1):
        if norm(needle) in norm(r["text"]):
            return rank
    return None

def methods():
    units = load_passages()
    for q in QUERIES:
        assert any(norm(q["needle"]) in norm(u["text"]) for u in units), f"{q['id']}: needle not in any unit"
    kw = Keyword(units)
    return {
        "tfidf": lambda q: kw.search(q, 5, method="tfidf"),
        "bm25": lambda q: kw.search(q, 5),
    }

if __name__ == "__main__":
    ms = methods()
    totals = dict.fromkeys(ms, 0)
    print("query ", *(name.rjust(8) for name in ms))
    for q in QUERIES:
        cells = []
        for name, run in ms.items():
            rank = first_hit(run(q["query"]), q["needle"])
            totals[name] += rank is not None and rank <= 3
            cells.append(str(rank or "-").rjust(8))
        print(q["id"].ljust(6), *cells)
    print("hit@3 ", *(f"{totals[n]}/{len(QUERIES)}".rjust(8) for n in ms))
```

```bash
uv run python -m rag.scoreboard
```

*You should see* a table with the rank of the answer passage per query per method (`-` means not in the top 5), then a `hit@3` total for each. Expect your two `rare-name` queries to be found by both and your three `paraphrase` queries to be missed by both. If a paraphrase query was found, it shares a word with the passage after stemming; say so rather than rewriting it.

*If it broke:* `AssertionError: q04: needle not in any unit` means the needle is not an exact copy, or it crosses a blank line. Fix the needle, not the check.

Save the output as `rag/SCOREBOARD.md` under a heading `## A14`. A15, A16 and A17 each add a heading below it.

**Part 3. `rag/failures_wk7.md`.** Five queries where keyword search returned something wrong. Your scoreboard misses are the first place to look; add others from your corpus until you have five. For each: the query, the passage it returned at rank 1 (id and first 100 characters), and one sentence naming the mechanism: vocabulary mismatch, morphology the stemmer missed, negation, or a term so common in your corpus it is weighted to nothing. "It just did not work" is not a mechanism. A15 re-runs these through semantic search.

**Part 4. Where it got worse.** Under the scoreboard, one paragraph: the query where BM25 ranked the answer lower than plain tf-idf did, with both ranks. If there is none, run five more queries of your own until you find one or can say with numbers that there is none.

```bash
git add rag/scoreboard.py rag/SCOREBOARD.md rag/failures_wk7.md evidence/a14-bootdev-ch3.png
git commit -m "A14: tfidf vs bm25 scoreboard and five keyword failures"
git push
```

`evidence/a14-bootdev-ch3.png` is the chapter page with every lesson complete. Then close the log: `bash scripts/sign-off.sh`, and `git add logs && git commit && git push` in the log repo.

**Deliverable**
`rag/keyword.py` + `rag/queries.json` (committed before any score) + `rag/scoreboard.py` + `rag/SCOREBOARD.md` (A14 section) + `rag/failures_wk7.md` + `scratch/a14-bm25.md` + `evidence/a14-bootdev-ch3.png`.

**Reflection Questions**

1. Paste your A14 scoreboard. For each of your three `paraphrase` queries, paste the `terms:` line `rag.keyword` printed for the query and list three content words from the answer passage. Point to the exact reason no term matched. For any paraphrase query that *was* found, name the shared stem.
2. Paste the two ranked lists from Step 5, your two-word query, and the prediction you wrote during the chapter. Were you right about which word decided the ranking? Using the `bm25_idf` values of your two terms (print them), explain why the ranking came out the way it did.
3. Pick the failure in `rag/failures_wk7.md` you are least sure about the mechanism of. Paste the returned passage's first 200 characters, and the `preprocess` output of the query and of the answer passage. Say whether your stated mechanism survives looking at those lists, and if it does not, what the mechanism actually is.
