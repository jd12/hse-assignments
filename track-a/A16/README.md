# A16 · Chunking and Hybrid Search

**Meetings:** D31 · **Points:** 15 pts

**Watch**
Mon Nov 9 — [Boot.dev, *Learn Retrieval Augmented Generation*](https://www.boot.dev/courses/learn-retrieval-augmented-generation) ch. 5 Chunking and ch. 6 Hybrid Search, about 25 minutes of lessons each

That is about 50 minutes at the keyboard, over the cap. Both chapters are here because hybrid search is not worth measuring on passages you are about to throw away. If you do not finish ch. 6 in class, finish it before you start the extension.

**During the video**

Same folder, `~/version_control/bootdev-rag/`. Type every solution, `bootdev run <lesson-id>` before `-s`.

**Ch. 5 · break the chunker on purpose.** When you have the chapter's fixed-size chunker passing, call it once with the overlap equal to the chunk size, and write what happened in `scratch/a16-chunks.md`: an error, a hang (press Ctrl-C), or a result. Then call it on a string whose length is not a multiple of the chunk size and write down what the last chunk looks like.

**Ch. 6 · write down the fusion.** The chapter combines a keyword ranking and a semantic ranking. Write in `scratch/a16-chunks.md` which method it uses (a weighted sum of normalized scores, rank fusion, or both) and every constant it uses, with its value. Step 5 uses the rank-fusion constant.

**Notes**

**Overlap equal to or larger than the chunk size is an infinite loop or a crash.** The step is `size - overlap`; at zero, `range` raises, and in a hand-written `while` loop it never advances. The walkthrough asserts `overlap < size` on the first line of the function, so the mistake is a message rather than a hang.

**The last chunk is short and that is correct.** A window that runs past the end of the text gets clamped by the slice, silently. The failure is not a short last chunk; it is a *missing* last sentence, when the loop stops one step early. The walkthrough checks that the last chunk ends at the last word of the file.

**Scores on different scales cannot be added.** Cosine similarity lives between -1 and 1. BM25 is unbounded and grows with the corpus. Add them and BM25 decides everything. Either normalize both into the same range, or throw the scores away and combine the ranks. The walkthrough does ranks (reciprocal rank fusion) and shows you the scale problem first.

**Your chunk ids are temporary until the end of today.** Everything from A19 on labels answers by chunk id, and a chunk id is only meaningful for one `SIZE` and `OVERLAP`. The extension picks your final values and freezes them.

**Walkthrough — chunks with offsets, then fusion**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch main && git pull
git switch dev/semantic && git pull     # skip this line if A15 has merged
git switch -c dev/hybrid
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Boot.dev ch. 5 Chunking, every lesson green
- [ ] Boot.dev ch. 6 Hybrid Search, every lesson green
- [ ] `scratch/a16-chunks.md`: the overlap break, the last chunk, the fusion constants
- [ ] `rag/chunking.py` with its three invariants (Steps 2–3)
- [ ] `rag/hybrid.py` and `rag/search.py`, the scale check (Steps 4–6)
- [ ] Scoreboard on chunks (Step 7)
- [ ] Extension: three chunk sizes, chosen and frozen
- [ ] Push, PR, sign off
```

**Step 2. `rag/chunking.py`.**

Word windows rather than character windows, so no chunk starts or ends mid-word. Every chunk is a record, not a string: it carries its id and the exact character positions it came from in `data/corpus.txt`. A18's citations depend on those positions.

```python
# rag/chunking.py
import re
import sys

from rag.corpus import load_text

SIZE, OVERLAP = 200, 50

def chunk(text, size=SIZE, overlap=OVERLAP):
    assert 0 <= overlap < size, f"overlap {overlap} must be smaller than size {size}"
    spans = [m.span() for m in re.finditer(r"\S+", text)]
    step = size - overlap
    chunks = []
    for n, i in enumerate(range(0, len(spans), step)):
        window = spans[i:i + size]
        start, end = window[0][0], window[-1][1]
        chunks.append({"id": f"c{n:05d}", "char_start": start, "char_end": end,
                       "text": text[start:end]})
        if i + size >= len(spans):
            break
    return chunks

def load_chunks(size=SIZE, overlap=OVERLAP):
    return chunk(load_text(), size, overlap)

if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else SIZE
    overlap = int(sys.argv[2]) if len(sys.argv) > 2 else OVERLAP
    text = load_text()
    cs = chunk(text, size, overlap)
    words = len(re.findall(r"\S+", text))
    print(f"{words} words -> {len(cs)} chunks of {size} words, overlap {overlap}")
    print("every chunk is an exact slice:", all(text[c["char_start"]:c["char_end"]] == c["text"] for c in cs))
    print("last chunk ends at the last word:", cs[-1]["char_end"] == len(text.rstrip()))
    print("last chunk words:", len(cs[-1]["text"].split()))
```

The `break` is the part to understand. Without it, the loop keeps producing windows that start inside the last full window, each one a shorter copy of the end of the file.

**Step 3. Three invariants.**

```bash
uv run python -m rag.chunking
uv run python -m rag.chunking 300 0
uv run python -m rag.chunking 100 100
```

*You should see*, for the first two runs, both checks `True`, and a chunk count equal to `(words - overlap) / (size - overlap)` rounded up: work that out on paper from your word count before you look. The last chunk's word count is anywhere from 1 to `size`. The third run stops with `AssertionError: overlap 100 must be smaller than size 100`.

*If it broke:* `exact slice: False` means you built `text` from the words and lost the original spacing; slice the original instead. `last chunk ends at the last word: False` means the loop stopped a step early.

**Step 4. `rag/hybrid.py`: reciprocal rank fusion.**

```python
# rag/hybrid.py
def rrf(rankings, k=60, top=5):
    scores = {}
    for ranking in rankings:
        for rank, uid in enumerate(ranking, start=1):
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda uid: (-scores[uid], uid))[:top]
```

Each list votes `1 / (k + rank)` for each chunk it ranked. Use the constant the chapter gave you for `k`; 60 is the common default. A chunk at rank 1 in both lists scores `2 / 61`, about 0.0328, the largest score this function can produce with two lists. The size of `k` decides how much rank 1 is worth over rank 10: small `k`, the top of each list dominates; large `k`, the lists behave like a vote among equals.

**Step 5. `rag/search.py`: one place that builds the engines.**

```python
# rag/search.py
from functools import lru_cache

from rag.chunking import OVERLAP, SIZE, load_chunks
from rag.hybrid import rrf
from rag.keyword import Keyword
from rag.semantic import Semantic

@lru_cache(maxsize=None)
def engines(size=SIZE, overlap=OVERLAP):
    units = load_chunks(size, overlap)
    return units, Keyword(units), Semantic(units, f"chunks_{size}_{overlap}")

def keyword(query, k=5, size=SIZE, overlap=OVERLAP):
    return engines(size, overlap)[1].search(query, k)

def semantic(query, k=5, size=SIZE, overlap=OVERLAP):
    return engines(size, overlap)[2].search(query, k)

def hybrid(query, k=5, pool=50, size=SIZE, overlap=OVERLAP):
    units, kw, sem = engines(size, overlap)
    by_id = {u["id"]: u for u in units}
    rankings = [[r["id"] for r in kw.search(query, pool)],
                [r["id"] for r in sem.search(query, pool)]]
    return [by_id[i] for i in rrf(rankings, top=k)]
```

Each chunk configuration gets its own vector file, `data/emb_chunks_200_50.npy` and so on, and the A15 hash guard re-embeds if the chunks change. The first call to `engines` embeds your chunks: one paid call per configuration, then never again.

**Step 6. See the scale problem.** Pick any query from `rag/queries.json`:

```bash
uv run python -c "
from rag import search
q = '<one of your queries>'
print('bm25    ', [round(r['score'], 2) for r in search.keyword(q)])
print('semantic', [round(r['score'], 3) for r in search.semantic(q)])
print('hybrid  ', [r['id'] for r in search.hybrid(q)])
"
```

*You should see* BM25 scores in the single or double digits and cosine scores below 1. Add them and BM25 decides the order; the cosine score only reorders chunks whose BM25 scores are within a few hundredths of each other. The hybrid ids are drawn from both lists.

**Step 7. Move the scoreboard to chunks.**

Replace `methods()` and the top of `__main__` in `rag/scoreboard.py`. Passages are finished; from here every method runs on chunks.

```python
import sys

from rag import search
from rag.chunking import OVERLAP, SIZE, load_chunks

def methods(size=SIZE, overlap=OVERLAP):
    units = load_chunks(size, overlap)
    for q in QUERIES:
        assert any(norm(q["needle"]) in norm(u["text"]) for u in units), f"{q['id']}: needle not in any unit"
    return {
        "bm25": lambda q: search.keyword(q, 5, size, overlap),
        "semantic": lambda q: search.semantic(q, 5, size, overlap),
        "hybrid": lambda q: search.hybrid(q, 5, size=size, overlap=overlap),
    }

if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else SIZE
    overlap = int(sys.argv[2]) if len(sys.argv) > 2 else OVERLAP
    ms = methods(size, overlap)
```

Change `from rag.corpus import ROOT, load_passages` to `from rag.corpus import ROOT` (`ROOT` is still used for `QUERIES`), delete the `from rag.keyword import Keyword` and `from rag.semantic import Semantic` lines, and keep the rest of `__main__` as it was.

```bash
uv run python -m rag.scoreboard
```

*You should see* three columns and every needle found in some chunk. A needle of ten words always fits inside some chunk as long as the overlap is at least ten words, which is why the queries survived the switch.

```bash
git add rag/chunking.py rag/hybrid.py rag/search.py rag/scoreboard.py scratch/a16-chunks.md
git commit -m "A16: word-window chunks with offsets, RRF hybrid search"
git push -u origin dev/hybrid
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**.

**Extension — three chunk sizes, one choice (ASSIGNED)**

Chunk size is the first setting in this course with no right answer. You are going to measure three and pick one, and the pick is final: A19's golden set is labeled on it.

**Part 1. Run three configurations.** Overlap is a quarter of the size in each:

```bash
uv run python -m rag.scoreboard 100 25
uv run python -m rag.scoreboard 200 50
uv run python -m rag.scoreboard 400 100
```

Each one embeds its chunks once. Paste all three tables into `rag/SCOREBOARD.md` under `## A16`.

**Part 2. `rag/chunking_notes.md`.** One table:

| size / overlap | chunks | bm25 hit@3 | semantic hit@3 | hybrid hit@3 | a query that got better | a query that got worse |
|---|---|---|---|---|---|---|

"Better" and "worse" are compared with the row above it; for the first row, compare with the A15 passage scoreboard. Each cell in the last two columns is a query id and its rank change, like `q07: 5 → 1`. Somewhere in the three rows there is a query that got worse. If you cannot find one in the top 5, print the top 10 for your paraphrase queries and look there.

**Part 3. Where hybrid lost.** Find the query where hybrid ranked the answer lower than the better of its two inputs did. Paste the three ranks. RRF rewards chunks both lists liked; say what that did to a chunk only one list found.

**Part 4. Choose and freeze.** Set `SIZE` and `OVERLAP` in `rag/chunking.py` to your choice, and write under the table in `chunking_notes.md`: the values, the reason in two sentences, and the query class you knowingly gave up to get there. Then:

```bash
uv run python -m rag.scoreboard
git add rag/chunking.py rag/SCOREBOARD.md rag/chunking_notes.md evidence/a16-bootdev-ch5.png evidence/a16-bootdev-ch6.png
git commit -m "A16: chunk size chosen and frozen at <size>/<overlap>"
git push
```

The last scoreboard run is the one that proves the default matches the row you chose. Then close the log: `bash scripts/sign-off.sh`, and `git add logs && git commit && git push` in the log repo.

**Deliverable**
`rag/chunking.py` (frozen `SIZE`/`OVERLAP`) + `rag/hybrid.py` + `rag/search.py` + `rag/scoreboard.py` on chunks + `rag/chunking_notes.md` + `rag/SCOREBOARD.md` (A16 section) + `scratch/a16-chunks.md` + two Boot.dev screenshots in `evidence/`.

**Reflection Questions**

1. Paste the three lines `uv run python -m rag.chunking` printed for your final configuration, and the chunk count you worked out on paper in Step 3 before you ran it. Did they agree? Then paste the last chunk's first 150 characters and say what part of your corpus it is and how many words it holds.
2. Paste your Part 3 ranks: the query, its BM25 rank, its semantic rank and its hybrid rank. Compute by hand the RRF score of the answer chunk and of the chunk hybrid put at rank 1, using your `k`, and show the arithmetic. Would a smaller `k` have saved the answer? Try it and paste the new hybrid rank.
3. Paste the row of `chunking_notes.md` for the size you chose, and the "got worse" query from it. Print that query's answer chunk at your chosen size and paste it. Is the answer cut by a chunk boundary, buried in a chunk about something else, or fine and outranked? Say what that tells you the size you chose costs.
