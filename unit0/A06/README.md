# A06 · Semantic Search from Scratch (NumPy Only)

**Meetings:** D13–D14 · **Points:** 15 pts

**Watch — 22 min**

**Day 1 — none.** Corpus day, shared with A05b. Run the checker, fix what it flags, then start Step 2 below and get the chunker working before you leave.

**Day 2 — 22 min**
[Vector Databases: from Embeddings to Applications](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/) · lesson 2, How to Obtain Vector Representations of Data (11m, code) · lesson 3, Search for Similar Vectors (6m, code)
[How Transformer LLMs Work](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/) · lesson 3, (Word) Embeddings (5m), a rewatch
 · Have the DLAI notebook open in one window and your repo open in VS Code in the other. Lessons 2 and 3 are the code-along; lesson 3 of the transformer course is the picture you already saw on A05, watched now with a corpus of your own on disk.

**During the video**

Make one file now, `scratch/a06-video.py`, and leave it open. Everything you type along with today goes in there.

**Type what he types, and run it.** Run the notebook in the DLAI page as he goes. Then type the parts named below into your scratch file and run them with `uv run python scratch/a06-video.py`, because the page is not where your corpus is.

**Lesson 2 · type the distance functions.** The first part makes vectors with libraries you are not installing; watch it. When he gets to measuring how far apart two vectors are, type every distance he writes, with NumPy, on two small vectors you make up. Then add one line of your own: normalize both vectors first and print the dot product next to his cosine. They are the same number, and that is the whole reason Step 5 below normalizes once instead of computing cosine every time.

**Lesson 3 · type the brute-force search and the timing.** This is today's walkthrough in his shape rather than yours: compare one query against every stored vector, sort, keep the top few. Type the search and run it. Then type the timing he does as the number of vectors grows, run it at the sizes he uses, and write the times in your log. Question 3 of the reflection asks you to put your own corpus on that curve.

**(Word) Embeddings · write, do not type.** No code. Pause when the lookup table appears and write its shape in your log with your own numbers: one row per chunk of your corpus, and a width you already know from A05.

**Notes**

**Your corpus is locked.** `data/corpus.txt` is the file A05b's checker passed. Do not swap it; every extension for the rest of the year runs on it.

**Banned:** Chroma, FAISS, Pinecone, pgvector, LangChain, LlamaIndex, `SentenceTransformer`, any `VectorStore`, any `.similarity_search()`. **Allowed:** `numpy`, the standard library, and a direct call to the embeddings endpoint. `search/search.py` lands under eighty lines with the query runner in it. If you are at 200, you imported something on that list.

**Short chunks lie.** A three-character chunk gets an embedding that still comes back with a high score, because short strings land in strange places in the space. Step 3 merges them away.

**Embed once.** `search/chunks.npy` is saved on the first run and loaded after that. It is git-ignored, like every `*.npy` except A05's two. Your key has a hard cap and it does not warn you.

**Changing the chunker after you embed silently breaks everything.** The saved array still has the old number of rows, or worse, the same number of rows holding different text. The `assert` in Step 5 catches a count change. It cannot catch a same-count change, so any time you touch `chunk()`, delete `search/chunks.npy` and re-embed.

**Walkthrough — search your own corpus**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
git switch main && git pull
git switch -c dev/semantic-search
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Day 1: corpus passes A05b's checker
- [ ] Day 1: Steps 2–3, chunker written and the length distribution printed twice
- [ ] Day 2: Vector Databases lesson 2, typed along in scratch/a06-video.py
- [ ] Day 2: Vector Databases lesson 3, typed along
- [ ] Day 2: How Transformer LLMs Work lesson 3, table shape written down
- [ ] Day 2: Steps 4–7, embed once, self-retrieval check, keyword baseline
- [ ] Extension: ten queries, marked by hand, cutoff from A05's floor
- [ ] Push and open the PR
```

Two meetings, one branch. Push again each day; one PR, not two; one log entry per meeting.

**Step 2. Make the file.**

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
mkdir -p search scratch
ls -la data/corpus.txt
```

*You should see* your corpus with the size A05b's checker reported. If `ls` says no such file, run `bash scripts/fetch_corpus.sh` and check again.

Create `search/search.py` with the top of the file. Type it.

```python
# search/search.py
import json, os, re, sys, urllib.request
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus.txt"
VECS = ROOT / "search" / "chunks.npy"
STOP = {"the", "a", "an", "of", "and", "to", "in", "is", "was", "what", "who", "how", "why", "does", "did"}
```

`ROOT` is the repo root worked out from where this file sits, so the script finds the corpus no matter which folder you run it from.

**Step 3. Chunk, and look at the distribution before you embed anything.**

Add the chunker:

```python
def chunk(text, min_chars=200, max_chars=2000):
    pieces = []
    for p in re.split(r"\n\s*\n", text):
        p = " ".join(p.split())
        pieces += [p[i:i + max_chars] for i in range(0, len(p), max_chars)]
    out = []
    for p in pieces:
        if out and len(out[-1]) < min_chars:
            out[-1] += " " + p
        else:
            out.append(p)
    if len(out) > 1 and len(out[-1]) < min_chars:
        out[-2] += " " + out.pop()
    return out
```

`" ".join(p.split())` collapses whitespace, so the text you embed is exactly the text you search with. Anything over `max_chars` is cut, because the endpoint refuses an input over its token limit. Any chunk under `min_chars` absorbs the piece after it, and a short last chunk folds into the one before.

Now compare it with the naive split. From the repo root:

```bash
uv run python -c "
import re, sys; sys.path.insert(0, '.')
from search.search import chunk, CORPUS
text = CORPUS.read_text(encoding='utf-8')
for name, cs in [('naive', [p for p in re.split(r'\n\s*\n', text) if p.strip()]), ('chunk', chunk(text))]:
    L = sorted(len(c) for c in cs)
    print(name, len(cs), 'min', L[0], 'median', L[len(L)//2], 'max', L[-1])
"
```

*You should see* two lines. The `naive` line has a minimum in the single or low double digits and a maximum in the thousands. The `chunk` line has a minimum of at least 200, a maximum no higher than about 2,200 (a short chunk can absorb one full piece), and a count of at least 300. Paste both lines into your log.

*If it broke:* a `chunk` count under 300 means your corpus is too small for these settings; lower `min_chars` to 150 and write down that you did. A `chunk` minimum under 200 means the corpus is one chunk long, which A05b's checker should have refused.

**Step 4. Embed in batches, sorted by `index`.**

```python
def embed(texts, batch=100):
    rows = []
    for i in range(0, len(texts), batch):
        body = json.dumps({"model": "text-embedding-3-small", "input": texts[i:i + batch]}).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings", data=body,
            headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            items = json.load(r)["data"]
        items.sort(key=lambda it: it["index"])
        rows += [it["embedding"] for it in items]
    return np.array(rows, dtype=np.float32)
```

A05's function with a loop around it. The `items.sort` line is what keeps `chunks[i]` and `V[i]` the same passage; arrival order is not input order.

**Step 5. Normalize, search, and load from disk.**

```python
def normalize(V):
    return V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-10)

def search(query, Vn, k=3):
    q = normalize(embed([query])[0])
    sims = Vn @ q
    top = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in top]
```

`keepdims=True` is A05's `[:, None]` written so the same line works on one vector or a whole matrix. `Vn @ q` is one row of A05's `S`: the query against every chunk.

Then the bottom of the file:

```python
if __name__ == "__main__":
    chunks = chunk(CORPUS.read_text(encoding="utf-8"))
    if VECS.exists():
        V = np.load(VECS)
    else:
        V = embed(chunks)
        np.save(VECS, V)
    assert len(V) == len(chunks), "chunking changed since you embedded: delete search/chunks.npy"
    Vn = normalize(V)
    print(len(chunks), "chunks", V.shape, Vn.shape, np.allclose(np.linalg.norm(Vn, axis=1), 1.0))
```

Run it twice:

```bash
uv run python search/search.py
uv run python search/search.py
```

*You should see* the same line both times: your chunk count, `(n, 1536)` twice, and `True`. The first run takes seconds and spends money; the second is instant, or the load branch is not being taken.

**Step 6. Self-retrieval check.**

Add these lines inside the `__main__` block, at the bottom:

```python
    if sys.argv[1:] == ["--check"]:
        for i in (0, len(chunks) // 2, len(chunks) - 1):
            print(i, search(chunks[i], Vn, k=2))
```

```bash
uv run python search/search.py --check
```

The first, middle and last chunk, because an off-by-one only shows at an end. *You should see* three lines, each starting with the same index you searched with at 0.999 or higher, then some other chunk clearly lower.

*If it broke:* another index at rank 1 means `chunks[i]` and `V[i]` are not the same passage, and every result after this is noise; the cause is almost always a missing `items.sort`. The right index at 0.97 rather than 0.999 means the text you searched with is not byte-for-byte the text you embedded.

**Step 7. The keyword baseline.**

```python
def words(s):
    return set(re.findall(r"[a-z0-9']+", s.lower())) - STOP

def keyword_search(query, chunks, k=3):
    qw = words(query)
    scores = np.array([len(qw & words(c)) for c in chunks])
    top = np.argsort(scores)[::-1][:k]
    return [(int(i), int(scores[i])) for i in top]
```

The score is how many distinct query words appear in the chunk, ignoring a few that appear everywhere. It knows nothing about meaning, which is what makes it the baseline.

Then the runner, at the bottom of `__main__`, after the `--check` block:

```python
    else:
        for line in (ROOT / "search" / "queries.txt").read_text(encoding="utf-8").splitlines():
            q = line.split(" | ")[0].strip()
            if not q or q.startswith("#"):
                continue
            print("\n##", q)
            for (i, s), (j, n) in zip(search(q, Vn), keyword_search(q, chunks)):
                print(f"  sem {s:.3f} [{i}] {chunks[i][:90]!r}\n  kw  {n:>5} [{j}] {chunks[j][:90]!r}")
```

The `else` pairs with the `if` from Step 6, so indent it to match. Do not run the runner yet. The queries file does not exist, and writing it is the extension.

```bash
wc -l search/search.py
```

*You should see* a number under eighty.

**Extension — ten queries and a cutoff (ASSIGNED)**

**E1. Write the queries and the prediction first.** Create `search/queries.txt` with ten real questions about your corpus, one per line, each followed by ` | ` and a short phrase copied from the chunk you expect to answer it. At least three must share no content words with that chunk: ask about the thing in words the text never uses. Then two more lines asking something your corpus cannot answer, each followed by ` | none`. Keep them on your corpus's topic so they are hard to tell apart from a real question:

```text
Who tricks the Cyclops by giving a false name? | my name is Noman
What year was the Odyssey first printed in English? | none
```

Those are Odyssey examples. Yours come from your file.

In the same commit, open `embed/FINDINGS.md` from A05 and copy your anisotropy floor into a new file `search/RESULTS.md`, with one line under it: the similarity below which you predict "no good answer", and why that number follows from the floor. Commit before you run anything:

```bash
git add search/queries.txt search/RESULTS.md
git commit -m "A06: ten queries, expected chunks, predicted cutoff (before any run)"
```

I will check your commit timestamps. Queries written after seeing results measure nothing.

**E2. Measure the floor on chunks.** A05's floor was measured on 200 single words. Chunks are paragraphs, and their floor is not the same number. Measure it the same way A05 did:

```bash
uv run python -c "
import sys, numpy as np; sys.path.insert(0, '.')
from search.search import normalize
Vn = normalize(np.load('search/chunks.npy'))
i, j = np.random.default_rng(0).integers(0, len(Vn), (2, 1000)); keep = i != j
s = (Vn[i[keep]] * Vn[j[keep]]).sum(axis=1)
print('chunk floor mean', round(float(s.mean()), 3), ' 95th pct', round(float(np.percentile(s, 95)), 3))
"
```

*You should see* a clearly positive mean and a 95th percentile above it. Whether the chunk floor is above or below your A05 word floor depends on your corpus: one book on one subject pulls every chunk toward the same place. Write both numbers in `RESULTS.md`.

**E3. Run and mark.**

```bash
uv run python search/search.py > search/run1.txt
```

That is one run over all twelve lines. For each of the ten answerable queries, read the three semantic results and the three keyword results. A method gets the query if any of its top three actually answers the question. You decide by reading, not by checking whether your expected phrase appears. Build this table in `RESULTS.md`:

| # | query | semantic top score | semantic got it | keyword top overlap | keyword got it |
|---|---|---|---|---|---|

Then the totals: semantic got X of 10, keyword got Y of 10.

*You should see* the two methods agree on most queries and disagree on a few. If semantic gets ten of ten, your three no-shared-words queries were not honest; rewrite them and say so in the log. A keyword win is a real result: a rare name or number is something a 1,536-number summary of a paragraph holds on to badly.

**E4. Test the cutoff.** Add the two off-corpus queries to the bottom of the table with their top semantic scores. Now lay every top score out on one line: the ten answerable ones marked got-it or missed, the two off-corpus ones marked. Your predicted cutoff either separates "has an answer" from "does not" or it does not. Report how many of the twelve land on the wrong side of it, then the cutoff that would have made the fewest mistakes, and how far it is from the floor.

Write the paragraph that ends `RESULTS.md`: where semantic search did worse than keyword, and where your cutoff got it wrong.

**E5. Commit, push, PR, sign off.**

```bash
git add search/search.py search/RESULTS.md search/run1.txt scratch/a06-video.py
git commit -m "A06: semantic vs keyword on 10 queries, cutoff tested on 2 off-corpus"
git push -u origin dev/semantic-search
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the query you are least sure you marked fairly.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`search/search.py` (under 80 lines, NumPy and stdlib only) + `search/queries.txt` + `search/RESULTS.md` (floors, marked table, totals, cutoff test) + `search/run1.txt` + `scratch/a06-video.py`.

**Reflection Questions**

1. Paste the two lines from Step 3, `naive` and `chunk`. Find the shortest chunk the naive split produced by printing it, paste it, and say what it is in your file (a heading, a line number, a stage direction, a page break). Then find the chunk it ended up inside after `chunk()` (`[i for i, c in enumerate(chunks) if frag in c]`), give its index, paste its first 90 characters, and say whether that chunk appeared in any of your thirty semantic results.

2. Your predicted cutoff, committed in E1, and the cutoff that made the fewest mistakes in E4: give both, with the commit hash of the prediction. Which off-corpus query scored highest, what was its top chunk, and why does a question your corpus cannot answer still land above the chunk floor you measured in E2?

3. Pick the query where semantic and keyword disagreed most. Paste both top results. Then use your timing from the lesson 3 code-along: how many chunks do you have, how long does one `Vn @ q` over them take on your machine (time it), and how many times larger would your corpus need to be before that search, not the API call, is the slow part of a query?
