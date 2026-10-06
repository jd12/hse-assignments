# A06 · Semantic Search from Scratch (NumPy Only)

**Meetings:** D13–D14 · **Points:** 15 pts

**Watch — 22 min**

**Day 1 — none.** Corpus day, shared with A05b. Run the checker, fix what it flags, then start Step 2 below and get the chunker running before you leave.

**Day 2 — 22 min**
[Vector Databases: from Embeddings to Applications](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/) · lesson 2, How to Obtain Vector Representations of Data (11m, code) · lesson 3, Search for Similar Vectors (6m, code)
[How Transformer LLMs Work](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/) · lesson 3, (Word) Embeddings (5m), a rewatch
 · Have the DLAI notebook open in one window and your repo open in VS Code in the other. Lessons 2 and 3 are the code-along; lesson 3 of the transformer course is the picture you already saw on A05, watched now with a corpus of your own on disk.

**During the video**

He runs his code on vectors he made up. You run the same operations on your corpus, from one file you make before the video starts. Create `scratch/a06-video.py` (right-click `scratch`, **New File**) and paste all of this in. It needs the `search/search.py` you made on Day 1, Steps 2 and 3, and nothing from the API.

```python
# scratch/a06-video.py
# Run from the repo root:  uv run python scratch/a06-video.py
# Needs search/search.py from Day 1 (Steps 2 and 3), for chunk() and CORPUS.
# Part 1 (lesson 2): the four distances from the video, on three chunks of your corpus.
# Part 2 (lesson 3): brute-force search in his shape, then the timing curve as N grows.
# Part 3 (word embeddings lesson): the shape your lookup table will have.
import re, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so "from search.search" works
from search.search import chunk, CORPUS

def normalize(V):                 # the same line Step 5 puts in search.py
    return V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-10)

chunks = chunk(CORPUS.read_text(encoding="utf-8"))
n = len(chunks)

# ------------------------------------------------------------ Part 1: distances
# No API yet. The crudest vector a paragraph can have is a count of each word in it.
# These are real vectors about your corpus, and every distance below works on them
# exactly as it will on the 1,536-number vectors in Step 4.
def tokens(s):
    return re.findall(r"[a-z0-9']+", s.lower())     # splits on anything that is not a letter, digit or apostrophe

A, B, C = chunks[n // 2], chunks[n // 2 + 1], chunks[n // 4]      # two neighbors, and one far away
vocab = sorted(set(tokens(A)) | set(tokens(B)) | set(tokens(C)))
col = {w: i for i, w in enumerate(vocab)}
def count_vector(s):
    v = np.zeros(len(vocab))
    for w in tokens(s):
        v[col[w]] += 1
    return v
a, b, c = count_vector(A), count_vector(B), count_vector(C)
print("three chunks of your corpus:", n // 2, n // 2 + 1, n // 4, " vocabulary", len(vocab), " vector shape", a.shape)

def euclidean(x, y): return float(np.sqrt(((x - y) ** 2).sum()))
def manhattan(x, y): return float(np.abs(x - y).sum())
def dot(x, y):       return float(x @ y)
def cosine(x, y):    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
def normalized_dot(x, y):
    return float(normalize(x) @ normalize(y))      # the line that is not in the video

print(f"{'pair':<10}{'euclidean':>12}{'manhattan':>12}{'dot':>10}{'cosine':>10}{'norm-dot':>10}")
for name, x, y in [("A vs B", a, b), ("A vs C", a, c), ("B vs C", b, c), ("A vs A", a, a)]:
    print(f"{name:<10}{euclidean(x, y):>12.3f}{manhattan(x, y):>12.1f}{dot(x, y):>10.1f}{cosine(x, y):>10.4f}{normalized_dot(x, y):>10.4f}")
print("chunk lengths in words: A", len(tokens(A)), " B", len(tokens(B)), " C", len(tokens(C)))

# ------------------------------------------------------------ Part 2: brute force, and the timing curve
rng = np.random.default_rng(0)

def brute_force(q, X, k=3):
    sims = X @ q                        # one score per stored vector
    top = np.argsort(sims)[::-1][:k]    # argsort is ascending; flip it; keep k
    return [(int(i), round(float(sims[i]), 4)) for i in top]

X = normalize(rng.standard_normal((1000, 1536), dtype=np.float32))   # 1,000 pretend chunks, same width as the real ones
q = X[7]                                                             # search with one of them
print("\nbrute force, query = row 7 of 1,000 random vectors:", brute_force(q, X))

print("\nN vectors     microseconds per search")
for N in (1_000, 2_000, 5_000, 10_000, 20_000, n):
    X = normalize(rng.standard_normal((N, 1536), dtype=np.float32))
    q = X[0]
    t = time.perf_counter()
    for _ in range(50):
        X @ q
    us = (time.perf_counter() - t) / 50 * 1e6
    tag = "   <- your chunk count" if N == n else ""
    print(f"{N:>9}     {us:>10.0f}{tag}")

# ------------------------------------------------------------ Part 3: the lookup table
print("\nyour lookup table in Step 4 will be one row per chunk:", (n, 1536))
```

Run it once before you press play, so you know it works:

```bash
uv run python scratch/a06-video.py
```

*You should see* a distance table with four pairs in it, one brute-force result, a six-row timing table and one shape line. The whole thing takes a few seconds. Paste the full output into your log; the three lessons below each point at one part of it.

*If it broke* with `ModuleNotFoundError: No module named 'search'`, Day 1 is not done: `search/search.py` has to exist with `chunk` in it. `FileNotFoundError` on `corpus.txt` means the same thing A05b's checker would have told you.

**Lesson 2 · the four distances, on three paragraphs of your corpus.** The first part of the lesson makes vectors with libraries you are not installing; watch it. When he gets to measuring how far apart two vectors are, he writes four functions: Euclidean, Manhattan, dot product, cosine. Part 1 of your output has the same four, on three chunks of your file: A and B are next to each other in the text, C is far away. Read the table as he writes each one. **Euclidean and Manhattan grow with the length of the paragraphs**; the dot product does too, and the `A vs A` row shows it: a chunk against itself is not 1, it is the sum of its squared counts. Cosine is the only column that ignores length, and the `norm-dot` column is the line he does not write: normalize both vectors first, then take the dot product. It matches the cosine column to the last digit, and that is the whole reason Step 5 below normalizes every chunk once instead of computing cosine every time. On Homer the `A vs B` row read `12.806  132.0  104.0  0.5631  0.5631` and `A vs A` read `0.000  0.0  208.0  1.0000  1.0000`.

Count vectors are a bad embedding, and your table shows why: on Homer the two neighboring paragraphs scored 0.563 and a paragraph from the other poem scored 0.531, barely apart, because `the`, `and` and `he` dominate every count. Step 4 pays for a vector that knows what the paragraph is about.

**Lesson 3 · brute force, and what it costs.** He compares one query against every stored vector, sorts, and keeps the top few. That is `brute_force` in Part 2, and it is today's walkthrough in his shape: `X @ q` is one score per stored vector, `argsort` sorts them, `[::-1][:k]` keeps the top k. The result line shows row 7 finding itself at 1.0 and two strangers near 0.07, which is what random vectors in 1,536 dimensions look like. Then he grows the number of vectors and times the search. Your timing table does the same at 1,000 to 20,000 vectors, plus one row marked `<- your chunk count`. **Which of these is `search.py`?** `brute_force` is `search()` in Step 5 with the API call taken out: the only line Step 5 adds is the one that turns the query text into a vector. The marked row is the time your own search will take once the vectors are on disk, and Reflection Question 3 asks you to compare it with the API call.

*You should see* the microseconds roughly double when N doubles: brute force is a straight line in N. On Homer, 1,000 vectors took about 130 microseconds, 20,000 about 2,700, and the 2,142-chunk row about 300.

**(Word) Embeddings · one line in your log.** No code. When the lookup table appears in the lesson, one row per token, write down the shape Part 3 printed: one row per chunk of your corpus, 1,536 wide, the width you read off A05's `(200, 1536)`. On Homer it printed `(2142, 1536)`. That is the table Step 4 builds and Step 5 searches.

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
- [ ] Day 1: Steps 2–3, search.py started, both chunk snippets run and pasted in the log
- [ ] Day 2: scratch/a06-video.py run, output in the log
- [ ] Day 2: Vector Databases lessons 2 and 3, read against Parts 1 and 2
- [ ] Day 2: How Transformer LLMs Work lesson 3, lookup-table shape written down
- [ ] Day 2: Steps 4–7, embed once, self-retrieval check, keyword baseline
- [ ] Extension: ten queries checked and committed before any run, run, marked, cutoff tested
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

Create `search/search.py` (right-click `search`, **New File**) and paste the top of the file:

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

`ROOT` is the repo root worked out from where this file sits, so the script finds the corpus no matter which folder you run it from. `VECS` is where the embedded vectors will be saved. `STOP` is the short list of words the keyword baseline in Step 7 ignores.

**Step 3. Chunk, and look at the distribution before you embed anything.**

Paste this below the `STOP` line. `chunk()` turns the whole corpus into a list of passages, each between 200 and 2,000 characters, that Step 4 will embed one row each.

```python
def chunk(text, min_chars=200, max_chars=2000):
    pieces = []
    for p in re.split(r"\n\s*\n", text):          # split on a blank line (a newline, optional spaces, another newline)
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

Now compare it with the naive split, which is the blank-line split with nothing merged. From the repo root:

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

*You should see* two lines. The `naive` line has a minimum in the single or low double digits and a maximum in the thousands. The `chunk` line has a minimum of exactly 200 or a little above, a maximum no higher than about 2,200 (a short chunk can absorb one full piece), and a count of at least 300. On Homer:

```text
naive 2487 min 6 median 529 max 3745
chunk 2142 min 200 median 596 max 2095
```

The checker's `chunks` line said 2079 on the same file, because the checker drops short pieces and `chunk()` merges them; the count you carry from here on is the `chunk` one. Paste both lines into your log.

*If it broke:* a `chunk` count under 300 means your corpus is too small for these settings; change `min_chars=200` to `min_chars=150` in the `def chunk` line and write down that you did. A `chunk` minimum under 200 means the corpus is one chunk long, which A05b's checker should have refused.

Then look at the piece behind that `naive` minimum, and where it went. Same place, same shape:

```bash
uv run python -c "
import re, sys; sys.path.insert(0, '.')
from search.search import chunk, CORPUS
text = CORPUS.read_text(encoding='utf-8')
naive = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
frag = ' '.join(min(naive, key=len).split())
print('shortest naive piece:', repr(frag), '(', len(frag), 'chars )')
chunks = chunk(text)
hits = [i for i, c in enumerate(chunks) if (' ' + frag + ' ') in (' ' + c + ' ')]
print('lands in chunk(s)', hits, 'of', len(chunks))
print(repr(chunks[hits[0]][:90]) if hits else '(not found)')
"
```

The spaces padded around `frag` make the search match the piece as a whole word, so `BOOK I` does not match inside `BOOK II`.

*You should see* three lines: the piece itself in quotes, the index of the chunk it was absorbed into, and that chunk's first 90 characters. On Homer:

```text
shortest naive piece: 'BOOK I' ( 6 chars )
lands in chunk(s) [23] of 2142
'HENRY FESTING JONES. 120 MAIDA VALE, W.9. 4th _December_, 1921. THE ODYSSEY BOOK I THE GOD'
```

The piece is almost always a heading, a page break, a stage direction or a stray line number, which is why the naive split has a minimum in the single digits and yours does not. On Homer the heading was folded into the end of the translator's preface, which is a chunk about nothing in particular. Keep all three lines; Reflection Question 1 asks for them.

**Step 4. Embed in batches, sorted by `index`.**

Paste this below `chunk()`. `embed()` sends a list of texts to the model a hundred at a time and returns one row of 1,536 numbers per text, in the order you sent them.

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

A05's function with a loop around it. The `items.sort` line is what keeps `chunks[i]` and `V[i]` the same passage; arrival order is not input order. Nothing to run yet.

**Step 5. Normalize, search, and load from disk.**

Paste this below `embed()`. `normalize()` makes every row length 1, so a dot product is a cosine. `search()` embeds one query and scores it against every chunk in one line.

```python
def normalize(V):
    return V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-10)

def search(query, Vn, k=3):
    q = normalize(embed([query])[0])
    sims = Vn @ q
    top = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in top]
```

`keepdims=True` is A05's `[:, None]` written so the same line works on one vector or a whole matrix. `Vn @ q` is one row of A05's `S`: the query against every chunk. Compare `search()` with `brute_force` in your video script: the first line is the only difference.

Then paste the main block at the very bottom of the file. It chunks the corpus, embeds once and saves, loads on every run after that, and prints the shapes.

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

*You should see* the same line both times: your chunk count, `(n, 1536)` twice with your count in place of `n`, and `True`. The first run takes seconds and spends money; the second is instant, or the load branch is not being taken. On Homer the line is `2142 chunks (2142, 1536) (2142, 1536) True`.

*If it broke* with `KeyError: 'OPENAI_API_KEY'`, open a new terminal, as in A05. An `HTTP Error 400` on the first run means one chunk is over the model's input limit, which `max_chars=2000` should prevent; check that you did not change it.

**Step 6. Self-retrieval check.**

Replace the whole `if __name__ == "__main__":` block, from that line to the end of the file, with this one. It is the same block with three lines added at the bottom that search for the first, middle and last chunk by their own text.

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
    if sys.argv[1:] == ["--check"]:
        for i in (0, len(chunks) // 2, len(chunks) - 1):
            print(i, search(chunks[i], Vn, k=2))
```

```bash
uv run python search/search.py --check
```

The first, middle and last chunk, because an off-by-one only shows at an end. *You should see* the shape line, then three lines, each starting with the index you searched with, then that same index again at 0.999 or higher inside the brackets, then some other chunk clearly lower. On Homer the three indexes are `0`, `1071` and `2141`.

<!-- JD: fill the three Homer --check lines after a run; the shape is "i [(i, 0.999+), (j, lower)]". -->

*If it broke:* another index at rank 1 means `chunks[i]` and `V[i]` are not the same passage, and every result after this is noise; the cause is almost always a missing `items.sort`. The right index at 0.97 rather than 0.999 means the text you searched with is not byte-for-byte the text you embedded.

**Step 7. The keyword baseline.**

Two pastes. First, paste these two functions **above** the `if __name__ == "__main__":` line, directly below `search()`. `words()` turns a text into the set of distinct words in it, minus `STOP`. `keyword_search()` scores every chunk by how many of the query's words it contains, and keeps the top three.

```python
def words(s):
    return set(re.findall(r"[a-z0-9']+", s.lower())) - STOP     # same splitting rule as tokens() in the video script

def keyword_search(query, chunks, k=3):
    qw = words(query)
    scores = np.array([len(qw & words(c)) for c in chunks])
    top = np.argsort(scores)[::-1][:k]
    return [(int(i), int(scores[i])) for i in top]
```

The score is how many distinct query words appear in the chunk, ignoring a few that appear everywhere. It knows nothing about meaning, which is what makes it the baseline.

Second, replace the whole `if __name__ == "__main__":` block one more time with the final version. The new part is the `else`: without `--check`, the script reads `search/queries.txt` and, for every question in it, prints the top three semantic results and the top three keyword results side by side.

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
    if sys.argv[1:] == ["--check"]:
        for i in (0, len(chunks) // 2, len(chunks) - 1):
            print(i, search(chunks[i], Vn, k=2))
    else:
        for line in (ROOT / "search" / "queries.txt").read_text(encoding="utf-8").splitlines():
            q = line.split(" | ")[0].strip()
            if not q or q.startswith("#"):
                continue
            print("\n##", q)
            for (i, s), (j, n) in zip(search(q, Vn), keyword_search(q, chunks)):
                print(f"  sem {s:.3f} [{i}] {chunks[i][:90]!r}\n  kw  {n:>5} [{j}] {chunks[j][:90]!r}")
```

Do not run the runner yet. The queries file does not exist, and writing it is the extension.

```bash
wc -l search/search.py
```

*You should see* `78 search/search.py` or `79 search/search.py`, depending on the blank lines between pastes. Over 100 means a piece was pasted twice; the file has exactly one `def` of each of `chunk`, `embed`, `normalize`, `search`, `words` and `keyword_search`, and one `if __name__` block.

**Extension — ten queries and a cutoff (ASSIGNED)**

**E1. Write the queries and the prediction first.** `search/queries.txt` is a plain text file, one query per line, in this format: the question, a space, a `|`, a space, and a short phrase copied exactly from the paragraph that answers it. Two more lines ask something your corpus cannot answer and end in ` | none`. Lines starting with `#` are ignored. This is the whole Homer file, twelve lines:

```text
Who tricks the Cyclops by giving a false name? | my name is Noman
What does the goddess hand the swimmer so he will not drown? | bound Ino’s veil under his arms
How long did the hero live with the nymph on her island? | stayed with Calypso seven years straight on end
Who is the swineherd that shelters the disguised beggar? | swineherd Eumaeus
Which prophet must be consulted in the house of Hades? | blind Theban prophet Teiresias
Who taught Pandarus to use the bow? | whom Apollo had taught to use the bow
How many axes are set up for the contest of the bow? | set up twelve axes in the court
Who holds the meat while Achilles chops it? | Automedon held the meat
What is the beggar Irus really called? | was Arnaeus, but the young men of the place called him Irus
Where does Apollo stand when he calls to the Trojans? | Apollo looked down from Pergamus
What year was the Odyssey first printed in English? | none
What color was the scar on Ulysses' leg? | none
```

Yours come from your file, and here is how to find ten facts in an hour you do not have. For each one:

1. Search the corpus for a name, a place or a term you know is in it, and pick one line number from the output:

   ```bash
   grep -n -i "noman" data/corpus.txt | head
   ```

   On Homer this printed six lines, the first `4082:therefore, the present you promised me; my name is Noman; this is what`.

2. Read the paragraph around that line, with your line number in place of 4082 and about eight lines either side:

   ```bash
   sed -n '4074,4090p' data/corpus.txt
   ```

3. Write a question that paragraph answers, **in words the paragraph does not use**, and copy four to eight words from the paragraph, exactly, as the expected phrase. The Homer paragraph says "Cyclops, you ask my name and I will tell it you ... my name is Noman", so "Who tricks the Cyclops by giving a false name?" shares `cyclops` and `name` with it. The paragraph behind the second Homer query says "bound Ino's veil under his arms, and plunged into the sea, meaning to swim on shore"; the question says goddess, swimmer and drown, three words that paragraph never uses. At least three of your ten have to be built that way, because they are the only ones that can tell semantic search apart from keyword search; the checker below says which of yours qualify, and the Homer file passes it on one query cleanly (the veil) and on two more if you allow one shared noun (`cyclops`, `nymph`), which is a fair reading of the rule.

4. The two `| none` questions stay on your corpus's topic, so they are hard to tell apart from a real question: a fact about the author, a date, a color the text never gives.

Before you commit, check every line with this command. For each answerable query it says how many chunks hold the expected phrase and which words the question shares with the first of them:

```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
from search.search import chunk, CORPUS, words
chunks = chunk(CORPUS.read_text(encoding='utf-8'))
for line in open('search/queries.txt', encoding='utf-8'):
    if ' | ' not in line or line.startswith('#'): continue
    q, phrase = [s.strip() for s in line.split(' | ')]
    if phrase == 'none': continue
    hits = [c for c in chunks if ' '.join(phrase.split()).lower() in c.lower()]
    print(len(hits), 'chunk(s) hold the phrase; shared words', sorted(words(q) & words(hits[0])) if hits else 'PHRASE NOT FOUND', '|', q)
"
```

*You should see* ten lines, none of them saying `PHRASE NOT FOUND`, most saying `1 chunk(s)`. A query shares no content words when the printed list holds only words like `he`, `not`, `so`, `will`: on Homer the second query printed `['he', 'not', 'so', 'will']`, which counts, and the first printed `['cyclops', 'name']`, which does not. A phrase that lands in many chunks (`swineherd Eumaeus` is in 8 on Homer) is a weaker check in E3, so pick a longer phrase if you can. `PHRASE NOT FOUND` means you retyped the phrase instead of copying it; curly quotes and straight quotes are different characters.

Now the prediction. Open `embed/FINDINGS.md` from A05 and find your Probe 3 line, the mean over 1,000 random word pairs. Create `search/RESULTS.md` and paste this in, with your numbers:

```markdown
# A06 results: <your name>

## Prediction (before any run)

A05 word floor (Probe 3 mean): <0.xxx>
Predicted cutoff: <0.xxx>. Below this score I predict "no good answer".
Why this number: <one sentence. The cutoff has to sit above the floor, because the floor is what two
unrelated texts score; how far above is your guess, and the point of E4 is to find out.>
```

Commit before you run anything:

```bash
git add search/queries.txt search/RESULTS.md
git commit -m "A06: ten queries, expected chunks, predicted cutoff (before any run)"
```

I will check your commit timestamps. Queries written after seeing results measure nothing.

**E2. Measure the floor on chunks.** A05's floor was measured on 200 single words. Chunks are paragraphs, and their floor is not the same number. **The floor is the mean cosine between two chunks picked at random from your corpus**; the 95th percentile is the score that 95 of 100 random pairs stay under, so a top score below it is a score a random pair could have produced. Same method as A05's Probe 3, 1,000 random pairs with a fixed seed, on the chunk vectors Step 5 saved:

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

*You should see* one line, `chunk floor mean` a clearly positive number, then a `95th pct` above it. Whether the chunk floor is above or below your A05 word floor depends on your corpus: one book on one subject pulls every chunk toward the same place. Worked example with made-up numbers: if the line reads `chunk floor mean 0.300  95th pct 0.420`, a top score of 0.600 is 0.300 above the floor and well clear of the 95th percentile, so it means something; a top score of 0.410 is one a random pair produces one time in twenty, so it means almost nothing. Add both numbers to `RESULTS.md` under a new heading:

```markdown
## Floors

A05 word floor: <0.xxx>
Chunk floor mean: <0.xxx>   95th percentile: <0.xxx>
The chunk floor is <above / below> the word floor, and I think that is because <one sentence>.
```

<!-- JD: fill the Homer chunk floor mean and 95th pct after a run; E2 is API-dependent. -->

**E3. Run and mark.**

```bash
uv run python search/search.py > search/run1.txt
grep -c "^##" search/run1.txt
```

*You should see* `12`: one `##` block per line of `queries.txt`. Under each is a `sem` line and a `kw` line for rank 1, then the same for ranks 2 and 3. Each `sem` line is the cosine score, the chunk index in brackets and the chunk's first 90 characters; each `kw` line is the overlap count, the index and the same preview. On Homer the keyword half of the first block was `kw      3 [322] '“Therefore, Sir, do you on your part affect no more concealment nor reserve in the matter '`, and so on.

<!-- JD: fill the Homer `sem` lines after a run. The `kw` lines above and in the table below are computed offline and are exact. -->

Now the definition. **A method got a query when at least one of its three chunks contains a sentence that answers the question.** You decide by reading the chunk, not by checking whether your expected phrase appears in it; the phrase is where you expected the answer, not the only place it can be. Ninety characters is not enough to read, so print any chunk in full by its index, with your number in place of 399:

```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
from search.search import chunk, CORPUS
chunks = chunk(CORPUS.read_text(encoding='utf-8'))
print(chunks[399])
"
```

Two rows worked, on the Homer keyword results. For "Which prophet must be consulted in the house of Hades?" the top keyword chunk was `[399]` with overlap 5; printed in full it begins "And the goddess answered, 'Ulysses, noble son of Laertes ... You must go to the house of Hades and of dread Proserpine to consult the ghost of the blind Theban prophet Teiresias", which answers the question, so keyword got it. For "What does the goddess hand the swimmer so he will not drown?" the top keyword chunk was `[2124]` with overlap 6, "Thus spoke Priam, and the heart of Achilles yearned as he bethought him of his father": six shared words, four of them `he`, `not`, `so` and `will`, the other two `goddess` and `hand`, and nothing about a veil in any of the three, so keyword missed it. Build this table in `RESULTS.md`, one row per answerable query:

```markdown
## Marked results

| # | query | semantic top score | semantic got it | keyword top overlap | keyword got it |
|---|---|---|---|---|---|
| 5 | Which prophet must be consulted in the house of Hades? | <0.xxx> | <yes/no> | 5 | yes |
| 2 | What does the goddess hand the swimmer so he will not drown? | <0.xxx> | <yes/no> | 6 | no |

Semantic got <X> of 10. Keyword got <Y> of 10.
```

*You should see* the two methods agree on most queries and disagree on a few. On Homer, keyword got 6 of 10, and the four it missed were the queries whose paragraph the question avoids: the Cyclops, the veil, the nymph's island, the swineherd. If semantic gets ten of ten, your three no-shared-words queries were not honest; rewrite them and say so in the log. A keyword win is a real result: a rare name or number is something a 1,536-number summary of a paragraph holds on to badly.

<!-- JD: fill "semantic got N of 10" on Homer after a run. Keyword 6 of 10 is exact (expected phrase in top-3 keyword chunk; by-reading marks may differ by one). -->

**E4. Test the cutoff.** The prediction in E1 said: below this score, no good answer. Twelve top scores test it: ten answerable queries, each marked got or miss from your table, and the two off-corpus queries, marked none. **A score is on the wrong side of a cutoff when the cutoff says one thing and the mark says the other**: a score at or above the cutoff with a mark of miss or none (the cutoff let through a wrong answer), or a score below the cutoff with a mark of got (the cutoff threw away a right one). A miss counts the same as a none, because the cutoff's one job is to say "nothing good here", and a miss is exactly that.

Create `scratch/a06-cutoff.py` and paste this in. It reads the twelve top scores from `search/run1.txt`, so the only two lines to edit are `MARKS` and `PREDICTED`:

```python
# scratch/a06-cutoff.py
# Run from the repo root:  uv run python scratch/a06-cutoff.py
# Reads the top semantic score of every query from search/run1.txt, lines them up with your
# marks, draws your predicted cutoff on the line, counts the wrong side, then finds the
# cutoff that would have made the fewest mistakes.
from pathlib import Path

MARKS = "got got got got got got got got got got none none"   # one word per query, in the order of queries.txt:
                                                                # got / miss for the ten answerable, none for the two off-corpus
PREDICTED = 0.000                                               # the cutoff you committed in E1

marks = MARKS.split()
lines = Path("search/run1.txt").read_text(encoding="utf-8").splitlines()
queries, scores = [], []
for k, line in enumerate(lines):
    if line.startswith("## "):
        queries.append(line[3:])
        scores.append(float(lines[k + 1].split()[1]))   # the next line is "  sem 0.512 [i] '...'"
assert len(scores) == len(marks) == 12, f"{len(scores)} queries in run1.txt, {len(marks)} marks; both must be 12"

def wrong_side(cutoff):
    # A score at or above the cutoff claims "this has an answer". That claim is wrong when the
    # mark is miss or none. A score below the cutoff claims "no good answer", wrong when the mark is got.
    return [q for q, s, m in zip(queries, scores, marks)
            if (s >= cutoff and m != "got") or (s < cutoff and m == "got")]

rows = sorted(zip(scores, marks, queries), reverse=True)
above = [f"{s:.3f}{m[0]}" for s, m, q in rows if s >= PREDICTED]
below = [f"{s:.3f}{m[0]}" for s, m, q in rows if s < PREDICTED]
print("twelve top scores, highest first (g = got, m = miss, n = none), | = your predicted cutoff:")
print(" ", " ".join(above), "|", " ".join(below))
w = wrong_side(PREDICTED)
print(f"  wrong side of {PREDICTED:.3f}: {len(w)}")
for q in w:
    print("    " + q)

print("\ncutoff      mistakes")
best = None
for c in sorted(set(scores)):
    n = len(wrong_side(c))
    print(f"{c:.3f}   {n:>8}")
    if best is None or n < best[1]:
        best = (c, n)
print(f"\nfewest mistakes: cutoff {best[0]:.3f} makes {best[1]} (set the cutoff at the lowest score that should count as an answer)")
```

Edit the two lines. `MARKS` is twelve words in the order of `queries.txt`, your `semantic got it` column as `got` or `miss`, then `none none`. `PREDICTED` is the number you committed in E1. Then:

```bash
uv run python scratch/a06-cutoff.py
```

*You should see* the twelve scores on one line, highest first, each tagged `g`, `m` or `n`, with a `|` where your predicted cutoff falls; the count of scores on the wrong side of it, and which queries they were; then a table that tries every distinct top score as the cutoff and counts the mistakes each would make; then the cutoff with the fewest. The arithmetic, on two rows: suppose the Cyclops query scored 0.58 and was got, and "What year was the Odyssey first printed in English?" scored 0.47 and is none, and your prediction was 0.45. Both are at or above 0.45, so the cutoff claims both have an answer; the first claim is right and the second is wrong: one mistake. Try each score as the cutoff: at 0.47, the none is still at or above, one mistake; at 0.58, the got is at or above and the none is below, zero mistakes. The fewest-mistakes cutoff on those two rows is 0.58. With twelve rows the script does the same count twelve times and reports the winner; a tie goes to the lowest score, which is the gentlest cutoff that does the job.

Paste the whole output into `RESULTS.md` under `## Cutoff`, then write the three numbers under it: your predicted cutoff and how many of the twelve were on its wrong side, the fewest-mistakes cutoff and its count, and how far the fewest-mistakes cutoff sits above your chunk floor from E2 (subtract).

<!-- JD: fill the Homer twelve-score line after a run. -->

Write the paragraph that ends `RESULTS.md`: where semantic search did worse than keyword, and where your cutoff got it wrong.

**E5. Commit, push, PR, sign off.**

```bash
git add search/search.py search/RESULTS.md search/run1.txt scratch/a06-video.py scratch/a06-cutoff.py
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
`search/search.py` (under 80 lines, NumPy and stdlib only) + `search/queries.txt` + `search/RESULTS.md` (prediction, floors, marked table, totals, cutoff output and the three numbers, closing paragraph) + `search/run1.txt` + `scratch/a06-video.py` + `scratch/a06-cutoff.py`.

**Reflection Questions**

1. Paste the two lines from Step 3, `naive` and `chunk`, and the three lines from the second snippet: the shortest piece the naive split produced, the index of the chunk it ended up inside after `chunk()`, and that chunk's first 90 characters. Say what the piece is in your file (a heading, a line number, a stage direction, a page break), and whether that chunk appeared in any of your thirty semantic results.

   *How to get it:* both snippets are in Step 3; run them again if you did not keep the output. The index is the number in square brackets on the `lands in chunk(s)` line, and it is the same numbering `search()` prints in its results, so for the last part search your run for it, with your number in place of 23:

   ```bash
   grep -n -F "[23]" search/run1.txt | grep sem
   ```

   A hit means that chunk was a top-3 semantic result for the query on that line; no output means it never appeared.

2. Your predicted cutoff, committed in E1, and the cutoff that made the fewest mistakes in E4: give both, with the commit hash of the prediction. Which off-corpus query scored highest, what was its top chunk, and why does a question your corpus cannot answer still land above the chunk floor you measured in E2?

   *How to get it:* the predicted cutoff is the line under `## Prediction` in `search/RESULTS.md`; the fewest-mistakes cutoff is the last line `scratch/a06-cutoff.py` printed. The hash is the first commit that touched the queries file:

   ```bash
   git log --oneline --follow -- search/queries.txt | tail -1
   ```

   The two off-corpus queries are the last two `##` blocks of `search/run1.txt`; the higher of their first `sem` lines is the one to report, with the chunk text on that line. For the why, look at the E2 numbers: every chunk sits above the floor because they all come from one file on one subject, and an off-topic question phrased in that subject's words lands in the same neighborhood.

3. Pick the query where semantic and keyword disagreed most. Paste both top results. Then put your corpus on the lesson 3 curve: how many chunks do you have, how long does one `Vn @ q` over them take on your machine, and how many times larger would your corpus need to be before that search, not the API call, is the slow part of a query?

   *How to get it:* "disagreed most" is the query in your E3 table where one method got it and the other did not, or where the two `[index]` values differ and the scores are furthest apart; its two top lines are in `search/run1.txt` under that `##` heading. For the timing, the row marked `<- your chunk count` in your `scratch/a06-video.py` output is the first number; this command times the real vectors and does the division:

   ```bash
   uv run python -c "
   import sys, time, numpy as np; sys.path.insert(0, '.')
   from search.search import normalize
   Vn = normalize(np.load('search/chunks.npy')); q = Vn[0]
   t = time.perf_counter()
   for _ in range(200): Vn @ q
   dt = (time.perf_counter() - t) / 200
   print(len(Vn), 'chunks;', round(dt * 1e6), 'microseconds per search;', round(0.3 / dt), 'x more chunks before the search matches a 300 ms API call')
   "
   ```

   *You should see* a few hundred microseconds and a multiplier in the hundreds or thousands. On a matrix of Homer's shape, 2,142 by 1,536, the search took about 300 microseconds on my machine, so the multiplier was about 1,000. The 300 ms is a typical embeddings round-trip; if you timed the first run in Step 5, use your own number in place of `0.3`.

<!-- JD: the Homer timing above was measured offline on a random float32 matrix of the same shape as chunks.npy; the time depends on the shape, not the contents, but re-run it on the real file if you want the exact number. -->
