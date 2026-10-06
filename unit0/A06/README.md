# A06 · Semantic Search from Scratch (NumPy Only)

**Meetings:** D13–D14 · **Points:** 15 pts

**Watch — 22 min**

**Day 1 — none.** Corpus day, shared with A05b. Run the checker, fix what it flags, then do Steps 2 and 3 below before you leave.

**Day 2 — 22 min**
[Vector Databases: from Embeddings to Applications](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/) · lesson 2, How to Obtain Vector Representations of Data (11m, code) · lesson 3, Search for Similar Vectors (6m, code)
[How Transformer LLMs Work](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/) · lesson 3, (Word) Embeddings (5m), a rewatch
 · Have the DLAI notebook open in one window and your repo open in VS Code in the other.

**During the video**

He runs his code on vectors he made up. You run the same operations on your corpus, once, before you press play:

```bash
uv run python scratch/a06-video.py
```

The script needs the `search/search.py` from Day 1 and nothing from the API. Part 1 turns three chunks of your corpus into word-count vectors and prints the four distances from lesson 2 on them. Part 2 is brute-force search in his shape, then the timing curve as the number of vectors grows. Part 3 prints the shape of your lookup table.

*You should see* a distance table with four rows, one brute-force result, a six-row timing table and one shape line, in a few seconds. Paste all of it under `## During the video` in `evidence/A06.md`.

*If it broke* with `ModuleNotFoundError: No module named 'search'`, Day 1 is not done: `search/search.py` has to exist with `chunk` in it.

**Lesson 2 · the four distances, on three paragraphs of your corpus.** When he writes Euclidean, Manhattan, dot product and cosine, read Part 1 as he writes each one. A and B are next to each other in your file, C is far away. **Euclidean, Manhattan and the dot product all grow with the length of the paragraph**; the `A vs A` row shows it, because a chunk against itself is not 1 but the sum of its squared counts. Cosine ignores length, and `norm-dot` is the line he does not write: normalize both vectors first, then take the dot product. It matches cosine to the last digit, which is why Step 5 normalizes every chunk once instead of computing cosine every time. On Homer `A vs B` read `12.806  132.0  104.0  0.5631  0.5631` and `A vs A` read `0.000  0.0  208.0  1.0000  1.0000`. Count vectors are a bad embedding, and the table shows why: on Homer the two neighbors scored 0.563 and a chunk from the other poem 0.531, because `the`, `and` and `he` dominate every count. Step 4 pays for a vector that knows what the paragraph is about.

**Lesson 3 · brute force, and what it costs.** His search is one query against every stored vector, sort, keep the top few: `X @ q`, `argsort`, `[::-1][:k]`. That is `brute_force` in Part 2, and it is `search()` in Step 5 with the API call taken out. The result line shows row 7 finding itself at 1.0 and two strangers near 0.07, which is what random vectors in 1,536 dimensions look like. *You should see* the microseconds roughly double when N doubles: brute force is a straight line in N. On Homer, 1,000 vectors took about 130 microseconds, 20,000 about 2,700, and the row marked `<- your chunk count` about 300. Reflection Question 3 compares that row with the API call.

**(Word) Embeddings · one line in `evidence/A06.md`.** When the lookup table appears, one row per token, write down the shape Part 3 printed in the `<answer: ...>` slot under `## During the video`: one row per chunk of your corpus, 1,536 wide. On Homer it printed `(2142, 1536)`. Step 4 builds that table and Step 5 searches it.

**Notes**

**Your corpus is locked.** `data/corpus.txt` is the file A05b's checker passed. Do not swap it; every extension for the rest of the year runs on it.

**Banned:** Chroma, FAISS, Pinecone, pgvector, LangChain, LlamaIndex, `SentenceTransformer`, any `VectorStore`, any `.similarity_search()`. **Allowed:** `numpy`, the standard library, and a direct call to the embeddings endpoint. `search/search.py` lands at about a hundred lines, comments included, with the query runner in it. If you are at 250, you imported something on that list.

**Short chunks lie.** A three-character chunk gets an embedding that still comes back with a high score, because short strings land in strange places in the space. Step 3 merges them away.

**Embed once, and never change the chunker after you embed.** `search/chunks.npy` is saved on the first run and loaded after that; it is git-ignored, and your key has a hard cap that does not warn you. If you touch `chunk()` after embedding, the saved rows no longer match the chunks. The `assert` in Step 5 catches a count change but not a same-count change, so delete `search/chunks.npy` and re-embed.

**Walkthrough — search your own corpus**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
git switch main && git pull
git switch -c dev/semantic-search
```

Step 2 of A05b's template update, or the pull request I opened on your repo, put `a06-video.py` and `a06-cutoff.py` in `scratch/`, and `evidence/A06.md`; `ls scratch evidence` should show them.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Day 1: corpus passes A05b's checker
- [ ] Day 1: Steps 2–3, search.py started, both chunk commands run and pasted in evidence/A06.md
- [ ] Day 2: scratch/a06-video.py run, output in evidence/A06.md, the three lessons read against it
- [ ] Day 2: Steps 4–6, embed once, self-retrieval check, keyword baseline
- [ ] Extension: queries and prediction committed before any run; run, marked, cutoff tested
- [ ] Fill every slot in evidence/A06.md (check_evidence.py: all slots filled); push and open the PR
```

Two meetings, one branch. Push again each day; one PR, not two; one log entry per meeting.

**Step 2. Make the file.**

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
mkdir -p search
ls -la data/corpus.txt
```

*You should see* your corpus with the size A05b's checker reported. If `ls` says no such file, run `bash scripts/fetch_corpus.sh` and check again.

Create `search/search.py` (right-click `search`, **New File**) and paste the top of the file:

```python
# search/search.py
# A search engine over data/corpus.txt in NumPy and the standard library: chunk the corpus, embed
# every chunk once, score a question against every chunk by cosine, and compare with keyword overlap.
# Run from the repo root:  uv run python search/search.py --check   (Step 5: self-retrieval check)
#                          uv run python search/search.py           (E3: runs every query in search/queries.txt)
# Nothing in this file is yours to edit.
import json, os, re, sys, urllib.request
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent     # this file sits in search/, so .parent.parent is the repo root
CORPUS = ROOT / "data" / "corpus.txt"             # / joins path pieces, whatever folder you run from
VECS = ROOT / "search" / "chunks.npy"             # where the embedded vectors are saved
STOP = {"the", "a", "an", "of", "and", "to", "in", "is", "was", "what", "who", "how", "why", "does", "did"}   # words the keyword baseline ignores
```

**Step 3. Chunk, and look at the distribution before you embed anything.**

Paste this below the `STOP` line. `chunk()` turns the corpus into a list of passages between 200 and 2,000 characters, one embedding each: it splits on blank lines, collapses whitespace so the text you embed is exactly the text you search, cuts anything over `max_chars` because the endpoint refuses long inputs, and folds anything under `min_chars` into its neighbor.

```python
# In: the whole corpus as one string. Out: a list of passages, each between min_chars and max_chars long.
def chunk(text, min_chars=200, max_chars=2000):
    pieces = []
    for p in re.split(r"\n\s*\n", text):          # split on a blank line (a newline, optional spaces, another newline)
        p = " ".join(p.split())                   # collapse every run of whitespace to one space
        for i in range(0, len(p), max_chars):     # i = 0, 2000, 4000, ... up to the length of p
            pieces.append(p[i:i + max_chars])     # the slice from i to i + max_chars, so no piece is over max_chars
    out = []
    for p in pieces:
        if out and len(out[-1]) < min_chars:      # out[-1] is the last chunk so far; if it is short, glue p onto it
            out[-1] += " " + p
        else:
            out.append(p)
    if len(out) > 1 and len(out[-1]) < min_chars: # a short last chunk folds into the one before it
        out[-2] += " " + out.pop()                # pop() removes and returns the last chunk
    return out
```

Compare it with the naive split, the same blank-line split with nothing merged. From the repo root:

```bash
uv run python -c "
import re, sys; sys.path.insert(0, '.')                    # '.' is the repo root, so 'from search.search' works
from search.search import chunk, CORPUS
text = CORPUS.read_text(encoding='utf-8')
naive = []
for p in re.split(r'\n\s*\n', text):                      # the same blank-line split chunk() starts from
    if p.strip():                                          # keep the piece unless it is only whitespace
        naive.append(p)
for name, cs in [('naive', naive), ('chunk', chunk(text))]:
    L = sorted(len(c) for c in cs)                         # every passage length, smallest first
    print(name, len(cs), 'min', L[0], 'median', L[len(L)//2], 'max', L[-1])   # L[-1] is the last, the biggest
"
```

*You should see* two lines: `naive` with a minimum in the single or low double digits, `chunk` with a minimum of exactly 200 or a little above, a maximum no higher than about 2,200, and a count of at least 300. On Homer:

```text
naive 2487 min 6 median 529 max 3745
chunk 2142 min 200 median 596 max 2095
```

The checker's `chunks` line said 2079 on the same file because it drops short pieces where `chunk()` merges them; the count you carry from here on is the `chunk` one.

*If it broke:* a `chunk` count under 300 means your corpus is too small for these settings; change `min_chars=200` to `min_chars=150` in the `def chunk` line and write down that you did, in the `<answer: ...>` slot under `## Step 3` in `evidence/A06.md`.

Now find the piece behind that `naive` minimum, and where `chunk()` put it:

```bash
uv run python -c "
import re, sys; sys.path.insert(0, '.')
from search.search import chunk, CORPUS
text = CORPUS.read_text(encoding='utf-8')
naive = []
for p in re.split(r'\n\s*\n', text):                      # the same blank-line split chunk() starts from
    if p.strip():
        naive.append(p)
frag = ' '.join(min(naive, key=len).split())             # key=len: the shortest piece; join/split collapses its whitespace as chunk() does
print('shortest naive piece:', repr(frag), '(', len(frag), 'chars )')
chunks = chunk(text)
hits = [i for i, c in enumerate(chunks) if (' ' + frag + ' ') in (' ' + c + ' ')]   # same as: for each chunk c at index i, keep i if the piece is in c; the padded spaces make it match whole words only
print('lands in chunk(s)', hits, 'of', len(chunks))
print(repr(chunks[hits[0]][:90]) if hits else '(not found)')   # the first 90 characters of the first chunk that holds it, or a note if none does
"
```

*You should see* the piece in quotes, the index of the chunk it was absorbed into, and that chunk's first 90 characters. On Homer:

```text
shortest naive piece: 'BOOK I' ( 6 chars )
lands in chunk(s) [23] of 2142
'HENRY FESTING JONES. 120 MAIDA VALE, W.9. 4th _December_, 1921. THE ODYSSEY BOOK I THE GOD'
```

The piece is almost always a heading, a page break or a stray line number. Paste both outputs under `## Step 3` in `evidence/A06.md`; Reflection Question 1 asks for them.

**Step 4. Embed in batches, sorted by `index`.**

Paste this below `chunk()`. `embed()` is A05's function with a loop around it: a hundred texts per request, one row of 1,536 numbers per text. **The `items.sort` line keeps `chunks[i]` and `V[i]` the same passage**, because arrival order is not input order. Nothing to run yet.

```python
# In: a list of strings. Out: a NumPy matrix with one row of 1,536 numbers per string, in the same order.
def embed(texts, batch=100):
    rows = []
    for i in range(0, len(texts), batch):                     # i = 0, 100, 200, ...: the start of each batch
        body = json.dumps({"model": "text-embedding-3-small", "input": texts[i:i + batch]}).encode()   # the request, as JSON bytes
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings", data=body,
            headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],     # your key, read from the environment
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:                # send it and wait for the reply
            items = json.load(r)["data"]                      # a list of {"index": ..., "embedding": [...]}, one per text
        items.sort(key=lambda it: it["index"])                # put the replies back in the order the texts were sent (lambda: sort by each item's "index")
        for it in items:
            rows.append(it["embedding"])
    return np.array(rows, dtype=np.float32)                   # one row per text; float32 halves the file size
```

**Step 5. Normalize, search, and check that the search finds what you embedded.**

Paste this below `embed()`. `normalize()` makes every row length 1, so a dot product is a cosine; `keepdims=True` is A05's `[:, None]` written so the same line works on one vector or a whole matrix. `search()` embeds one query and scores it against every chunk in one line, `Vn @ q`, which is one row of A05's `S`. Compare it with `brute_force` in the video script: apart from the first line, it is the same search.

```python
# In: a vector, or a matrix of vectors (one per row). Out: the same, with every vector scaled to length 1.
def normalize(V):
    return V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-10)   # norm = each row's length; keepdims keeps it as a column so the division lines up; np.maximum guards a zero length

# In: a question as a string, and the normalized chunk matrix. Out: the k best (chunk index, cosine score) pairs, best first.
def search(query, Vn, k=3):
    q = normalize(embed([query])[0])          # embed the question (a list of one), take row 0, scale to length 1
    sims = Vn @ q                             # one cosine score per chunk: each row of Vn dotted with q
    top = np.argsort(sims)[::-1][:k]          # argsort sorts lowest first; [::-1] reverses so the highest comes first; [:k] keeps k
    result = []
    for i in top:
        result.append((int(i), float(sims[i])))   # (chunk index, score) as plain Python numbers
    return result
```

Then paste the main block at the very bottom of the file. It chunks the corpus, embeds once and saves, loads on every run after that, and prints the shapes. With `--check` it searches for the first, middle and last chunk by their own text, because an off-by-one only shows at an end. Without `--check` it runs the queries file, which does not exist yet, and calls `keyword_search`, which Step 6 adds: **until Step 6, run it only with `--check`**. Then run it twice:

```python
# Runs only when you run this file directly, not when another script imports it.
if __name__ == "__main__":
    chunks = chunk(CORPUS.read_text(encoding="utf-8"))
    if VECS.exists():                         # embedded before: load the saved vectors instead of paying again
        V = np.load(VECS)
    else:
        V = embed(chunks)
        np.save(VECS, V)
    assert len(V) == len(chunks), "chunking changed since you embedded: delete search/chunks.npy"   # stops with this message if the counts differ
    Vn = normalize(V)
    print(len(chunks), "chunks", V.shape, Vn.shape, np.allclose(np.linalg.norm(Vn, axis=1), 1.0))   # True when every row has length 1
    if sys.argv[1:] == ["--check"]:           # sys.argv[1:] is the list of words typed after the file name
        for i in (0, len(chunks) // 2, len(chunks) - 1):     # first, middle, last
            print(i, search(chunks[i], Vn, k=2))
    else:
        for line in (ROOT / "search" / "queries.txt").read_text(encoding="utf-8").splitlines():
            q = line.split(" | ")[0].strip()  # the question is the part before " | "
            if not q or q.startswith("#"):    # skip blank lines and comment lines
                continue
            print("\n##", q)
            sem_hits = search(q, Vn)                  # three (chunk index, score) pairs, best first
            kw_hits = keyword_search(q, chunks)       # three (chunk index, shared-word count) pairs, best first
            for rank in range(3):
                i, s = sem_hits[rank]                 # i = chunk index, s = cosine score
                j, n = kw_hits[rank]                  # j = chunk index, n = shared-word count
                # :.3f = three decimals; :>5 = right-aligned in 5 spaces; [:90] = the first 90 characters; !r = in quotes
                print(f"  sem {s:.3f} [{i}] {chunks[i][:90]!r}\n  kw  {n:>5} [{j}] {chunks[j][:90]!r}")
```

```bash
uv run python search/search.py --check
uv run python search/search.py --check
```

*You should see* the same four lines both times: your chunk count with `(n, 1536)` twice and `True`, then three lines, each starting with the index you searched with, then that same index again at 0.999 or higher inside the brackets, then some other chunk clearly lower. The first run takes seconds and spends money; the second is instant, or the load branch is not being taken. Paste the second run's four lines under `## Step 5` in `evidence/A06.md`. On Homer the first line is `2142 chunks (2142, 1536) (2142, 1536) True` and the three indexes are `0`, `1071` and `2141`.

<!-- JD: fill the three Homer --check lines after a run; the shape is "i [(i, 0.999+), (j, lower)]". -->

*If it broke* with `KeyError: 'OPENAI_API_KEY'`, open a new terminal, as in A05. Another index at rank 1 means `chunks[i]` and `V[i]` are not the same passage, and the cause is almost always a missing `items.sort`. The right index at 0.97 rather than 0.999 means the text you searched with is not byte-for-byte the text you embedded.

**Step 6. The keyword baseline.**

Paste these two functions directly below `search()`, above the `if __name__ == "__main__":` line. `words()` is the set of distinct words in a text minus `STOP`; `keyword_search()` scores every chunk by how many of the query's words it contains and keeps the top three. **It knows nothing about meaning, which is what makes it the baseline.**

```python
# In: a string. Out: the set of distinct words in it, lowercased, minus the STOP words.
def words(s):
    return set(re.findall(r"[a-z0-9']+", s.lower())) - STOP     # a "word" is a run of letters, digits or apostrophes (same rule as the video script); - removes the stop words

# In: a question and the list of chunks. Out: the k best (chunk index, number of shared words) pairs, best first.
def keyword_search(query, chunks, k=3):
    qw = words(query)
    scores = []
    for c in chunks:
        scores.append(len(qw & words(c)))     # & keeps the words in both sets; len counts them
    scores = np.array(scores)
    top = np.argsort(scores)[::-1][:k]        # same trick as search(): highest score first, keep k
    result = []
    for i in top:
        result.append((int(i), int(scores[i])))
    return result
```

Do not run the runner yet; writing the queries file is the extension. Count the lines instead:

```bash
wc -l search/search.py
```

*You should see* about `106 search/search.py`, give or take the blank lines between pastes; paste the line under `## Step 6` in `evidence/A06.md`. Over 130 means a piece was pasted twice; the file has exactly one `def` of each of `chunk`, `embed`, `normalize`, `search`, `words` and `keyword_search`, and one `if __name__` block.

**Extension — ten queries and a cutoff (ASSIGNED)**

**E1. Write the queries and the prediction first.** `search/queries.txt` is a plain text file, one query per line: the question, a space, a `|`, a space, and a short phrase copied exactly from the paragraph that answers it. Two more lines ask something your corpus cannot answer and end in ` | none`. Five of the Homer file's twelve lines (queries 1, 2, 5, 11 and 12):

```text
Who tricks the Cyclops by giving a false name? | my name is Noman
What does the goddess hand the swimmer so he will not drown? | bound Ino’s veil under his arms
Which prophet must be consulted in the house of Hades? | blind Theban prophet Teiresias
What year was the Odyssey first printed in English? | none
What color was the scar on Ulysses' leg? | none
```

Yours come from your file. For each one, search for a name or term you know is in it, pick one line number from the hits, and read the paragraph around it:

```bash
grep -n -i "noman" data/corpus.txt | head      # on Homer the first hit was line 4082
sed -n '4074,4090p' data/corpus.txt           # about eight lines either side of your line number
```

Write a question that paragraph answers **in words the paragraph does not use**, and copy four to eight words from it, exactly, as the expected phrase. The paragraph behind the veil query says "bound Ino's veil under his arms, and plunged into the sea"; the question says goddess, swimmer and drown, none of which it uses. At least three of your ten have to be built that way, because they are the only ones that can tell semantic search apart from keyword search. The two `| none` questions stay on your corpus's topic, so they are hard to tell apart from a real question: a fact about the author, a date, a color the text never gives.

Before you commit, check every line. For each answerable query this says how many chunks hold the expected phrase and which words the question shares with the first of them:

```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
from search.search import chunk, CORPUS, words
chunks = chunk(CORPUS.read_text(encoding='utf-8'))
for line in open('search/queries.txt', encoding='utf-8'):
    if ' | ' not in line or line.startswith('#'): continue   # skip lines that are not queries
    parts = line.split(' | ')                               # parts[0] is the question, parts[1] the expected phrase
    q = parts[0].strip()                                    # strip() removes spaces and the newline at the ends
    phrase = parts[1].strip()
    if phrase == 'none': continue
    hits = [c for c in chunks if ' '.join(phrase.split()).lower() in c.lower()]   # same as: for each chunk c, keep it if the phrase is in it, ignoring case
    if hits:
        shared = sorted(words(q) & words(hits[0]))          # the words the question and the first hit have in common; & = in both sets
    else:
        shared = 'PHRASE NOT FOUND'
    print(len(hits), 'chunk(s) hold the phrase; shared words', shared, '|', q)
"
```

*You should see* ten lines, none saying `PHRASE NOT FOUND`, most saying `1 chunk(s)`; paste them under `## Extension E1` in `evidence/A06.md`. A query shares no content words when the printed list holds only words like `he`, `not`, `so`, `will`: on Homer the veil query printed `['he', 'not', 'so', 'will']`, which counts, and the Cyclops query printed `['cyclops', 'name']`, which does not. `PHRASE NOT FOUND` means you retyped the phrase instead of copying it; curly and straight quotes are different characters.

Now the prediction. Open `embed/FINDINGS.md` from A05 and find your Probe 3 line, the mean over 1,000 random word pairs. Fill the three slots under `## Prediction` in `evidence/A06.md`: that word floor, your predicted cutoff (below this score you predict "no good answer"), and one sentence on why that number: the cutoff sits above the floor, and how far above is your guess.

```bash
git add search/queries.txt evidence/A06.md
git commit -m "A06: ten queries, expected chunks, predicted cutoff (before any run)"
```

Commit before you run anything. I will check your commit timestamps: queries written after seeing results measure nothing.

**E2. Measure the floor on chunks.** A05's floor was measured on 200 single words; chunks are paragraphs, and their floor is a different number. **The floor is the mean cosine between two chunks picked at random from your corpus**, and the 95th percentile is the score 95 of 100 random pairs stay under, so a top score below it is one a random pair could have produced. Same method as A05's Probe 3, on the vectors Step 5 saved:

```bash
uv run python -c "
import sys, numpy as np; sys.path.insert(0, '.')
from search.search import normalize
Vn = normalize(np.load('search/chunks.npy'))                               # the saved vectors, scaled to length 1
i, j = np.random.default_rng(0).integers(0, len(Vn), (2, 1000)); keep = i != j   # two lists of 1,000 random chunk indexes (seed 0); keep marks the pairs that are two different chunks
s = (Vn[i[keep]] * Vn[j[keep]]).sum(axis=1)                               # the cosine of each pair: multiply slot by slot, add across the row
print('chunk floor mean', round(float(s.mean()), 3), ' 95th pct', round(float(np.percentile(s, 95)), 3))   # the average, and the score 95% of pairs stay under
"
```

*You should see* one line, a clearly positive `chunk floor mean` and a `95th pct` above it. If it read `chunk floor mean 0.300  95th pct 0.420`, a top score of 0.600 would mean something and a top score of 0.410 would be one a random pair produces one time in twenty. Whether the chunk floor sits above or below your word floor depends on your corpus: one book on one subject pulls every chunk toward the same place. Paste the line under `## Extension E2` in `evidence/A06.md`, and in the `<answer: ...>` slot under it say in one sentence why the chunk floor is above or below your A05 word floor.

<!-- JD: fill the Homer chunk floor mean and 95th pct after a run; E2 is API-dependent. -->

**E3. Run and mark.**

```bash
uv run python search/search.py > search/run1.txt
grep -c "^##" search/run1.txt
```

*You should see* `12`: one `##` block per query; paste that number under `## Extension E3` in `evidence/A06.md`. Under each are three `sem` lines (cosine score, chunk index in brackets, first 90 characters) interleaved with three `kw` lines (overlap count, index, preview), ranks 1 to 3.

**A method got a query when at least one of its three chunks contains a sentence that answers the question.** You decide by reading the chunk, not by checking whether your expected phrase is in it. Ninety characters is not enough to read, so print any chunk in full by its index, with your number in place of 399:

```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
from search.search import chunk, CORPUS
chunks = chunk(CORPUS.read_text(encoding='utf-8'))
print(chunks[399])                      # the chunk at index 399, in full
"
```

One row worked, on the Homer keyword results: for "Which prophet must be consulted in the house of Hades?" the top keyword chunk was `[399]` with overlap 5, and printed in full it says "You must go to the house of Hades ... to consult the ghost of the blind Theban prophet Teiresias", which answers the question, so keyword got it. The table is already in `evidence/A06.md` under `## Extension E3`, one row per answerable query, in the order of `queries.txt`; fill the `<   >` cells (query, semantic top score, semantic got it as yes or no, keyword top overlap, keyword got it) and the two totals in the sentence under it. On Homer the prophet row ends `| 5 | yes |`.

*You should see* the two methods agree on most queries and disagree on a few. On Homer, keyword got 6 of 10, and the four it missed were the queries whose paragraph the question avoids: the Cyclops, the veil, the nymph's island, the swineherd. If semantic gets ten of ten, your three no-shared-words queries were not honest; rewrite them and say so in the log. A keyword win is a real result: a rare name or number is something a 1,536-number summary of a paragraph holds on to badly.

<!-- JD: fill "semantic got N of 10" on Homer after a run. Keyword 6 of 10 is exact (expected phrase in a top-3 keyword chunk; by-reading marks may differ by one). -->

**E4. Test the cutoff.** The prediction in E1 said: below this score, no good answer. Twelve top scores test it: the ten answerable queries, each marked got or miss from your table, and the two off-corpus queries, marked none. **A score is on the wrong side of a cutoff when the cutoff says one thing and the mark says the other**: a score at or above the cutoff with a mark of miss or none (the cutoff let through a wrong answer), or a score below the cutoff with a mark of got (the cutoff threw away a right one). A miss counts the same as a none, because the cutoff's one job is to say "nothing good here".

Open `scratch/a06-cutoff.py` and edit its two marked lines: `MARKS` is twelve words in the order of `queries.txt`, your `semantic got it` column as `got` or `miss`, then `none none`; `PREDICTED` is the number you committed in E1. Then:

```bash
uv run python scratch/a06-cutoff.py
```

The script reads the twelve top scores from `search/run1.txt`, lines them up with your marks, draws your predicted cutoff on the line, counts the wrong side, then tries every distinct top score as the cutoff and reports the one with the fewest mistakes.

*You should see* the twelve scores on one line, highest first, each tagged `g`, `m` or `n`, with a `|` where your predicted cutoff falls; the count of scores on the wrong side of it and which queries they were; a table of cutoff against mistakes; and the fewest-mistakes cutoff. The arithmetic, on two rows: suppose the Cyclops query scored 0.58 and was got, "What year was the Odyssey first printed in English?" scored 0.47 and is none, and your prediction was 0.45. Both are at or above 0.45, so the cutoff claims both have an answer; the first claim is right and the second is wrong: one mistake. Trying each score as the cutoff: at 0.47 the none is still at or above, one mistake; at 0.58 the got is at or above and the none is below, zero mistakes, so 0.58 wins. A tie goes to the lowest score, the gentlest cutoff that does the job.

Paste the whole output under `## Extension E4` in `evidence/A06.md`, then the three numbers in the slots under it: your predicted cutoff and how many of the twelve were on its wrong side, the fewest-mistakes cutoff and its count, and how far the fewest-mistakes cutoff sits above your chunk floor from E2 (subtract). Then write the paragraph that ends the section: where semantic search did worse than keyword, and where your cutoff got it wrong.

<!-- JD: fill the Homer twelve-score line after a run. -->

**E5. Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/A06.md` (the questions are below) and run `uv run python scripts/check_evidence.py A06` until it says `all slots filled`. Then:

```bash
git add search/search.py evidence/A06.md search/run1.txt scratch/a06-cutoff.py
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
`search/search.py` (about 100 lines, NumPy and stdlib only) + `search/queries.txt` + `evidence/A06.md` with every slot filled (prediction, the Step 3, video-script, Step 5 and Step 6 output, the query check, floors, marked table and totals, cutoff output and the three numbers, closing paragraph, the three reflection answers) + `search/run1.txt` + `scratch/a06-cutoff.py` with your marks in it.

**Reflection Questions** (answer under `## Reflection` in `evidence/A06.md`)

1. Paste the two lines from Step 3, `naive` and `chunk`, and the three lines from the second command: the shortest piece the naive split produced, the index of the chunk it ended up inside after `chunk()`, and that chunk's first 90 characters. Say what the piece is in your file (a heading, a line number, a stage direction, a page break), and whether that chunk appeared in any of your thirty semantic results.

   *How to get it:* both commands are in Step 3; run them again if you did not keep the output. The index on the `lands in chunk(s)` line is the same numbering `search()` prints, so search your run for it, with your number in place of 23; a hit means that chunk was a top-3 semantic result for the query on that line, and no output means it never appeared.

   ```bash
   grep -n -F "[23]" search/run1.txt | grep sem
   ```

2. Your predicted cutoff, committed in E1, and the cutoff that made the fewest mistakes in E4: give both, with the commit hash of the prediction. Which off-corpus query scored highest, what was its top chunk, and why does a question your corpus cannot answer still land above the chunk floor you measured in E2?

   *How to get it:* the predicted cutoff is under `## Prediction` in `evidence/A06.md` and the fewest-mistakes cutoff is the last line `scratch/a06-cutoff.py` printed. The two off-corpus queries are the last two `##` blocks of `search/run1.txt`; report the higher of their first `sem` lines, with the chunk text on it. For the why, look at the E2 numbers: every chunk sits above the floor because they all come from one file on one subject, and an off-topic question phrased in that subject's words lands in the same neighborhood.

   ```bash
   git log --oneline --follow -- search/queries.txt | tail -1
   ```

3. Pick the query where semantic and keyword disagreed most. Paste both top results. Then put your corpus on the lesson 3 curve: how many chunks do you have, how long does one `Vn @ q` over them take on your machine, and how many times larger would your corpus need to be before that search, not the API call, is the slow part of a query?

   *How to get it:* "disagreed most" is the query in your E3 table where one method got it and the other did not, or where the two `[index]` values differ and the scores are furthest apart; its two top lines are in `search/run1.txt` under that `##` heading. The row marked `<- your chunk count` in your video-script output is the first timing; this command times the real vectors and does the division, and if you timed the first run in Step 5, put your own number in place of `0.3`. On a matrix of Homer's shape the search took about 300 microseconds, so the multiplier was about 1,000.

   ```bash
   uv run python -c "
   import sys, time, numpy as np; sys.path.insert(0, '.')
   from search.search import normalize
   Vn = normalize(np.load('search/chunks.npy')); q = Vn[0]    # search with the first chunk's own vector
   t = time.perf_counter()                                    # a stopwatch: the time now, in seconds
   for _ in range(200): Vn @ q                                # 200 searches, because one is too fast to time
   dt = (time.perf_counter() - t) / 200                       # seconds per search
   print(len(Vn), 'chunks;', round(dt * 1e6), 'microseconds per search;', round(0.3 / dt), 'x more chunks before the search matches a 300 ms API call')
   "
   ```

<!-- JD: the Homer timing was measured offline on a random float32 matrix of the same shape as chunks.npy; the time depends on the shape, not the contents. -->
