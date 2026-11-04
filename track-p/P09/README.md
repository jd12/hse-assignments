# P09 · Chunking

**Meetings:** D32 · **Points:** 10 pts

**Watch — none**

[Boot.dev, *Learn Retrieval Augmented Generation*](https://www.boot.dev/courses/learn-retrieval-augmented-generation) · chapter 5 Chunking, all eight lessons, about 25 minutes.

**During the chapter**

**Break the chunker on purpose.** When lesson 2's `--overlap` passes, run the `chunk` command once more with the overlap equal to the chunk size, and write what happened in `scratch/p09-chunks.md` in the rag repo: an error, a hang (press Ctrl-C), or output. Step 3 does the same to the walkthrough's chunker and the answer is different.

**Start lesson 4's build early.** `embed_chunks` embeds about 73,000 chunks and takes several minutes on a laptop. Start it, then read lessons 7 and 8 while the bar runs.

**Notes**

**Overlap equal to or larger than the size is a loop that never advances.** The step from one chunk to the next is `size - overlap`; at zero, a `range` raises an error, and a hand-written `while` loop runs forever. The walkthrough's chunker checks `overlap < size` on its first line, so the mistake is a message rather than a hang.

**The last chunk is short, and that is correct.** A window that runs past the end of the text is cut off by the slice, silently. The failure is not a short last chunk; it is a *missing* last word, when the loop stops a step early. The walkthrough checks that the last chunk ends at the last word of the file.

**Lesson 3 hands you a regular expression.** `r"(?<=[.!?])\s+"` means: split at any run of whitespace that comes right after a period, exclamation mark or question mark. You do not need to be able to write it; you need to know that is what it does, and that `re.split` with it turns a paragraph into a list of sentences.

**Walkthrough — the chapter, then chunks of your corpus with positions**

**Step 1. Merge, branch, log.**
Open the rag repo on GitHub and merge the P07 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch main && git pull
git switch -c dev/chunks
```

*If P07 is not approved yet:* `git switch dev/semantic && git switch -c dev/chunks`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Chapter 5, all eight lessons green; the chapter page saved
- [ ] scratch/p09-chunks.md: what overlap equal to size did, in the chapter's chunker and in mine
- [ ] rag/chunking.py: three checks on three configurations (Steps 3 and 4)
- [ ] Extension: SIZE and OVERLAP chosen and frozen, with the reason
- [ ] Fill every slot in evidence/P09.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. Chapter 5.**

In `~/version_control/rag-search-engine/`. When the chapter page shows every lesson complete, save it as `evidence/p09-bootdev-ch5.png` in the rag repo. Lessons 1 to 6 are code, all in `cli/lib/semantic_search.py` and `cli/semantic_search_cli.py`; 7 and 8 are readings with a question.

**Lesson 1: fixed-size chunks.** Groups of `n` words. The function goes in the library so lesson 4 can reuse it; the CLI prints.

1. In `cli/lib/semantic_search.py`, above the class:

```python
# In: text, a chunk size in words, an overlap in words.  Out: a list of chunk strings.
def chunk_words(text, chunk_size=200, overlap=0):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_size]
        chunks.append(" ".join(chunk))
        if start + chunk_size >= len(words):
            break
    return chunks
```

`overlap` is there already for lesson 2; with the default `0`, `step` is the chunk size and the windows do not share words. The `break` matters: without it, after the window that reaches the last word, the loop would keep making shorter copies of the end.

2. In the CLI, `chunk_words` joins the import, a `chunk` subcommand with a positional `text` and `--chunk-size` (default 200), and its case:

```python
    chunk_parser = subparsers.add_parser("chunk", help="Split text into fixed-size word chunks")
    chunk_parser.add_argument("text", type=str, help="The text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Words per chunk")
```

```python
        case "chunk":
            chunks = chunk_words(args.text, args.chunk_size)
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")
```

`args.chunk_size` with an underscore: argparse turns the dash in `--chunk-size` into one.

3. Run, then submit:

```bash
uv run cli/semantic_search_cli.py chunk "This is a test text with more than ten words to see how chunking works" --chunk-size 5
uv run cli/semantic_search_cli.py chunk "Short text"
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* `Chunking 70 characters` and three chunks of five, five and five words, then `Chunking 10 characters` and one chunk; those are the checks, with a ten-word one between them.

**Lesson 2: overlap.** Each chunk repeats the last `overlap` words of the one before it, so a sentence cut by a boundary is whole in at least one chunk.

1. The function already takes `overlap`. In the CLI, one more argument and pass it through:

```python
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Words shared between neighbouring chunks")
```

```python
            chunks = chunk_words(args.text, args.chunk_size, args.overlap)
```

2. Run, then submit:

```bash
uv run cli/semantic_search_cli.py chunk "This is a test text with two chunks" --chunk-size 5 --overlap 2
uv run cli/semantic_search_cli.py chunk "The bear attack was very terrifying." --chunk-size 4 --overlap 1
uv run cli/semantic_search_cli.py chunk "The bear attack was very terrifying." --chunk-size 4 --overlap 4
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* `This is a test text` then `test text with two chunks` for the first, and `The bear attack was` then `was very terrifying.` for the second; those are the checks, with a zero-overlap one too. The third line is the break: `range` refuses a step of `0` with `ValueError: range() arg 3 must not be zero`. Write that in `scratch/p09-chunks.md`. *If it broke* with a third chunk `terrifying.` on the second command, the `break` is missing.

**Lesson 3: sentence chunks.** Split at sentence ends instead of every `n` words, so a chunk is whole thoughts. The regular expression is the chapter's; the Notes say what it does.

1. At the top of `cli/lib/semantic_search.py`, `import re`. Above the class, beside `chunk_words`:

```python
# In: text, a maximum number of sentences per chunk, an overlap in sentences.  Out: a list of chunk strings.
def semantic_chunk(text, max_chunk_size=4, overlap=0):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    step = max_chunk_size - overlap
    for start in range(0, len(sentences), step):
        group = sentences[start:start + max_chunk_size]
        chunks.append(" ".join(group))
        if start + max_chunk_size >= len(sentences):
            break
    return chunks
```

The same loop as `chunk_words`, over sentences instead of words.

2. In the CLI, a `semantic_chunk` subcommand with `text`, `--max-chunk-size` (default 4) and `--overlap` (default 0), and a case that prints `Semantically chunking N characters` and the numbered chunks, the same shape as `chunk`.

3. Run, then submit:

```bash
uv run cli/semantic_search_cli.py semantic_chunk "First sentence here. Second sentence here. Third sentence here. Fourth sentence here." --max-chunk-size 2 --overlap 1
uv run cli/semantic_search_cli.py semantic_chunk "Sentence one. Sentence two. Sentence three." --max-chunk-size 3 --overlap 1
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* three two-sentence chunks for the first, each starting with the previous one's last sentence, and exactly one chunk for the second; those are two of the four checks. *If it broke* with a second chunk `Sentence three.` on the second command, the `break` is missing again.

**Lesson 4: embeddings for chunks.** The descriptions get chunked with `semantic_chunk`, four sentences with one of overlap, every chunk gets a vector, and a metadata list remembers which movie each chunk came from.

1. `import json` at the top. Below the `SemanticSearch` class, a subclass. `super().__init__(model_name)` runs the parent's constructor first, so `self.model` exists:

```python
class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    # In: a list of movie dicts.  Out: one vector per chunk of every description, saved with a map back to the movies.
    def build_chunk_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        all_chunks = []
        chunk_metadata = []
        for movie_idx, doc in enumerate(documents):
            if not doc["description"].strip():
                continue
            chunks = semantic_chunk(doc["description"], max_chunk_size=4, overlap=1)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({"movie_idx": movie_idx, "chunk_idx": chunk_idx, "total_chunks": len(chunks)})
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        os.makedirs("cache", exist_ok=True)
        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)
        with open("cache/chunk_metadata.json", "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        return self.chunk_embeddings

    # In: a list of movie dicts.  Out: the saved chunk vectors if both cache files exist, otherwise freshly built ones.
    def load_or_create_chunk_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists("cache/chunk_metadata.json"):
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            with open("cache/chunk_metadata.json") as f:
                self.chunk_metadata = json.load(f)["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)
```

`movie_idx` is the movie's position in the list, not its id; `search_chunks` uses it to get back to `self.documents[movie_idx]`.

2. In the CLI, an `embed_chunks` subcommand and its case:

```python
        case "embed_chunks":
            search = ChunkedSemanticSearch()
            embeddings = search.load_or_create_chunk_embeddings(load_movies())
            print(f"Generated {len(embeddings)} chunked embeddings")
```

3. Run, then submit. The first run is the slow one; the second loads:

```bash
uv run cli/semantic_search_cli.py embed_chunks
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* a progress bar for several minutes, then `Generated 72909 chunked embeddings`; that is the check. *If it broke* with a count a few hundred off, your `semantic_chunk` differs from the chapter's on some descriptions; lesson 6's edge cases are the usual cause, and the fix is to delete the two cache files after lesson 6 and rebuild.

**Lesson 5: search over chunks.** Every chunk is scored against the query, each movie keeps its best chunk's score, and the movies are ranked by that.

1. In the subclass:

```python
    # In: a query and how many results.  Out: up to limit movies, each scored by its best-matching chunk, best first.
    def search_chunks(self, query, limit=10):
        query_embedding = self.generate_embedding(query)
        chunk_scores = []
        for chunk_idx, embedding in enumerate(self.chunk_embeddings):
            chunk_scores.append({"chunk_idx": self.chunk_metadata[chunk_idx]["chunk_idx"],
                                 "movie_idx": self.chunk_metadata[chunk_idx]["movie_idx"],
                                 "score": cosine_similarity(query_embedding, embedding)})
        movie_scores = {}
        for cs in chunk_scores:
            m = cs["movie_idx"]
            if m not in movie_scores or cs["score"] > movie_scores[m]:
                movie_scores[m] = cs["score"]
        ranked = sorted(movie_scores, key=lambda m: movie_scores[m], reverse=True)[:limit]
        results = []
        for m in ranked:
            doc = self.documents[m]
            results.append({"id": doc["id"], "title": doc["title"], "document": doc["description"][:100], "score": round(movie_scores[m], 3)})
        return results
```

2. In the CLI, a `search_chunked` subcommand with `query` and `--limit` (default 5), and a case that loads the movies, builds a `ChunkedSemanticSearch`, loads the chunk vectors, and prints each result as the chapter shows, the title with its score to four places on one line and the first 100 characters on the next.

3. Run, then submit:

```bash
uv run cli/semantic_search_cli.py search_chunked "superhero action movie" --limit 25
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* `Kick-Ass`, `The Incredibles` and `Logan` somewhere in the 25; the checks are those and the `romantic comedy` ones. *If it broke* with `TypeError` in `search_chunks`, `chunk_metadata` is `None`: `load_or_create_chunk_embeddings` was not called first.

**Lesson 6: edge cases.** Real text has leading spaces, no final period, and empty strings. The chapter's four fixes go into `semantic_chunk`, and both the command and the builder get them, because they share the function:

```python
def semantic_chunk(text, max_chunk_size=4, overlap=0):
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and not text.endswith((".", "!", "?")):
        sentences = [text]
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    step = max_chunk_size - overlap
    for start in range(0, len(sentences), step):
        group = sentences[start:start + max_chunk_size]
        chunks.append(" ".join(group))
        if start + max_chunk_size >= len(sentences):
            break
    return chunks
```

`[s.strip() for s in sentences if s.strip()]` keeps each sentence trimmed and drops any that are empty after trimming. Then delete `cache/chunk_embeddings.npy` and `cache/chunk_metadata.json`, since the chunks may have changed, and run and submit:

```bash
uv run cli/semantic_search_cli.py semantic_chunk " Leading and trailing spaces. "
uv run cli/semantic_search_cli.py semantic_chunk " "
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* `1. Leading and trailing spaces.` with no spaces inside the quotes, and nothing numbered at all for the blank string; those are two of the four checks.

**Lessons 7 and 8: ColBERT and late chunking.** Two readings on what to do when one vector per chunk is not enough. A multiple-choice question each.

**Step 3. `rag/chunking.py`: chunks with positions.**

The chapter's fixed-size chunker, on your corpus, with one addition: every chunk is a record carrying its id and the exact character positions it came from in `data/corpus.txt`, because a citation later has to point at the file. Open the file and read `word_spans` and `chunk`: `word_spans` walks the words with `find`, so each word's start and end are known; `chunk` makes windows of `SIZE` words stepping by `SIZE - OVERLAP`, and the `break` stops after the window that reached the last word. Then:

```bash
uv run python -m rag.chunking
```

*You should see* the word count, the chunk count, two `True`s, the last chunk's word count, and the first two chunks' ids and positions. On Homer:

```text
282328 words -> 1882 chunks of 200 words, overlap 50
every chunk is an exact slice of the file: True
last chunk ends at the last word: True
last chunk words: 178
c00000 4 1192 '[Illustration]\n\n\n\n\nThe Odyssey\n\nby Homer\n\nrendered into Engl'
c00001 921 2090 'dealt with in the work just\nmentioned; I have nothing either'
```

The chunk count is `(words - overlap) / (size - overlap)` rounded up; on Homer, `(282328 - 50) / 150` is 1881.85, so 1882. Work yours out from your word count before you look. The second chunk starts at character 921, inside the first (4 to 1192), which is the overlap in positions. Paste the six lines under `## Step 3` in `evidence/P09.md`.

**Step 4. Three more runs.**

```bash
uv run python -m rag.chunking 300 0
uv run python -m rag.chunking 100 25
uv run python -m rag.chunking 100 100
```

*You should see* both checks `True` on the first two and a chunk count matching the formula (on Homer, 942 and 3765). The second chunk on the `300 0` run starts exactly one character after the first ends: no overlap, no shared text. The third run stops at once with `AssertionError: overlap 100 must be smaller than size 100`: the walkthrough refuses what the chapter's chunker would have crashed on with a `range` error, and a hand-written `while` loop would have hung on. Write the two outcomes next to each other in `scratch/p09-chunks.md`. Paste the three runs' first lines under `## Step 4`.

*If it broke* with `exact slice: False`, the text was rebuilt from the words and lost the original spacing; the chunk must be a slice of the original. *If it broke* with `ends at the last word: False`, the loop stopped a step early.

**Extension — choose SIZE and OVERLAP, and freeze them (ASSIGNED)**

From P11 on, everything runs on chunks, and every chunk id is only meaningful for one `SIZE` and `OVERLAP`. Choose yours now.

**Part 1. Before any run**, write under `## Extension` in `evidence/P09.md` the size you expect to work best for your corpus and why, one sentence. Commit the file.

**Part 2. Three configurations.** Overlap a quarter of the size in each:

```bash
uv run python -m rag.chunking 100 25
uv run python -m rag.chunking 200 50
uv run python -m rag.chunking 400 100
```

For each, the chunk count and the last chunk's word count, in a three-row table under your prediction. Then check your needles survive: every needle in `rag/queries.json` is at most ten words, and a needle fits inside some chunk whenever the overlap is at least as long as the needle, so all three configurations keep your five queries scoreable. Say so in one line, with your longest needle's word count.

**Part 3. Choose and freeze.** Set `SIZE` and `OVERLAP` in `rag/chunking.py` to your choice and write under the table: the values, the reason in two sentences, and what a passage from P02 was that a chunk is not (a chunk can start mid-sentence; a passage never did). The chapter's advice is about 20% overlap; say whether you took it.

```bash
git add rag/chunking.py scratch/p09-chunks.md evidence/
git commit -m "P09: word chunks with positions; SIZE and OVERLAP chosen"
git push -u origin dev/chunks
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**.

**Commit, push, sign off.** Fill `## Reflection` in `evidence/P09.md` (the questions are below) and run `uv run python scripts/check_evidence.py P09` until it says `all slots filled`. Then `git add evidence/ && git commit -m "P09: evidence" && git push`, and:

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`evidence/P09.md` with every slot filled + `rag/chunking.py` with your `SIZE` and `OVERLAP` + `scratch/p09-chunks.md` + `evidence/p09-bootdev-ch5.png`.

**Reflection Questions** (answer under `## Reflection` in `evidence/P09.md`)

1. Paste your `scratch/p09-chunks.md` note: what overlap equal to size did in the chapter's chunker and in `rag.chunking`. Say what the step is in each when overlap equals size, and why the walkthrough checks the inputs before the loop rather than catching the error after.

   *How to get it:* `step = size - overlap` is 0 in both; `range` with a step of 0 raises, a `while i < n: i += step` never ends. An `assert` at the top names the mistake in the message; a crash inside `range` names `range`.

2. Paste your Step 3 and `300 0` first-two-chunk lines. Say from the character positions how many characters the overlap was on the `200 50` run, and why the chunk count formula has `- overlap` in both the top and the bottom.

   *How to get it:* `c00001` starts at 921 and `c00000` ends at 1192 on Homer; the difference is the overlap in characters. For the formula: the first chunk covers `size` new words, every later chunk covers `size - overlap` new ones.

3. Paste your extension table and your choice. Say which query class you expect your choice to hurt: a long chunk blurs a short answer among unrelated sentences, a short chunk can cut an answer from the sentence that explains it. Name one of your five queries that is the risky kind for your choice.

   *How to get it:* look at the needle's passage in `rag.corpus` and count how many sentences around the needle a reader needs; compare that to your `SIZE` in words.
