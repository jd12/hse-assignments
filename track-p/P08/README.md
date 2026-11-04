# P08 · Semantic Search on Your Corpus

**Meetings:** D31 · **Points:** 10 pts

**Watch — none**

[Boot.dev, *Learn Retrieval Augmented Generation*](https://www.boot.dev/courses/learn-retrieval-augmented-generation) · chapter 4 Semantic Search, lessons 10 to 16, about 25 minutes.

**Notes**

**Embed the corpus once, then load it.** The walkthrough saves your passage vectors under `data/`, next to a small file recording which model made them, how many there are, and a fingerprint of every passage's text. The vectors are only loaded when all three match; change the model, change `MIN_CHARS`, or edit the corpus, and it re-embeds rather than pairing new text with old vectors. The chapter's `load_or_create_embeddings` checks the count only.

**Same model on both sides.** Lesson 13's whole point. Every embedding model makes vectors of one fixed length in its own space; embed passages with one model and the query with another and you get a shape error at best, or confident nonsense at worst. The walkthrough has one `MODEL` constant and one `embed` function, used for both.

**Every score here is between -1 and 1.** The walkthrough scales every vector to length 1 before anything else, so the dot product of two of them is their cosine. A score of `47.3` means a vector was not normalized.

**Walkthrough — the chapter, then your passages**

**Step 1. Branch, log.**
Same branch as P07:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch dev/semantic && git pull
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Chapter 4, lessons 10 to 16, every lesson green; the chapter page saved
- [ ] rag/semantic.py: embedded once, loaded the second time, self-retrieval on three passages (Steps 3 and 4)
- [ ] Semantic column on the scoreboard, my P07 predictions checked (Step 5)
- [ ] Fill every slot in evidence/P08.md (check_evidence.py: all slots filled); push, sign off
```

**Step 2. Finish chapter 4: lessons 10 to 16.**

In `~/version_control/rag-search-engine/`. When the chapter page shows every lesson complete, save it as `evidence/p08-bootdev-ch4.png` in the rag repo. Lessons 10, 11, 12 and 14 are code; 13, 15 and 16 are readings with a question.

**Lesson 10: one text to one vector.** The model's `encode` takes a list and returns a list, so one text goes in as a list of one and the first result comes out.

1. In the class, below `__init__`:

```python
    # In: one string.  Out: its vector from the model, a numpy array.
    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("text is empty")
        return self.model.encode([text])[0]
```

2. Below `verify_model`, the function the CLI calls, with the chapter's three prints:

```python
# In: one string.  Out: nothing; prints the text, the first three numbers of its vector, and the vector's length.
def embed_text(text):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
```

3. In the CLI, `embed_text` joins the import, an `embed_text` subcommand with one argument, and its case:

```python
    embed_text_parser = subparsers.add_parser("embed_text", help="Embed one text and print the vector's start")
    embed_text_parser.add_argument("text", type=str, help="The text to embed")
```

```python
        case "embed_text":
            embed_text(args.text)
```

4. Run, then submit:

```bash
uv run cli/semantic_search_cli.py embed_text "Luke, I am your father"
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* three numbers starting `-0.035`, `-0.016`, `0.043` and `Dimensions: 384`; those are the checks, and 384 goes into `scratch/p07-model.md`. *If it broke* with `AttributeError: 'list' object has no attribute 'shape'`, `encode` was given the string and not a list of one, and you got back a list.

**Lesson 11: every movie to a vector, saved.** The slow step, done once. `uv add numpy==2.3.3` first; the chapter pins the version.

1. At the top of `cli/lib/semantic_search.py`, two imports, and three more attributes in `__init__`:

```python
import os

import numpy as np
```

```python
        self.embeddings = None
        self.documents = None
        self.document_map = {}
```

2. In the class, the builder and the loader. Each movie becomes one string, title and description together, and `encode` on the whole list does the work with a progress bar:

```python
    # In: a list of movie dicts.  Out: one vector per movie, also saved to cache/movie_embeddings.npy.
    def build_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        os.makedirs("cache", exist_ok=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings

    # In: a list of movie dicts.  Out: the saved vectors if there is one per movie, otherwise freshly built ones.
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)
```

`[f"{doc['title']}: {doc['description']}" for doc in documents]` is a list comprehension: a loop that builds a list in one line, the same as a `for` loop with `append`.

3. Below `embed_text`, the verifier, and in the CLI a `verify_embeddings` subcommand that calls it. `load_movies` comes from `lib.search_utils`, the same one the keyword CLI uses:

```python
from lib.search_utils import load_movies
```

```python
# In: nothing.  Out: nothing; builds or loads the movie vectors and prints their shape.
def verify_embeddings():
    search = SemanticSearch()
    documents = load_movies()
    embeddings = search.load_or_create_embeddings(documents)
    print(f"Number of docs: {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
```

```python
    subparsers.add_parser("verify_embeddings", help="Build or load the movie vectors")
```

```python
        case "verify_embeddings":
            verify_embeddings()
```

4. Run, then submit:

```bash
uv run cli/semantic_search_cli.py verify_embeddings
uv run cli/semantic_search_cli.py verify_embeddings
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* a progress bar for a minute or two the first time, then `5000 vectors in 384 dimensions`, and the second run print the same line at once with no bar: it loaded. That line is the check. *If it broke* with `NameError: name 'np' is not defined`, the import is missing. *If it broke* with the bar running again on the second run, the `len` comparison failed; `cache/movie_embeddings.npy` was saved from a different movie list.

**Lesson 12: the query to a vector.** The same `generate_embedding`, printed the chapter's way:

```python
# In: a query string.  Out: nothing; prints the query, the first three numbers of its vector, and the vector's shape.
def embed_query_text(query):
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")
```

An `embed_query` subcommand with one positional argument, its case, and the import. Run, then submit:

```bash
uv run cli/semantic_search_cli.py embed_query "funny bear movies"
uv run cli/semantic_search_cli.py embed_query "FUNNY bear MOVIES"
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* `-0.072`, `-0.014`, `0.001` and a shape of `(384,)` for both, the same numbers: the model handles case itself, which is what the lesson's reading says about preprocessing. The checks are those numbers and the `scary dinosaur` ones.

**Lesson 13: same model.** A reading and a Boots question. The answer is one sentence, and the walkthrough's `MODEL` constant is it in code.

**Lesson 14: the search.** The five steps, assembled. Cosine similarity comes back as NumPy, which the lesson gives you to paste:

1. Above the class:

```python
# In: two vectors, numpy arrays.  Out: the cosine of the angle between them, 0.0 if either has no length.
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)
```

This is lesson 8's function with `np.dot` for the loop and `np.linalg.norm` for `euclidean_norm`.

2. In the class, below `load_or_create_embeddings`. `zip` walks two lists together, one movie with its vector:

```python
    # In: a query and how many results.  Out: up to limit dicts with score, title and description, best first.
    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        scored = []
        for doc, embedding in zip(self.documents, self.embeddings):
            scored.append((cosine_similarity(query_embedding, embedding), doc))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        results = []
        for score, doc in scored[:limit]:
            results.append({"score": score, "title": doc["title"], "description": doc["description"]})
        return results
```

3. Below `embed_query_text`, the helper, with the chapter's output format, and in the CLI a `search` subcommand with a positional `query` and `--limit` (default 5):

```python
# In: a query and how many results.  Out: nothing; prints the results the chapter's way.
def semantic_search(query, limit):
    search = SemanticSearch()
    search.load_or_create_embeddings(load_movies())
    for i, result in enumerate(search.search(query, limit), start=1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description'][:100]}...")
        print()
```

```python
    search_parser = subparsers.add_parser("search", help="Search movies by meaning")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="How many results")
```

```python
        case "search":
            semantic_search(args.query, args.limit)
```

4. Run, then submit:

```bash
uv run cli/semantic_search_cli.py search "funny bear movies" --limit 5
uv run cli/semantic_search_cli.py search "space adventure" --limit 5
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* `Bear`, `The Great Bear` and `A Bear for Punishment` in the first list, and `Spaceflight IC-1`, `Adventureland` and `Odyssey 5` in the second, with scores around 0.4; those are the checks. Then run `uv run cli/keyword_search_cli.py bm25search "space adventure"` and look at the two lists side by side; that is chapter 6's subject, and P10 starts there.

**Lessons 15 and 16: hashing and vector databases.** Two readings on what replaces the loop in `search` when there are millions of vectors. A multiple-choice question and a Boots question.

**Step 3. `rag/semantic.py`, embedded once.**

Open the file and read `Semantic.__init__` before you run it: the fingerprint is one hash of every passage's text in order, the vectors go to `data/emb_passages.npy` next to `data/emb_passages.json`, and the saved vectors are used only when model, count and fingerprint all match. `normalize` scales every row to length 1, so `search` is one matrix multiply, `self.Vn @ query_vector`, and `np.argsort(-sims, kind="stable")` is descending with ties in file order, the P06 rule. Then:

```bash
uv add sentence-transformers
uv run python -m rag.semantic where does the story begin
```

*You should see* `embedding <your passage count> units with all-MiniLM-L6-v2`, a progress bar, then `matrix: (<your passage count>, 384)` and five results with scores between 0 and 1, best first. Run the same command again. *You should see* no `embedding` line and no bar: it loaded from disk. Paste the second run under `## Step 3` in `evidence/P08.md`.

*If it broke* with the bar running on both runs, `data/emb_passages.json` did not match; `cat` it and compare the count to `rag.corpus`. *If it broke* with scores far outside -1 to 1, `normalize` was skipped on one side.

**Step 4. Self-retrieval, on three passages.**

The A06 check, because it is the one that catches a vector sitting in the wrong row:

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

*You should see* each id map to itself at rank 1 with a score of `1.0` or within a rounding of it, and a second-best score clearly below. The first, the middle and the last, because an off-by-one only shows at an end. Paste the three lines under `## Step 4`. *If it broke* with a different id at rank 1, `V[i]` is not the vector of `units[i]`: delete `data/emb_passages.*` and rebuild, and if it persists, the passages changed between the build and the search.

**Step 5. The semantic column.**

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

Your predictions are already under `## Predictions` in `evidence/P07.md`, committed. Then:

```bash
uv run python -m rag.scoreboard
```

*You should see* the P06 columns unchanged, which proves the queries and needles did not move, and a third column. Expect at least one `paraphrase` query found that both keyword methods missed, because the vector compares meanings and not stems; and expect a `rare-name` query to be at risk, because a name that is one token to BM25 is a small part of a 384-number summary of a whole passage. Paste the table under `## Step 5` and check your predictions, one line each: right or wrong.

**Commit, push, sign off.** Fill `## Reflection` in `evidence/P08.md` (the questions are below) and run `uv run python scripts/check_evidence.py P08` until it says `all slots filled`. Then:

```bash
git add rag/scoreboard.py scratch/ evidence/
git commit -m "P08: semantic search over passages, cached and guarded; semantic column"
git push
```

The P07 pull request is already open; these commits land on it. Edit its body to name the query where semantic search surprised you most.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`evidence/P08.md` with every slot filled + `rag/scoreboard.py` with three methods + `scratch/p07-model.md` + `evidence/p08-bootdev-ch4.png`.

**Reflection Questions** (answer under `## Reflection` in `evidence/P08.md`)

1. Paste your three self-retrieval lines. Then search for one of those passages with one character changed (add a period at the end, or change one word) and paste the new rank-1 score. Say what the size of that change tells you about reading a `0.97` from A06's check.

   *How to get it:* `sem.search(ps[0]['text'] + '.', k=1)` in the same `python -c` shape as Step 4. The score drops from 1.0 by some amount; that amount is the scale of "nearly the same text", and it says where a 0.97 sits.

2. Paste your scoreboard and your P07 predictions. For one `paraphrase` query that semantic search found, paste the rank-1 passage's first 100 characters and say what it has in common with the query besides the words. For one query semantic search ranked worse than BM25, say what the embedding matched on instead.

   *How to get it:* `uv run python -m rag.semantic <the query>` prints the top five with scores; the passage at rank 1 is the one to read.

3. Paste the top semantic score for each of your five queries, marked hit or miss by whether the answer was in the top 3. Say whether there is a score that separates your hits from your misses, and what a miss scoring as high as a hit means for a system that wanted to say "no good answer".

   *How to get it:* for each query, `sem.search(q, 5)[0]['score']` and `first_hit` from `rag.scoreboard`; five lines. The chapter's own scores on its checks sit around 0.4 for good matches.
