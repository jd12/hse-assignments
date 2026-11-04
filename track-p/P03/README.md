# P03 · TF-IDF, Chapter 2

**Meetings:** D26 · **Points:** 10 pts

**Watch — none**

[Boot.dev, *Learn Retrieval Augmented Generation*](https://www.boot.dev/courses/learn-retrieval-augmented-generation) · chapter 2 TF-IDF, lessons 1 to 4, about 25 minutes.

**During the chapter**

**Copy the chapter's definitions exactly.** Lesson 4 defines term frequency. Copy its definition into `scratch/p03-formula.md` in the rag repo character for character: what is counted, and whether the count is divided by anything. Lesson 5, in P04, gives the idf formula, and it goes under the same heading, every `+ 1` included. P04 has a line in `rag/tfidf.py` marked `# CHAPTER FORMULA`, and that file is checked against this one.

**Notes**

**An inverted index is the chapter's real subject.** TF-IDF is a number; the index is what makes it fast to find every document a term is in without reading every document. Reflection 1 asks what it maps from and to.

**Every lookup goes through `tokenize_text`.** The index holds processed tokens, `grizzli` and not `grizzly`, so a word typed at the command line has to go through the same pipeline before it is looked up. Lesson 4 adds a helper for that, and Reflection 2 is about what it does to a stop word.

**Walkthrough — the chapter, and a prediction for your corpus**

**Step 1. Merge, branch, log.**
Open the rag repo on GitHub and merge the P02 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch main && git pull
git switch -c dev/tfidf
```

*If P02 is not approved yet:* `git switch dev/preprocess && git switch -c dev/tfidf`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-track, not main
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Chapter 2, lessons 1 to 4, every lesson run locally before it is submitted
- [ ] scratch/p03-formula.md: the chapter's term frequency, character for character
- [ ] Step 4: two predictions for my corpus, in evidence/P03.md
- [ ] Fill every slot in evidence/P03.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. The chapter's term frequency, character for character.**

The moment lesson 4 states it, before the next lesson. `scratch/p03-formula.md` holds the chapter's definition of term frequency as the lesson shows it: what it counts, and whether it divides. Paste it under `## Step 2` in `evidence/P03.md`. P04 adds the idf formula under it.

**Step 3. Chapter 2, lessons 1 to 4.**

In `~/version_control/rag-search-engine/`, lessons 1 to 4, each run locally before it is submitted. Stop after lesson 4; lessons 5 and 6 are P04. Save the chapter page as `evidence/p03-bootdev.png`. Every lesson below edits `cli/lib/keyword_search.py` and `cli/keyword_search_cli.py`, the two files chapter 1 left you, and every lookup goes through the `tokenize_text` you finished in P02.

**Lesson 1: the inverted index.** Chapter 1's search read every title on every query. This lesson reads every movie once, builds a table from each token to the ids of the movies that contain it, and saves the table to disk. The course wants a class, `InvertedIndex`, in `cli/lib/keyword_search.py`, with a `build` command that uses it.

1. At the top of `cli/lib/keyword_search.py`, two more imports:

```python
import os
import pickle
```

2. Below `tokenize_text`, the class:

```python
class InvertedIndex:
    def __init__(self):
        self.index = {}    # token -> set of document ids
        self.docmap = {}   # document id -> the movie dict

    # In: a document id and its text.  Out: nothing; every token of the text now maps to that id.
    def __add_document(self, doc_id, text):
        for token in tokenize_text(text):
            self.index.setdefault(token, set()).add(doc_id)

    # In: one processed token.  Out: the ids of the documents that contain it, ascending.
    def get_documents(self, term):
        return sorted(self.index.get(term, set()))

    # In: nothing.  Out: nothing; the index and the docmap now hold every movie.
    def build(self):
        for movie in load_movies():
            self.docmap[movie["id"]] = movie
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")

    # In: nothing.  Out: nothing; cache/index.pkl and cache/docmap.pkl now hold the two dictionaries.
    def save(self):
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
```

`setdefault(token, set())` returns the set already stored under that token, or stores an empty one first; either way `.add(doc_id)` goes into the right set. A set, not a list, so a token that appears twice in one description does not list the movie twice. `get_documents` uses `.get(term, set())` so an unknown token gives an empty list instead of a `KeyError`; lesson 2 depends on that. `pickle.dump` writes a Python object to a file as bytes, which is why the files are opened with `"wb"`.

3. Below the class, the function the CLI will call. The lesson's check is that the first document for `merida` is 4651, *Brave*:

```python
# In: nothing.  Out: nothing; cache/ now holds the index built from data/movies.json.
def build_command():
    ix = InvertedIndex()
    ix.build()
    ix.save()
    docs = ix.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")
```

4. In `cli/keyword_search_cli.py`, add `build_command` to the import, a `build` subcommand next to the `search` one, and its case:

```python
from lib.keyword_search import search_titles, build_command
```

```python
    subparsers.add_parser("build", help="Build the inverted index and save it to cache/")
```

```python
        case "build":
            build_command()
```

5. Run, then submit:

```bash
uv run cli/keyword_search_cli.py build
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* `First document for token 'merida' = 4651` after a few seconds, and a `cache/` folder with two `.pkl` files in it; the checks are that line and those two files. Add `/cache/` to `.gitignore`, the same way `data/` is. *If it broke* with `IndexError: list index out of range` on the print, `get_documents("merida")` came back empty: `build` did not go through `tokenize_text`, or it indexed the title alone. *If it broke* with `FileNotFoundError: cache/index.pkl`, `os.makedirs` is missing or ran after the `open`.

**Lesson 2: use the index.** The index is on disk, so a search loads it instead of rebuilding it, and looks each query token up instead of reading every movie.

1. Delete the two `merida` lines from `build_command`; they were a check, not a feature.

2. In the class, below `save`, the reverse of it. `open` on a missing file raises `FileNotFoundError` on its own, which is the error the lesson asks for:

```python
    # In: nothing.  Out: nothing; the index and the docmap are back from cache/.
    def load(self):
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
```

3. Replace the whole of `search_titles`. The loop over every movie goes; each query token is looked up, and the search stops at five:

```python
# In: a query string.  Out: up to five movie dicts that share a token with it, found through the index.
def search_titles(query):
    ix = InvertedIndex()
    try:
        ix.load()
    except FileNotFoundError:
        print("No index in cache/. Run: uv run cli/keyword_search_cli.py build")
        return []
    results = []
    seen = set()
    for token in tokenize_text(query):
        for doc_id in ix.get_documents(token):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            results.append(ix.docmap[doc_id])
            if len(results) == 5:
                return results
    return results
```

`seen` keeps a movie that matches two query tokens from appearing twice. The order is the query's token order, then ascending id within a token, which is what `get_documents` sorted for.

4. In the CLI's `search` case, print the id with the title:

```python
                print(f"{i}. {movie['title']} ({movie['id']})")
```

5. Run, then submit:

```bash
uv run cli/keyword_search_cli.py search "brave"
uv run cli/keyword_search_cli.py search "nonsensetoken assault"
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* `Madrasapattinam` in the first list, found through its description, which the chapter 1 search never read, and `Klansman` in the second: `nonsensetoken` is in no movie, so its lookup is an empty list and the loop moves on to `assault`. Those two titles are the checks. *If it broke* with `KeyError: 'nonsensetoken'`, `get_documents` indexes `self.index[term]` directly instead of using `.get`.

**Lesson 3: boolean search.** A reading, no code: AND is set intersection, OR is set union, NOT is set difference, each on the sets of ids the index already holds. The lesson ends with a Boots question; answer it in the chat box until it is satisfied, then move on.

**Lesson 4: term frequency.** How many times a token is in one document. The index can say *which* movies contain `trapper`; this lesson records *how many times* each one does, in a `Counter` per movie, and adds a `tf` command to read it back.

1. At the top of `cli/lib/keyword_search.py`:

```python
from collections import Counter
```

2. In the class, a third dictionary, filled in `__add_document` as each token goes by:

```python
    def __init__(self):
        self.index = {}             # token -> set of document ids
        self.docmap = {}            # document id -> the movie dict
        self.term_frequencies = {}  # document id -> Counter of token -> count

    # In: a document id and its text.  Out: nothing; every token of the text now maps to that id, and its count is stored.
    def __add_document(self, doc_id, text):
        self.term_frequencies[doc_id] = Counter()
        for token in tokenize_text(text):
            self.index.setdefault(token, set()).add(doc_id)
            self.term_frequencies[doc_id][token] += 1
```

A `Counter` is a dict whose missing keys read as 0, so `+= 1` works on a token the first time it is seen.

3. A method to read a count back, and a third file in `save` and `load`:

```python
    # In: a document id and one processed token.  Out: how many times the token is in that document, 0 if it is not.
    def get_tf(self, doc_id, term):
        return self.term_frequencies.get(doc_id, Counter())[term]
```

```python
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)
```

```python
        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)
```

4. Below `tokenize_text`, the helper the lesson asks for. The index holds processed tokens, `grizzli` and not `grizzly`, so a word typed at the command line has to go through the same pipeline before it is looked up:

```python
# In: one word.  Out: its single processed token, or an exception if processing did not leave exactly one.
def tokenize_single_term(term):
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError(f"expected one token from {term!r}, got {tokens}")
    return tokens[0]
```

5. In the CLI: `InvertedIndex` and `tokenize_single_term` join the import, a `tf` subcommand takes a document id and a term, and its case loads the index and prints the count:

```python
from lib.keyword_search import search_titles, build_command, InvertedIndex, tokenize_single_term
```

```python
    tf_parser = subparsers.add_parser("tf", help="Term frequency of a term in one document")
    tf_parser.add_argument("doc_id", type=int, help="Document id")
    tf_parser.add_argument("term", type=str, help="One word")
```

```python
        case "tf":
            ix = InvertedIndex()
            ix.load()
            print(ix.get_tf(args.doc_id, tokenize_single_term(args.term)))
```

6. Rebuild, so the counts are on disk, then run and submit:

```bash
uv run cli/keyword_search_cli.py build
uv run cli/keyword_search_cli.py tf 424 bear
uv run cli/keyword_search_cli.py tf 424 trapper
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* `1` and then `4`; those are the checks. *If it broke* with `FileNotFoundError: cache/term_frequencies.pkl`, you ran `tf` before rebuilding. *If it broke* with `0` for `trapper`, the id arrived as the string `"424"` and the Counter is keyed by the number: `type=int` is missing from the argument.

**Step 4. Two predictions.**

P04 builds a tf-idf index over your corpus, and its lesson 5 scores a term by how few passages contain it. Before any of that runs, write under `## Step 4` in `evidence/P03.md` the word you expect to be in the fewest of your passages and the word you expect to be in the most, one sentence each. Fewest passages, not fewest occurrences: a word used two hundred times in one passage is in one passage.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/P03.md` (the questions are below) and run `uv run python scripts/check_evidence.py P03` until it says `all slots filled`. Then:

```bash
git add scratch/ evidence/
git commit -m "P03: chapter 2 started, the chapter's tf copied, two predictions"
git push -u origin dev/tfidf
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the lesson you stopped on. It stays open through P04.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`evidence/P03.md` with every slot filled + `scratch/p03-formula.md` + `evidence/p03-bootdev.png`.

**Reflection Questions** (answer under `## Reflection` in `evidence/P03.md`)

1. The chapter's `InvertedIndex` keeps two dictionaries, `index` and `docmap`. Say what each maps from and to, and why a search cannot afford to scan every document for every query.

   *How to get it:* lesson 1 shows the shape: `index` has one kind of key and sets of another thing as values, and `docmap` goes the other way. For the second half, count your passages from P02 and imagine a query of three terms against them, once with the index and once without.

2. `tf 424 trapper` prints `4` and `tf 424 the` raises an exception. Say what `tokenize_single_term` did with each word, and why the count has to be looked up by the processed token and not by the word you typed.

   *How to get it:* run both, and `tf 424 Trappers`. The index was built from `tokenize_text`, so its keys are lowercased stems with the stop words gone; `the` comes out of the pipeline as nothing at all, and the helper refuses anything that is not exactly one token.

3. `docmap` stores each movie once, under its id, and `index` stores ids, not movies. Say what the index would cost if every token's set held the movie dicts themselves, using one movie with twenty tokens as the example, and what it would cost to change one movie's title.

   *How to get it:* count the copies. One movie with twenty tokens would be in twenty sets. Then ask how many of those copies a title change would have to find.
