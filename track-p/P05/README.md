# P05 · BM25, the Three Fixes

**Meetings:** D28 · **Points:** 10 pts

**Watch — none**

[Boot.dev, *Learn Retrieval Augmented Generation*](https://www.boot.dev/courses/learn-retrieval-augmented-generation) · chapter 3 Keyword Search, lessons 1 to 3, about 25 minutes.

**During the chapter**

**Write the two constants down.** Lessons 2 and 3 each introduce a tuning constant, `k1` and `b`. When the chapter gives you a value, write it into `scratch/p05-bm25.md` in the rag repo with one sentence, in the chapter's own words, saying what it controls. P06 puts both into `rag/keyword.py`.

**Predict a two-word ranking.** Before lesson 3, pick two words from your corpus, one common and one rare, and write under `## Prediction` in `evidence/P05.md` which word you think will decide the ranking when P06 searches for both together, and whether a long passage or a short one will win. Commit the file; I check the timestamps. P06 checks the prediction.

**Notes**

**BM25 is TF-IDF with three repairs, and the chapter builds them one per lesson.** A steadier idf (lesson 1), a term frequency that stops growing after a few occurrences (lesson 2), and a penalty for long documents (lesson 3). Each lesson is one new method on the `InvertedIndex` class from chapter 2 and one new CLI command; nothing from chapter 2 is removed.

**`tf 424 trapper` is still 4.** The raw count stays as it was. BM25's `get_bm25_tf` reads that count and turns it into a number between 0 and `k1 + 1`, which is 2.5 with the chapter's `k1`. That cap is the saturation: the tenth `trapper` adds almost nothing.

**Walkthrough — the chapter, and the constants for your corpus**

**Step 1. Merge, branch, log.**
Open the rag repo on GitHub and merge the P03 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch main && git pull
git switch -c dev/keyword
```

*If P03 is not approved yet:* `git switch dev/tfidf && git switch -c dev/keyword`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-track, not main
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Chapter 3, lessons 1 to 3, every lesson run locally before it is submitted
- [ ] scratch/p05-bm25.md: k1 and b with what each controls; my two-word prediction in evidence/P05.md, committed
- [ ] Step 3: the three idf lines on my corpus, in evidence/P05.md
- [ ] Fill every slot in evidence/P05.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. Chapter 3, lessons 1 to 3.**

In `~/version_control/rag-search-engine/`, lessons 1 to 3, each run locally before it is submitted. Stop after lesson 3; lesson 4 is P06. All three edit `cli/lib/keyword_search.py` and `cli/keyword_search_cli.py`.

**Lesson 1: a steadier idf.** Chapter 2's idf was `log((N + 1) / (df + 1))`. The chapter's BM25 idf compares the documents *without* the term to the documents *with* it, with `0.5` added to each count so neither can be zero, and `1` added to the ratio so the log is never negative:

```python
log((N - df + 0.5) / (df + 0.5) + 1)
```

1. In the class, below `get_idf`:

```python
    # In: one processed token.  Out: chapter 3's idf: log((N - df + 0.5) / (df + 0.5) + 1).
    def get_bm25_idf(self, term):
        N = len(self.docmap)
        df = len(self.get_documents(term))
        return math.log((N - df + 0.5) / (df + 0.5) + 1)
```

2. In the CLI, a `bm25idf` subcommand and its case. The course names a helper function `bm25_idf_command` for the loading and tokenizing; the case below does the same three lines in place, which is how the `idf` case already works:

```python
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
```

```python
        case "bm25idf":
            ix = InvertedIndex()
            ix.load()
            bm25idf = ix.get_bm25_idf(tokenize_single_term(args.term))
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
```

3. Run, then submit:

```bash
uv run cli/keyword_search_cli.py bm25idf grizzly
uv run cli/keyword_search_cli.py bm25idf actor
uv run cli/keyword_search_cli.py bm25idf love
uv run cli/keyword_search_cli.py idf grizzly
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* `5.55`, `3.29` and `0.95`; those are the checks. The fourth line is chapter 2's idf for the same word, `5.52`: the two formulas agree to a few hundredths on a rare word and the BM25 one is slightly larger. *If it broke* with `grizzly` scoring like a term in no document, the lookup skipped `tokenize_single_term`; the index holds `grizzli`.

**Lesson 2: term frequency saturation.** A document that says `bear` four times is not four times as relevant as one that says it once. The chapter's saturated tf, with `k1 = 1.5`:

```python
(tf * (k1 + 1)) / (tf + k1)
```

Put the chapter's table in `scratch/p05-bm25.md`: 1 occurrence gives 1.0, 2 gives 1.4, 5 gives 1.9, 10 gives 2.2, 20 gives 2.3. The number can never reach `k1 + 1`.

1. At the top of `cli/lib/keyword_search.py`, next to `PUNCT`, the constant. The course suggests a separate module for constants; one file is fine for this project:

```python
BM25_K1 = 1.5     # chapter 3 lesson 2: how fast term frequency saturates
```

2. In the class, below `get_bm25_idf`:

```python
    # In: a document id, one processed token and k1.  Out: the saturated term frequency, between 0 and k1 + 1.
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1):
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1)
```

3. In the CLI: `BM25_K1` joins the import, a `bm25tf` subcommand with an optional third argument, and its case. `nargs="?"` makes the argument optional, with `default` used when it is left out:

```python
from lib.keyword_search import BM25_K1, InvertedIndex, build_command, search_titles, tokenize_single_term
```

```python
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")
```

```python
        case "bm25tf":
            ix = InvertedIndex()
            ix.load()
            bm25tf = ix.get_bm25_tf(args.doc_id, tokenize_single_term(args.term), args.k1)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
```

4. Run, then submit:

```bash
uv run cli/keyword_search_cli.py bm25tf 1 anbuselvan
uv run cli/keyword_search_cli.py tf 1 anbuselvan
uv run cli/keyword_search_cli.py bm25tf 1 maya
uv run cli/keyword_search_cli.py bm25tf 1 police
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* `2.31`, then the raw count the chapter 2 command still gives, then `2.17` and `2.00`; the three BM25 numbers are the checks. Put the raw count next to its saturated value in `scratch/p05-bm25.md`. *If it broke* with a number larger than 2.5, the formula has `k1` in the wrong place; the ceiling is `k1 + 1`.

**Lesson 3: document length normalization.** A long document has more words, so it has more matches, so it scores higher for being long. The chapter scales the saturation by the document's length against the average, with `b = 0.75` deciding how much length matters:

```python
length_norm = 1 - b + b * (doc_length / avg_doc_length)
tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
```

An average-length document has `length_norm` 1 and is scored as in lesson 2; a longer one is pushed down, a shorter one up. `b = 0` switches the whole thing off.

1. The constant, under `BM25_K1`:

```python
BM25_B = 0.75     # chapter 3 lesson 3: how much document length counts against a document
```

2. The index needs each document's length and a fourth cache file. In `__init__`:

```python
        self.doc_lengths = {}       # document id -> how many tokens the document had
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
```

The course keeps a `CACHE_DIR` constant for the folder name; add `CACHE_DIR = "cache"` next to the other constants.

3. In `__add_document`, count the tokens before the loop, so the length is stored once per document:

```python
    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)
            self.term_frequencies[doc_id][token] += 1
```

4. A fourth file in `save` and `load`, the same shape as the other three:

```python
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)
```

```python
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)
```

5. The average, as a private helper, and `get_bm25_tf` with the normalization in it. The course asks for the empty-index case to return `0.0`:

```python
    # In: nothing.  Out: the average document length in tokens, 0.0 if there are no documents.
    def __get_avg_doc_length(self):
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    # In: a document id, one processed token, k1 and b.  Out: the saturated, length-normalized term frequency.
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        length_norm = 1 - b + b * (self.doc_lengths.get(doc_id, 0) / self.__get_avg_doc_length())
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
```

6. In the CLI: `BM25_B` joins the import, a fourth optional argument, and the case passes it through:

```python
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter")
```

```python
            bm25tf = ix.get_bm25_tf(args.doc_id, tokenize_single_term(args.term), args.k1, args.b)
```

7. Rebuild, so the lengths are on disk, then run and submit:

```bash
uv run cli/keyword_search_cli.py build
uv run cli/keyword_search_cli.py bm25tf 1 anbuselvan
uv run cli/keyword_search_cli.py bm25tf 1 police
uv run cli/keyword_search_cli.py bm25tf 1 maya
uv run cli/keyword_search_cli.py bm25tf 1 cheese
uv run cli/keyword_search_cli.py bm25tf 1 anbuselvan 1.5 0
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* `2.35`, `2.09`, `2.24` and `0.00`; those are the checks. Every number went up a little from lesson 2, which tells you document 1 is shorter than average. The last line, with `b` set to `0`, gives lesson 2's `2.31` back: that is what switching normalization off means. *If it broke* with `FileNotFoundError: cache/doc_lengths.pkl`, you ran `bm25tf` before rebuilding. *If it broke* with `ZeroDivisionError`, `__get_avg_doc_length` returned `0.0` because `doc_lengths` is empty: `load` does not read the fourth file, or `__add_document` never stored the count.

**Step 3. The same idf on your corpus.**

The P06 scorer uses the chapter's BM25 idf on your passages. Before it exists, see the two idfs side by side on three words: the two you chose for the prediction, and one you expect to be in every other passage:

```bash
uv run python -m rag.tfidf idf <your rare word> <your common word> <a very common word>
```

*You should see* one line per word with its df and its chapter 2 idf; this command keeps stopwords in, so the very common word gets a number. On Homer, `cyclops` is in 28 passages at idf 4.29, `ship` in 559 at 1.33, and `the` in 2076 of the 2118 at 0.02. Paste the lines under `## Step 3` in `evidence/P05.md`, and under them work out the BM25 idf of your rare word and your very common word by hand from the formula, with your `N` and their df: on Homer, `log((2118 - 28 + 0.5) / (28 + 0.5) + 1)`, which is 4.31, and `log((2118 - 2076 + 0.5) / (2076 + 0.5) + 1)`, which is 0.02.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/P05.md` (the questions are below) and run `uv run python scripts/check_evidence.py P05` until it says `all slots filled`. Then:

```bash
git add scratch/ evidence/
git commit -m "P05: chapter 3 started, k1 and b written down, two-word prediction"
git push -u origin dev/keyword
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. It stays open through P06.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`evidence/P05.md` with every slot filled + `scratch/p05-bm25.md`.

**Reflection Questions** (answer under `## Reflection` in `evidence/P05.md`)

1. Paste your `bm25tf 1 anbuselvan` line from lesson 2 and the `tf 1 anbuselvan` line next to it. Say what number the saturated tf can never reach with the chapter's `k1`, and what that does to a document that repeats a word fifty times.

   *How to get it:* put a huge `tf` into `(tf * (k1 + 1)) / (tf + k1)`; the `k1` in the denominator stops mattering and the fraction approaches `k1 + 1`. The chapter's table shows 20 occurrences already at 2.3.

2. Paste your lesson 3 `bm25tf 1 anbuselvan` line, the lesson 2 one, and the `1.5 0` one. Say from the formula why the first is larger than the second, and why the third equals the second.

   *How to get it:* with `b = 0.75`, a document shorter than average has `length_norm` below 1, so `k1 * length_norm` in the denominator is smaller and the fraction is larger. With `b = 0`, `length_norm` is exactly 1 and the formula is lesson 2's.

3. Paste your three Step 3 lines and your two hand-worked BM25 idfs. Say what happens to a word that is in every single document under each formula, and why the chapter calls its version more stable.

   *How to get it:* with `df = N`, chapter 2 gives `log((N + 1) / (N + 1))`, which is exactly 0, so the word counts for nothing; BM25 gives `log(0.5 / (N + 0.5) + 1)`, a small positive number, so it still counts a little and the log never goes negative. The lesson's three reasons are the first paragraph of its page.
