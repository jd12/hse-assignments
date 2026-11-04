# P04 · The Index on Your Corpus, and Its Own Stopwords

**Meetings:** D27 · **Points:** 10 pts

**Watch — none**

[Boot.dev, *Learn Retrieval Augmented Generation*](https://www.boot.dev/courses/learn-retrieval-augmented-generation) · chapter 2 TF-IDF, lessons 5 and 6, about 15 minutes.

**Notes**

**`rag/tfidf.py` shipped with the chapter's idf and a different tf.** The line marked `# CHAPTER FORMULA` is lesson 5's formula with the names changed: `self.N` for the document count, `self.df[term]` for the documents containing the term, the same two `+ 1`s. Check it against `scratch/p03-formula.md` anyway; a formula you did not compare is a formula you trust on my word. The shipped `tf` divides the count by the passage's length, and lesson 4's tf is the raw count, so Step 3 changes that one line. Nothing printed changes when you do, and Reflection 1 asks why.

**A generic stopword list does not know your corpus.** In a corpus like the Odyssey, the hero's name is in a large share of the passages, so it separates few of them from each other, and no English stopword list includes it. Your corpus has words like that too. The extension finds them, measures what removing them does, and names what it cost.

**Walkthrough — your index, read**

**Step 1. Branch, log.**
Same branch as P03:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch dev/tfidf && git pull
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Chapter 2, lessons 5 and 6, every lesson run locally before it is submitted
- [ ] Step 3: the CHAPTER FORMULA line checked, tf changed to the raw count, rag.tfidf run, my two P03 words checked
- [ ] Extension: prediction in evidence/P04.md and committed, corpus stopwords, rag/corpus_notes.md
- [ ] Fill every slot in evidence/P04.md (check_evidence.py: all slots filled); push, sign off
```

**Step 2. Finish chapter 2: lessons 5 and 6.**

In `~/version_control/rag-search-engine/`, the last two lessons, each run locally before it is submitted. Both add one command to `cli/keyword_search_cli.py` and one method to the class in `cli/lib/keyword_search.py`. When the chapter page shows every lesson complete, save it as `evidence/p04-bootdev-ch2.png` in the rag repo.

**Lesson 5: inverse document frequency.** `actor` is in a large share of movie descriptions, so a match on it says little; `grizzly` is in a handful, so a match on it says a lot. idf scores a term by how few documents contain it. The lesson's formula, which goes into `scratch/p03-formula.md` under your term frequency line, character for character:

```python
math.log((total_doc_count + 1) / (term_match_doc_count + 1))
```

1. At the top of `cli/lib/keyword_search.py`:

```python
import math
```

2. In the class, below `get_tf`. The index already knows both counts: `docmap` has one entry per document, and `get_documents` returns the documents that contain the token:

```python
    # In: one processed token.  Out: its inverse document frequency, by the chapter's formula.
    def get_idf(self, term):
        return math.log((len(self.docmap) + 1) / (len(self.get_documents(term)) + 1))
```

3. In the CLI, an `idf` subcommand that takes a term, and its case:

```python
    idf_parser = subparsers.add_parser("idf", help="Inverse document frequency of a term")
    idf_parser.add_argument("term", type=str, help="One word")
```

```python
        case "idf":
            ix = InvertedIndex()
            ix.load()
            idf = ix.get_idf(tokenize_single_term(args.term))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
```

4. Run, then submit:

```bash
uv run cli/keyword_search_cli.py idf grizzly
uv run cli/keyword_search_cli.py idf actor
uv run cli/keyword_search_cli.py idf man
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* `5.52`, `3.29` and `0.76`; those are the checks, and the order is the point: the rarer the term, the larger the number. *If it broke* with `grizzly` scoring like a term in no document at all, the lookup skipped `tokenize_single_term`: the index holds the stem `grizzli`, and `grizzly` is not a key in it. *If it broke* with values close to the checks but not equal, your `data/stopwords.txt` or `data/movies.json` differs from the course's; the lesson page says so, and the fix is to re-copy the file.

**Lesson 6: tf-idf.** The product of the two numbers: a term scores high in a document when it is frequent there (tf) and rare across the collection (idf). The lesson's example has `cyborg` at idf 3.9 and `bear` at 0.05, so one `cyborg` outweighs two `bear`s.

1. In the CLI, a `tfidf` subcommand that takes a document id and a term, and its case. No new method: `get_tf` and `get_idf` already exist, and the score is their product:

```python
    tfidf_parser = subparsers.add_parser("tfidf", help="TF-IDF of a term in one document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document id")
    tfidf_parser.add_argument("term", type=str, help="One word")
```

```python
        case "tfidf":
            ix = InvertedIndex()
            ix.load()
            token = tokenize_single_term(args.term)
            tf_idf = ix.get_tf(args.doc_id, token) * ix.get_idf(token)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
```

2. Run, then submit:

```bash
uv run cli/keyword_search_cli.py tfidf 424 trapper
uv run cli/keyword_search_cli.py tfidf 424 push
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* `24.13` and `2.14`; those are the checks. Lesson 4 said `trapper` is in document 424 four times, so its idf is 24.13 divided by 4, about 6.03, and `uv run cli/keyword_search_cli.py idf trapper` confirms it. *If it broke* with `0.00` for `trapper`, `get_tf` returned 0: the id came in as a string, or the index on disk predates lesson 4 and has no `term_frequencies.pkl`; rebuild.

That is the chapter's whole scorer: a raw count times a log of a ratio of document counts. Step 3 runs the same two formulas over your corpus, where the documents are your P02 passages.

**Step 3. The index on your corpus.**

Open `rag/tfidf.py` and read the class, down to `top_terms`: a `TfidfIndex` holds one `Counter` of terms per passage, the passage lengths, and `df`, how many passages each term is in; `tf`, `idf` and `tfidf` are three one-line functions; `top_terms` sorts a passage's terms by tf-idf. The code under `if __name__` prints the report and you can skip it. Check the `# CHAPTER FORMULA` line against `scratch/p03-formula.md`. Then make `tf` the chapter's raw count, so the one-line body of `tf` reads:

```python
        return self.docs[i][term]
```

Then:

```bash
uv run python -m rag.tfidf
```

*You should see* seven lines. On Homer:

```text
passages: 2118  vocabulary: 7374
terms in exactly one passage: 2535 (34%)
idf('the') with stopwords kept: 0.02  df: 2076
largest possible idf: 6.9656
p0000 ['book', 'prefac', 'edit', 'odyssey', 'prose']
p1059 ['183', 'carp', 'er', '182', '273']
p2117 ['barrow', 'togeth', 'heap', 'assembl', 'hurriedli']
```

Whatever your corpus: a large share of the vocabulary is in exactly one passage, often a third to over half, because real vocabularies are mostly words that appear once. `idf('the')` is tiny, under a tenth of the largest possible idf: `the` is in most passages, so it carries almost no information about which one you want, and its tf-idf in any passage is near zero even though its count is the highest. The largest possible idf is the idf of every term in exactly one passage, your rarest words. The three `top_terms` lists are names, rare nouns and distinctive words from that passage, no function words. A list of numbers and fragments, like Homer's `p1059`, means you landed in a footnote block or a table of contents; that is a real artifact of your file, and Reflection 3 wants it. Paste all seven lines under `## Step 3` in `evidence/P04.md`. Then check your two P03 predictions, with your two words in place of these:

```bash
uv run python -m rag.tfidf idf ulysses the
```

*You should see* one line per word with its stem, its df and its idf, stopwords kept so a stopword guess still counts. On Homer: `ulysses: stem 'ulyss'  df 506  idf 1.4302` and `the: stem 'the'  df 2076  idf 0.0200`. A predicted word that is in no passage prints `df 0` and `not in any passage`. Paste the two lines under your predictions and say whether they held.

*If it broke:* `idf('the')` larger than the largest possible idf means `raw.df["the"]` is 0: the `stopwords=frozenset()` argument never reached `preprocess`. A typo in a term gives the same symptom and no error, because a `Counter` returns 0 for anything it has not seen.

**Extension — your corpus's own stopwords (ASSIGNED)**

Before you run anything, write under `## Extension` in `evidence/P04.md` three words you expect to be in at least a fifth of your passages. Commit that file. I will check your commit timestamps. Then:

```bash
uv run python -m rag.tfidf stopwords
```

*You should see* the seven lines again, then how many terms sit in at least 10%, 20% and 30% of your passages, the 20% list, and three before-and-after lines with those stems dropped. On Homer:

```text
df >= 10%: 108 terms
df >= 20%: 22 terms
df >= 30%: 3 terms
corpus stopwords: ['achaean', 'come', 'go', 'god', 'hand', 'jove', 'man', 'men', 'now', 'one', 'other', 'over', 'own', 'said', 'ship', 'some', 'son', 'take', 'trojan', 'ulyss', 'upon', 'went']
vocabulary before/after: 7374 7352
term occurrences removed: 11.8%
top-5 slots changed: 38 of 10590
```

The list is stems, which is why `drop_stems` in `preprocess` filters after stemming: `ulyss`, not `ulysses`. Filter it against raw tokens and `ulyss` never matches anything, because no word in the text is spelled that way. The script wrote the list to `rag/corpus_stopwords.txt`. Then write `rag/corpus_notes.md`:

1. The three threshold counts and which threshold you kept. 20% is a starting point, not an answer.
2. Your list, with the words that carry meaning marked.
3. Vocabulary before and after, the share of term occurrences removed, and top-5 slots changed: the three lines, pasted.
4. **Where it got worse.** Pick one word on your list that carries meaning for a question someone would ask about your corpus, write that question, and say what removing the word does to it. On Homer, `jove` is on the list, and "what did Jove do to help Ulysses get home" comes out as `['help', 'get', 'home']`: the god and the man are both gone from the question. A list with nothing worth keeping on it means your threshold is too low.

Paste the output under `## Extension` in `evidence/P04.md`. Commit the list and the notes separately from the code, so the decision is its own commit. The `tf` change goes first:

```bash
git add rag/tfidf.py
git commit -m "P04: TF-IDF index with the chapter's tf"
git add rag/corpus_stopwords.txt rag/corpus_notes.md
git commit -m "P04: corpus stopwords and what they changed"
```

**Commit, push, sign off.** Fill `## Reflection` in `evidence/P04.md` (the questions are below) and run `uv run python scripts/check_evidence.py P04` until it says `all slots filled`. Then:

```bash
git add evidence/
git commit -m "P04: evidence"
git push
```

The P03 pull request is already open; these commits land on it. Edit its body to name the word on your list you were least sure about.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`evidence/P04.md` with every slot filled + `rag/tfidf.py` with the chapter's tf + `rag/corpus_stopwords.txt` + `rag/corpus_notes.md` + `evidence/p04-bootdev-ch2.png`.

**Reflection Questions** (answer under `## Reflection` in `evidence/P04.md`)

1. Paste your `idf('the')` line and your `largest possible idf` line. Say, from the formula, why the first is near zero and the second is the biggest number the formula can give on your corpus. Then say why changing `tf` from a count divided by length to the raw count moved nothing in the seven lines.

   *How to get it:* `df` for `the` is nearly `N`, so the formula's fraction is nearly 1 and its log is nearly 0; a term in exactly one passage has `df = 1`, the smallest df there is, so its idf is the largest. For the last part, look at what `top_terms` sorts: every term in one passage is divided by that passage's one length, and dividing every score by the same number does not change their order.

2. For the one word you marked "meaning" in `corpus_notes.md`: paste the question you wrote, and `preprocess(question)` with and without the list.

   *How to get it:* `uv run python -c "from rag.preprocess import preprocess; q = 'YOUR QUESTION'; s = set(open('rag/corpus_stopwords.txt').read().split()); print(preprocess(q)); print(preprocess(q, drop_stems=s))"`.

3. Pick three of your one-passage terms. For each, say whether it is a real word, a name, a typo, or an artifact of your preprocessing, and what `strip_punctuation` or the stemmer did to produce the artifacts.

   *How to get it:* `uv run python -c "from rag.corpus import load_passages; from rag.tfidf import TfidfIndex; ix = TfidfIndex(load_passages()); print(sorted(t for t, n in ix.df.items() if n == 1 and not t[0].isdigit())[:40])"` prints forty of them, alphabetically, skipping the ones that are numbers (footnote markers and years, on a Gutenberg file). On Homer the forty start `['aba', 'abarbarea', 'abet', 'abian', 'ablaz', 'ableru', ...]`: a name, a stem the stemmer cut oddly, a real word. `grep -n -i` one odd one in `data/corpus.txt` to see the text it came from.
