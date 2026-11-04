# P06 · BM25 on Your Corpus

**Meetings:** D29 · **Points:** 10 pts

**Watch — none**

[Boot.dev, *Learn Retrieval Augmented Generation*](https://www.boot.dev/courses/learn-retrieval-augmented-generation) · chapter 3 Keyword Search, lesson 4, about 10 minutes.

**Notes**

**The query goes through the same `preprocess` as the passages.** Lowercase the passages and not the query, and every capitalized query word scores zero, with no error anywhere. The walkthrough's scorer calls one function for both sides with the same arguments, stored on the index.

**A query made of stopwords is an empty list of terms.** Every passage scores zero. The scorer returns nothing rather than five arbitrary passages at `0.000`.

**Ties break the same way every time.** Two passages with the same score come back in a fixed order, by passage position, or a result that is right today is wrong after an unrelated change. `sorted` on `(-score, i)` does that; `sorted(scores)` on its own is ascending and hands you the worst passages first.

**Your five queries are written before anything scores them.** The extension's scoreboard follows the queries through P08 and P11. A query written after seeing a method's results is a query for that method. I check the commit timestamps.

**Walkthrough — tf-idf against BM25 on your passages**

**Step 1. Branch, log.**
Same branch as P05:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch dev/keyword && git pull
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Chapter 3, lesson 4, run locally before it is submitted; the chapter page saved
- [ ] rag/keyword.py: three checks (Steps 3 to 5), my P05 prediction checked
- [ ] Extension: rag/queries.json written and committed BEFORE the scoreboard runs; rag/scoreboard.py run
- [ ] Fill every slot in evidence/P06.md (check_evidence.py: all slots filled); push, sign off
```

**Step 2. Finish chapter 3: lesson 4.**

In `~/version_control/rag-search-engine/`. When the chapter page shows every lesson complete, save it as `evidence/p06-bootdev-ch3.png` in the rag repo.

**Lesson 4: BM25 search.** The three pieces become one score, `bm25_tf * bm25_idf`, summed over the query's tokens, for every document, sorted.

1. In the class, below `get_bm25_tf`:

```python
    # In: a document id and one processed token.  Out: the BM25 score of that term in that document.
    def bm25(self, doc_id, term):
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    # In: a query string and how many results.  Out: up to limit (movie, score) pairs, highest score first.
    def bm25_search(self, query, limit=5):
        tokens = tokenize_text(query)
        scores = {}
        for doc_id in self.docmap:
            total = 0.0
            for token in tokens:
                total += self.bm25(doc_id, token)
            scores[doc_id] = total
        ranked = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)
        return [(self.docmap[doc_id], scores[doc_id]) for doc_id in ranked[:limit]]
```

`sorted(scores, key=...)` sorts the dictionary's keys, the document ids, by the score each one maps to; `reverse=True` puts the highest first. The loop visits every document, which is the chapter 1 habit the index was built to avoid; the course accepts it here. A document with none of the query's tokens scores 0 in every term and sorts to the bottom.

2. In the CLI, a `bm25search` subcommand with an optional `--limit`, and its case. The course's format puts the id in parentheses and the score to two places:

```python
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5, help="How many results")
```

```python
        case "bm25search":
            ix = InvertedIndex()
            ix.load()
            for i, (movie, score) in enumerate(ix.bm25_search(args.query, args.limit), start=1):
                print(f"{i}. ({movie['id']}) {movie['title']} - Score: {score:.2f}")
```

`for i, (movie, score) in enumerate(...)` unpacks two things at once: the counter from `enumerate`, and the `(movie, score)` pair inside each result.

3. Run, then submit:

```bash
uv run cli/keyword_search_cli.py bm25search "love story"
uv run cli/keyword_search_cli.py bm25search "animated family"
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* `The Inner Life of Martin Frost` at 4.60, `Definitely, Maybe` at 4.51 and `Cama de Gato` at 4.40 for the first, and `Gakuen Alice` at 7.35, `Day of the Animals` at 7.13 and `Fantastic Mr. Fox` at 6.92 for the second; those are the checks. *If it broke* with the right titles and scores a few hundredths off, `get_bm25_tf` is being called without `b`, or your `data/stopwords.txt` differs from the course's.

**Step 3. `rag/keyword.py`.**

Two scorers over your P04 index, both reading the same raw counts: `tfidf_score` adds up chapter 2's tf-idf for each query term, and `bm25` is chapter 3's, with the three fixes in the chapter's order. The constants at the top are the chapter's; change them if yours differ.

Open the file and read it down to `search`: `bm25_idf` is lesson 1's formula on `self.ix.df` and `self.ix.N`, `bm25_tf` is lessons 2 and 3's on the raw count and the passage's length, and `bm25` adds `bm25_tf * bm25_idf` over the terms. `search` runs the query through `preprocess` with the index's own arguments, scores every passage, drops the zeros, and sorts by `(-score, position)`. Then:

```bash
uv run python -m rag.keyword idf <your rare word> <your common word> the
```

*You should see* one line per word with its df, chapter 2's idf and chapter 3's, and the stopword reported as dropped. On Homer:

```text
cyclops: term 'cyclop'  df 28  idf 4.2914  bm25_idf 4.3088
ship: term 'ship'  df 559  idf 1.3308  bm25_idf 1.3317
the: dropped by preprocess (a stopword), so it is in no passage and has no idf
```

Your hand-worked number from P05 Step 3 should match the `bm25_idf` column to the second decimal. Paste the three lines under `## Step 3` in `evidence/P06.md`.

**Step 4. Two checks: a rare name finds its passage, stopwords score nothing.**

```bash
uv run python -m rag.keyword <a name that appears once or twice in your corpus>
uv run python -m rag.keyword the and of
```

*You should see*, for the first, the passage containing the name at rank 1 under both methods, and a `terms:` line showing what your query became after `preprocess`. For the second, `terms: []` and two empty lists. *If it broke* with the name's passage missing under both, the `terms:` line is the clue: the stem the query became is not the stem the passage holds, which means the two sides were not processed the same way. *If it broke* with five passages at `0.000` for the stopwords, zero scores were not dropped.

**Step 5. The two methods disagree in a particular way.**

Run the two-word query you predicted in P05, the common word and the rare one together:

```bash
uv run python -m rag.keyword <your common word> <your rare word>
```

*You should see* two lists of five that do not agree. On Homer, `ship cyclops`:

```text
terms: ['ship', 'cyclop']
--- tfidf
 13.906  p0329  “Now off their harbour there lies a wooded and fertile island not quit
 12.874  p0987  [62] [ The fact of Alcinous knowing anything about the Cyclopes sugges
 11.244  p0997  [78] [ I take all this passage about the Cyclopes having no ships to b
  9.914  p0345  “But the cruel wretch said, ‘Then I will eat all Noman’s comrades befo
  9.914  p0352  “As he spoke he drove the ram outside, but when we were a little way o
--- bm25
  9.393  p0997  [78] [ I take all this passage about the Cyclopes having no ships to b
  8.622  p0987  [62] [ The fact of Alcinous knowing anything about the Cyclopes sugges
  7.461  p0352  “As he spoke he drove the ram outside, but when we were a little way o
  6.725  p0360  “When at last we got to the island where we had left the rest of our s
  6.595  p0342  “‘Look here, Cyclops,’ said I, you have been eating a great deal of ma
```

tf-idf's winner, `p0329`, is 140 terms long and says `ship` four times and `cyclop` twice: raw counts reward a long passage that repeats the common word. BM25's winner, `p0997`, is 37 terms long with two of each: saturation makes the four `ship`s worth 1.48 instead of 4, and the length normalization pushes the long passage down and the short one up. Below rank 1 the lists disagree the same way, and the rare word's idf, three times the common word's, is what makes `cyclop` count at all. Paste both lists under `## Step 5` in `evidence/P06.md`, then check the prediction you committed in `evidence/P05.md` and write one line under the paste: right or wrong, and which of the three fixes decided it.

**Extension — five queries and a scoreboard (ASSIGNED)**

You are writing the five queries that every search method from here to P11 is scored on. They go in a file and are committed **before** anything scores them.

**Part 1. `rag/queries.json`.** Five entries, this shape:

```json
[
  {"id": "q01", "query": "how does the wanderer fool the one-eyed giant about who he is",
   "needle": "my name is Noman", "kind": "paraphrase"}
]
```

`needle` is three to ten consecutive words copied exactly from the passage that answers the query, within one paragraph. It is how the scoreboard knows a result is right without you marking it by hand, and it keeps working when P11 replaces passages with chunks. The mix is fixed:

| How many | `kind` | What it is |
|---|---|---|
| 2 | `paraphrase` | the answer passage shares **no content words** with the query. Check it: `preprocess` both and confirm the two lists do not overlap. |
| 2 | `rare-name` | the query names a person, place or thing that is in only a few passages. |
| 1 | your choice | anything a real reader of your corpus would ask. Say the kind in one word. |

The example is a Homer query; yours are about your corpus. A query you cannot write a needle for is not a query the scoreboard can score.

```bash
git add rag/queries.json
git commit -m "P06: five scoring queries, written before any run"
git push
```

**Part 2. `rag/scoreboard.py`.** Open it and read `first_hit` and `methods`: for every query it runs each method, finds the first result whose text contains the needle, and records that rank; the `assert` in `methods` refuses to run if a needle is in no passage at all. Then:

```bash
uv run python -m rag.scoreboard
```

*You should see* a table with the rank of the answer passage per query per method, `-` for not in the top 5, then a `hit@3` total for each method. On Homer:

```text
query     tfidf     bm25
q01           -        -
q02           -        -
q03           -        -
q04           1        5
q05           1        1
hit@3       2/5      1/5
```

Expect both paraphrase queries to be missed by both methods: there is no shared term to match on, and that is the ceiling of keyword search, which chapter 4 is about. A rare-name query can miss too: Homer's `q03`, `how was the Cyclops blinded`, is answered by a passage that says `the monster's eye` and never `Cyclops`, so the name in the question is not in the answer. *If it broke* with `AssertionError: q02: needle not in any passage`, the needle is not an exact copy, or it crosses a paragraph break; fix the needle, not the check.

Paste the table under `## Extension` in `evidence/P06.md`.

**Where it got worse.** Find a query where BM25 ranked the answer lower than tf-idf did (on Homer, `q04`: tf-idf 1, BM25 5), and say which fix did it. If there is none in your five, run `rag.keyword` on three more questions about your corpus until you find one, and paste that one. On Homer, `q04`'s answer passage is 121 terms long with one `nausicaa`, one `play` and one `beach`; tf-idf likes it for having three of the five query terms, and BM25's length normalization lets four shorter passages with fewer matching terms past it. Write the query, both ranks and the fix under the table.

**Commit, push, sign off.** Fill `## Reflection` in `evidence/P06.md` (the questions are below) and run `uv run python scripts/check_evidence.py P06` until it says `all slots filled`. Then:

```bash
git add rag/ scratch/ evidence/
git commit -m "P06: BM25 and tf-idf over passages, five queries scored"
git push
```

The P05 pull request is already open; these commits land on it. Edit its body to name the query you are least sure about.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`evidence/P06.md` with every slot filled + `rag/queries.json` (committed before any score) + `evidence/p06-bootdev-ch3.png`.

**Reflection Questions** (answer under `## Reflection` in `evidence/P06.md`)

1. Paste the two Step 5 lists and your P05 prediction. Say which of the three fixes decided rank 1, with the numbers that show it: both passages' lengths and the count of each term in each, from `rag.keyword`'s source if you add a print, or from `uv run python -m rag.tfidf idf` and the passage text.

   *How to get it:* the two winners differ in length and in how many times the common word repeats. Saturation caps the repeats; length normalization divides by the passage's length against the average. One of those two moved the long passage down, and the idf ratio between your two words says how much the rare one counted.

2. Paste your scoreboard. For each `paraphrase` query, paste the `terms:` line `rag.keyword` printed and three content words from the answer passage, and point to the reason no term matched. For a paraphrase query that *was* found, name the shared stem.

   *How to get it:* `uv run python -m rag.keyword <the query>` prints the terms; `uv run python -c "from rag.preprocess import preprocess; print(preprocess('<the answer passage's first sentence>'))"` prints the other side.

3. Paste the query from "where it got worse" with both ranks. Say what BM25 would need to change, `k1` or `b`, and in which direction, for that query to come out the way tf-idf had it, and what that change would do to your Step 5 query.

   *How to get it:* `b` is the length penalty; `b = 0` switches it off and the long passage climbs back. `Keyword(load_passages(), b=0)` builds the index that way; run `search` on both queries with it and see which one you broke. On Homer, `b = 0` puts `q04` back at rank 1 and hands Step 5's rank 1 back to the long `p0329`: the same constant fixes one query and undoes the other.
