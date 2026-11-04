# P10 · Two Engines, One Scale

**Meetings:** D33 · **Points:** 10 pts

**Watch — none**

[Boot.dev, *Learn Retrieval Augmented Generation*](https://www.boot.dev/courses/learn-retrieval-augmented-generation) · chapter 6 Hybrid Search, lessons 1 to 3, about 20 minutes.

**During the chapter**

**Write down the fusion.** Lesson 2's reading and the stub it gives you name two ways of combining a keyword ranking with a semantic one. Write both names in `scratch/p10-hybrid.md` in the rag repo, and as lessons 3 to 5 give you their constants (`alpha`, `k`), write each one down with its value. P11 puts `k` into `rag/hybrid.py`.

**Predict the fusion on your five queries.** Before P11 runs it, write under `## Predictions` in `evidence/P10.md`, for each of your five queries, whether you expect the hybrid rank to be better than the better of BM25 and semantic, the same, or worse. One line each. Commit the file; I check the timestamps.

**Notes**

**Scores on different scales cannot be added.** Cosine similarity lives between -1 and 1. BM25 is unbounded and grows with the corpus; your P06 scores were around 10. Add them and BM25 decides everything, and the cosine only reorders ties. Lesson 3's normalization is one answer; P11's rank fusion is the other, and it throws the scores away entirely.

**Min-max normalization is only as good as its ends.** It maps the lowest score in a list to 0 and the highest to 1. One freak high score squashes everything else toward 0, and a list of one score has no range at all. The chapter's two special cases (no scores, all equal) are the two places the formula divides by nothing.

**Walkthrough — the chapter, then your two engines side by side**

**Step 1. Merge, branch, log.**
Open the rag repo on GitHub and merge the P09 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch main && git pull
git switch -c dev/hybrid
```

*If P09 is not approved yet:* `git switch dev/chunks && git switch -c dev/hybrid`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Chapter 6, lessons 1 to 3, every lesson green
- [ ] scratch/p10-hybrid.md: the two fusion methods, the constants so far; my five predictions in evidence/P10.md, committed
- [ ] Step 3: the two chapter comparisons and two of my own, in evidence/P10.md
- [ ] Step 4: BM25 and cosine scores on one of my queries, and the normalized lists
- [ ] Fill every slot in evidence/P10.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. Chapter 6, lessons 1 to 3.**

In `~/version_control/rag-search-engine/`. Stop after lesson 3; lessons 4 and 5 are P11.

**Lesson 1: keyword against semantic.** A reading with a table of query types and which engine wins each. One multiple-choice question. Note the two blind-spot lists; Step 3 is about them.

**Lesson 2: the stub.** Before the assignment, the lesson asks you to run two queries through both engines and compare. Do that first; the output goes in Step 3:

```bash
uv run cli/semantic_search_cli.py search_chunked "gods among mortals"
uv run cli/keyword_search_cli.py bm25search "gods among mortals"
uv run cli/semantic_search_cli.py search_chunked "Bear DiCaprio"
uv run cli/keyword_search_cli.py bm25search "Bear DiCaprio"
```

Then the code. A third CLI and a third library module that imports the other two.

1. Create `cli/hybrid_search_cli.py`:

```python
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    args = parser.parse_args()

    match args.command:
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
```

2. Create `cli/lib/hybrid_search.py`. The course's starting point uses `from .keyword_search import ...`; this project runs the CLIs as scripts, so the imports say `lib.` as the other two modules do. The constructor builds the keyword index if `cache/` does not have one yet, and loads or builds the chunk embeddings; the two search methods are stubs until lessons 4 and 5:

```python
import os

from lib.keyword_search import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists("cache/index.pkl"):
            self.idx.build()
            self.idx.save()

    # In: a query and how many results.  Out: the keyword engine's (movie, score) pairs.
    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
```

The leading underscore on `_bm25_search` is a Python convention for "used inside this class"; nothing enforces it.

3. Run, then submit:

```bash
uv run cli/hybrid_search_cli.py
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* the help text with `Hybrid Search CLI` in it; that is the check. *If it broke* with `ModuleNotFoundError: No module named 'lib'`, you ran it from inside `cli/`.

**Lesson 3: min-max normalization.** Every score in a list is mapped to `(score - min) / (max - min)`, so the lowest becomes 0 and the highest 1.

1. In `cli/lib/hybrid_search.py`, above the class:

```python
# In: a list of scores.  Out: the same scores scaled so the lowest is 0.0 and the highest 1.0; all 1.0 if they are equal.
def normalize_scores(scores):
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if high == low:
        return [1.0 for _ in scores]
    return [(s - low) / (high - low) for s in scores]
```

`[1.0 for _ in scores]` makes a list with one `1.0` per score; `_` is the name you give a loop variable you do not use.

2. In the CLI, `normalize_scores` imported, a `normalize` subcommand whose argument takes zero or more numbers, and its case:

```python
from lib.hybrid_search import normalize_scores
```

```python
    normalize_parser = subparsers.add_parser("normalize", help="Min-max normalize a list of scores")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="Scores to normalize")
```

```python
        case "normalize":
            for score in normalize_scores(args.scores):
                print(f"* {score:.4f}")
```

`nargs="*"` gathers every remaining argument into a list, and an empty list when there are none.

3. Run, then submit:

```bash
uv run cli/hybrid_search_cli.py normalize 0.5 2.3 1.2 0.5 0.1
uv run cli/hybrid_search_cli.py normalize 5 5 5
uv run cli/hybrid_search_cli.py normalize
bootdev run <lesson-id>
bootdev run -s <lesson-id>
```

*You should see* `0.1818`, `1.0000`, `0.5000`, `0.1818`, `0.0000`, then three `1.0000`s, then nothing; the first two are the checks. The two `0.5`s got the same number, which is the point of a formula: equal in, equal out. *If it broke* with `ZeroDivisionError` on the second command, the `high == low` case comes after the division instead of before.

**Step 3. The two engines on the course data, then on yours.**

Under `## Step 3` in `evidence/P10.md`, paste the four lesson 2 outputs, two queries through two engines, and say in one line each which engine did better and why, using lesson 1's table: `Bear DiCaprio` is an actor's name, `gods among mortals` is a theme.

Then the same on your corpus, with your P06 and P08 engines on your passages. Pick one `rare-name` query and one `paraphrase` query from `rag/queries.json`:

```bash
uv run python -m rag.keyword <your rare-name query>
uv run python -m rag.semantic <your rare-name query>
uv run python -m rag.keyword <your paraphrase query>
uv run python -m rag.semantic <your paraphrase query>
```

*You should see* what the scoreboard already told you in numbers, now as the passages themselves: the name query found by keyword search on the name, the paraphrase query found by semantic search on the meaning, and each engine's miss. Paste the four outputs' top three lines and say which blind spot from lesson 1 each miss is.

**Step 4. The scale problem, on your numbers.**

Take the `paraphrase` query from Step 3 and put the two engines' top-five scores side by side:

```bash
uv run python -c "
from rag.corpus import load_passages
from rag.keyword import Keyword
from rag.semantic import Semantic
ps = load_passages()
q = '<your paraphrase query>'
bm = [round(r['score'], 2) for r in Keyword(ps).search(q)]
se = [round(r['score'], 3) for r in Semantic(ps, 'passages').search(q)]
print('bm25    ', bm)
print('semantic', se)
"
```

*You should see* BM25 scores in the single or double digits and cosine scores below 1. Add the first of each and the sum is the BM25 score plus a rounding error: that is lesson 3's problem. Now normalize each list by hand with the lesson 3 formula, or with your `normalize` command by pasting the numbers in, and put the two normalized lists under the raw ones. The first of each is now 1.0, the last 0.0, and the two lists can be compared. Paste all four lines under `## Step 4`.

*If it broke* with BM25 returning fewer than five results, your query's terms are in fewer than five passages; use the other query.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/P10.md` (the questions are below) and run `uv run python scripts/check_evidence.py P10` until it says `all slots filled`. Then:

```bash
git add scratch/ evidence/
git commit -m "P10: chapter 6 started, fusion methods and constants written down, predictions"
git push -u origin dev/hybrid
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. It stays open through P11.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`evidence/P10.md` with every slot filled + `scratch/p10-hybrid.md`.

**Reflection Questions** (answer under `## Reflection` in `evidence/P10.md`)

1. Paste your Step 4 raw lists and normalized lists. Say what adding the raw scores would have ranked by, and what information the normalized lists lose that the raw ones had.

   *How to get it:* the raw BM25 numbers say how far apart two passages are in absolute terms; after min-max, the top is 1.0 whether it was a strong match or the best of a bad five. A list where the top BM25 score was 2.1 and one where it was 14.9 normalize to the same 1.0.

2. Paste the `Bear DiCaprio` outputs. Say which engine found the actor and why the other could not, in the words of lesson 1's blind-spot lists, and what the same mechanism did to your rare-name query in Step 3.

   *How to get it:* a name is one rare token to BM25, and it is a small part of a 384-number summary to the model, which was never trained to know who DiCaprio is.

3. Paste your five predictions. For the query you are least sure about, say what would have to be true of the two engines' lists for the fusion to beat both of them.

   *How to get it:* lesson 2's reading: a fusion can only promote a passage that at least one engine ranked, and it rewards passages both engines ranked. A passage one engine missed entirely gets only the other's vote.
