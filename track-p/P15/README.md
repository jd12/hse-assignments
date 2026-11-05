# P15 · The Golden Set, Labeled Before Anything Runs

**Meetings:** D38 · **Points:** 10 pts

**Watch — none**

[Boot.dev, *Learn Retrieval Augmented Generation*](https://www.boot.dev/courses/learn-retrieval-augmented-generation) · chapter 9 Evaluation, lessons 1 to 4, about 25 minutes.

**During the chapter**

**Copy lesson 1's three questions.** The vibe check is three questions you ask of a result list. Write them in `scratch/p15-eval.md` in the rag repo, exactly as he words them. Step 3 asks them of your own results.

**Write the two formulas as fractions.** Lessons 3 and 4 each give one line of arithmetic. Write both in `scratch/p15-eval.md` with the words the chapter uses for numerator and denominator, and beside each, which of the two "how much of what you found" and "how much of what is there" it answers. P16's runner computes both and you will need to say which is which without looking.

**Notes**

**A golden set is labels made by someone who knows the text.** The chapter says a movie database needs a film critic. Your corpus has one reader who knows it, and that is you. Twenty-five questions, each with the chunk that answers it, is an afternoon's labeling and the only honest instrument you will have for the rest of the semester. P16, P20, P21 and P22 all run on it.

**Labels come from the finder, not from your search.** If you locate the answer chunk by running `rag.search`, your golden set contains only answers your pipeline can already find, and the eval will score the pipeline against itself. `rag/eval/find.py` is an exact-phrase scan of every chunk with no ranking in it. Label with that.

**The set is committed before the runner exists.** P16 writes the runner. A label written after you have seen what the pipeline returned is a label the pipeline helped write; I check the commit timestamps on `rag/eval/golden.json`.

**Walkthrough — the chapter, then twenty-five labels**

**Step 1. Merge, branch, log.**
Open the rag repo on GitHub and merge the P14 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch main && git pull
git switch -c dev/eval
```

*If P14 is not approved yet:* `git switch dev/rerank && git switch -c dev/eval`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Chapter 9, lessons 1 to 4, every lesson green
- [ ] scratch/p15-eval.md: the three questions, the two fractions
- [ ] Step 3: the vibe check on three queries, the table in evidence/P15.md
- [ ] Step 4: the finder, once on a phrase that is in two chunks
- [ ] Step 5: rag/eval/golden.json, twenty-five questions, check_golden green, committed
- [ ] Step 6: predictions committed before any runner exists
- [ ] Fill every slot in evidence/P15.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. Chapter 9, lessons 1 to 4.**

In `~/version_control/rag-search-engine/`. Stop after lesson 4; lessons 5 to 7 are P16. When the chapter page shows lessons 1 to 4 complete, save it as `evidence/p15-bootdev-ch9a.png` in the rag repo.

**Lesson 1: the vibe check.** A reading and one question; the answer is the one about usefulness and gaps that no metric captures.

**Lesson 2: the golden dataset.** Copy the JSON from the bottom of the lesson page into `data/golden_dataset.json`. Ten test cases, each a `query` and a list of `relevant_docs` titles. Run and submit:

```bash
bootdev run 8796f663-0953-4dc0-948c-a84734db429f
bootdev run -s 8796f663-0953-4dc0-948c-a84734db429f
```

*You should see* the check count ten test cases. *If it broke* with a count other than ten, a comma or bracket went missing in the paste; the lesson's JSON is complete as shown.

**Lesson 3: precision.** A new script with one flag and no subcommands. Create `cli/evaluation_cli.py`:

```python
import argparse
import json

from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    with open("data/golden_dataset.json", encoding="utf-8") as f:
        golden = json.load(f)

    searcher = HybridSearch(load_movies())
    print(f"k={limit}\n")
    for case in golden["test_cases"]:
        query = case["query"]
        relevant = case["relevant_docs"]
        results = searcher.rrf_search(query, 60, limit)[:limit]
        retrieved = [entry["doc"]["title"] for entry in results]
        relevant_retrieved = len([title for title in retrieved if title in relevant])
        precision = relevant_retrieved / len(retrieved) if retrieved else 0.0
        print(f"- Query: {query}")
        print(f"- Precision@{limit}: {precision:.4f}")
        print(f"- Retrieved: {', '.join(retrieved)}")
        print(f"- Relevant: {', '.join(relevant)}")
        print()


if __name__ == "__main__":
    main()
```

`[title for title in retrieved if title in relevant]` is a list of the retrieved titles that are also in the relevant list, so its length is "relevant retrieved". The `60` is RRF's `k`; the `--limit` is the other k, the chapter's P@K. Run, then submit:

```bash
uv run cli/evaluation_cli.py --limit 3
uv run cli/evaluation_cli.py --limit 6
bootdev run 89e06cde-6e37-417f-bd48-498163aeed6f
bootdev run -s 89e06cde-6e37-417f-bd48-498163aeed6f
```

*You should see* `dinosaur park` and `cute british bear marmalade` at `Precision@3: 0.3333` and `Precision@6: 0.1667`; those four lines are the check, and they are the arithmetic of one relevant title in three and in six. Write the two numbers in `evidence/P15.md` under Step 2. *If it broke* with every precision at `0.0000`, the titles in `relevant_docs` are compared with `entry["doc"]["title"]`; a different key name in your `rrf_search` entries would make every comparison fail.

**Lesson 4: recall.** One more fraction and one more line. After the precision line:

```python
        recall = relevant_retrieved / len(relevant) if relevant else 0.0
```

```python
        print(f"- Recall@{limit}: {recall:.4f}")
```

Run, then submit:

```bash
uv run cli/evaluation_cli.py --limit 5
uv run cli/evaluation_cli.py --limit 10
bootdev run eb547fd5-1b26-417a-9327-c0d3ecfeea3d
bootdev run -s eb547fd5-1b26-417a-9327-c0d3ecfeea3d
```

*You should see* `cute british bear marmalade` at `Precision@5: 0.2000` and `Recall@5: 1.0000`, one relevant title found out of one, and `car racing` with both lines at ten; those are the checks. *If it broke* with a recall above `1.0000`, a title is counted twice; `relevant_retrieved` counts retrieved titles, and RRF returns each movie once.

**Step 3. The vibe check, on your chunks.**

Three queries a reader of your corpus would type, not from `rag/queries.json` and not questions you will label in Step 5. Run each through the full pipeline as it stands:

```bash
uv run python -m rag.rerank "<query 1>"
uv run python -m rag.rerank "<query 2>"
uv run python -m rag.rerank "<query 3>"
```

*You should see* five chunks per query with scores. On Homer, for `the song of the Sirens`:

```text
cross-encoder rerank of the hybrid top 20 for: the song of the Sirens
1. c00393  score 4.588  was hybrid rank 1  Sirens who enchant all who come near them. If any one unwarily draws i...
2. c00764  score -1.426  was hybrid rank 18  heard the wondrous singing of the Sirens, and went on to the wandering...
3. c00402  score -2.493  was hybrid rank 2  Meanwhile I look a large wheel of wax and cut it up small with my swor...
4. c00403  score -4.448  was hybrid rank 5  way not only charmed, but wiser, for we know all the ills that the god...
5. c00390  score -6.270  was hybrid rank 10  men to go on board at once and loose the hawsers; so they embarked and...
```

and for `who killed Hector`, five chunks in which Hector kills people, with scores near 8, and not the chunk where Achilles kills him. For each of your three, answer lesson 1's three questions in the table under `## Step 3` in `evidence/P15.md`: what is here that should not be, what is not here that should be, would I click. Read the chunks; the first line of each is not enough. On Homer the Sirens list is good at the top and padding from rank 3 down, and `who killed Hector` is five wrong answers with confident scores, because every chunk has `Hector` and `killed` in it and the model reads the pair, not the sentence's subject.

**Step 4. The finder.**

```bash
uv run python -m rag.eval.find <a short phrase you know is in your corpus>
```

*You should see* one line per chunk containing the phrase, with its id and character position, and a count. A phrase near a chunk boundary is in two chunks because of the overlap, and a sentence Homer repeats is in more. On Homer, `She fooled us in this way for three years` is in four: two chunks each at two places in the Odyssey, because Antinous tells the story in Book II and the ghost of Amphimedon tells it again in Book XXIV. Every chunk that contains the answer is a correct label; list all of them. Paste a run where the phrase is in more than one chunk under `## Step 4`.

**Step 5. Write `rag/eval/golden.json`.**

Twenty-five entries. Open your corpus, pick the passage first, write the question a reader would ask about it, copy the sentence that answers it as the `quote`, run the finder on a few words of it to get the `chunk_ids`, and write the shortest string any correct answer must contain as the `answer` (P20 checks the agent's answers with it, so pick a word with one spelling). The shape:

```json
[
  {"id": "g01", "kind": "ordinary",
   "question": "How did the old nurse recognize Ulysses while washing his feet?",
   "chunk_ids": ["c00644"],
   "quote": "she at once knew the scar as one that had been given him by a wild boar",
   "answer": "scar"}
]
```

A two-chunk question has a second sentence in `quote2`, found in a different chunk, and both ids. The mix, with Homer's counts:

| How many | `kind` | What it is |
|---|---|---|
| 9 | `paraphrase` | no content words shared with the answer chunk ("the enchantress on the island" for Circe) |
| 9 | `ordinary` | what a reader of your corpus would actually ask |
| 5 | `exact` | turns on a name, number or term that appears in few chunks |
| 2 | `two-chunk` | the answer needs two chunks that are not neighbors; `quote` and `quote2`, both ids |

Do not reuse the ten queries in `rag/queries.json` and `rag/questions.json`; those have been tuned against since P06. Then:

```bash
uv run python -m rag.eval.check_golden
```

*You should see* your frozen chunking, the count, the four kinds, a mean between 1 and about 2, and `golden set is consistent with the chunking`. On Homer:

```text
chunking 200/50, 1882 chunks, 25 questions
kinds: {'ordinary': 9, 'exact': 5, 'paraphrase': 9, 'two-chunk': 2}
mean labeled chunks per question: 1.52
golden set is consistent with the chunking
```

The first line is the point of the script: the labels are tied to the `SIZE` and `OVERLAP` you froze in P09, and if either ever changes, every label is wrong and this script says so. *If it broke* with `quote not in its chunks`, the quote is not an exact copy, or the finder's chunk id was mistyped. *If it broke* with `answer not in its chunks`, the answer string is a word that is not in the chunk; it has to be, because that is how P20 will check an answer without a judge.

Paste the output under `## Step 5`, then commit the set before anything else:

```bash
git add rag/eval/golden.json scratch/p15-eval.md evidence/
git commit -m "P15: golden set of twenty-five, labeled with the finder, before any runner"
git push -u origin dev/eval
```

Write the commit hash under `## Step 5`. Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. It stays open through P16.

**Step 6. Predict, then commit, before the runner exists.**

P16's runner will print recall@5 and MRR for `bm25`, `semantic`, `hybrid` and `rerank` on your twenty-five. Fill the table under `## Step 6` in `evidence/P15.md` with numbers, not words, and name the three questions you expect every method to miss, with a reason each. The scoreboard's hit@3 on five queries is your only evidence; use it. Commit:

```bash
git add evidence/P15.md
git commit -m "P15: predictions for the four methods on the golden set"
git push
```

**Commit, sign off.** Fill `## Reflection` in `evidence/P15.md` (the questions are below) and run `uv run python scripts/check_evidence.py P15` until it says `all slots filled`. Then:

```bash
git add evidence/
git commit -m "P15: vibe check, the finder, reflections"
git push
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`rag/eval/golden.json` with twenty-five questions, `check_golden` green, committed before the predictions and the predictions before P16 + `evidence/P15.md` with every slot filled + `scratch/p15-eval.md` + `evidence/p15-bootdev-ch9a.png`.

**Reflection Questions** (answer under `## Reflection` in `evidence/P15.md`)

1. Paste your vibe-check table. Say what the three questions caught that the scoreboard's rank column never could, and which of your three queries you would change the retrieval for, not the labels.

   *How to get it:* the scoreboard asks whether one chunk is in the top five; the vibe check asks what the other four are doing there. `who killed Hector` scores five confident chunks that all answer a different question, and no rank column shows that. That one is a retrieval problem; a query whose needle was simply mislabeled is a label problem.

2. Paste your check_golden output. Say why the labels came from the finder and not from `rag.search`, and what the eval would be measuring if they had.

   *How to get it:* the finder cannot miss a chunk that contains the phrase; the search can. A set labeled by search holds only what search finds, so recall is 1.0 by construction and the number measures nothing.

3. The chapter's golden set has ten queries with up to thirteen relevant titles each; yours has twenty-five questions with one or two chunks each. Say what precision@5 can reach on each set at best, and why the chapter's set is a better fit for precision than yours.

   *How to get it:* a question with one relevant chunk cannot score above `0.2` at five, so Homer's set caps precision@5 at `0.304`, the runner's last line; the chapter's `children's animated bear adventure` has thirteen relevant titles and can reach `1.0`. Precision is for queries with many right answers; your questions have one, and recall and MRR are the numbers to read.
