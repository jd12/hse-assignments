# P16 · Four Numbers for Four Methods

**Meetings:** D39 · **Points:** 10 pts

**Watch — none**

[Boot.dev, *Learn Retrieval Augmented Generation*](https://www.boot.dev/courses/learn-retrieval-augmented-generation) · chapter 9 Evaluation, lessons 5 to 7, about 25 minutes.

**During the chapter**

**Do lesson 5's table by hand first.** Before the F1 formula, the lesson shows five precision and recall pairs with their mean and their F1. Cover the last column, compute the five F1 values with the formula, and write them in `scratch/p15-eval.md`. The one that comes out `0` is the reason the harmonic mean exists.

**Lesson 6 asks a question you answer from your own output.** Why is `Land Before Time XI` not in the top five? The debug lines you add are the only way to know. Run the command, read your four debug lines, then choose. Write the line that told you in `evidence/P16.md` under Step 2.

**Notes**

**One question moves a score by 0.04.** Twenty-five questions: a method that finds one more is `0.04` better on recall@5, and a difference of one question is not a finding. The runner prints `n` above every table so the number is never read without its size. The chapter's set is ten, and its numbers move by `0.1`.

**Recall@1 ≤ MRR ≤ recall@5 is arithmetic, not a result.** A question found at rank 1 adds 1 to all three; one found at rank 3 adds a third to MRR and 1 to recall@5. If a row breaks that order, the metric code is wrong, not the pipeline.

**Two cross-encoders, decided by twenty-five questions.** P14's five queries said the chapter's model made things worse and could not say whether a bigger one would help. The walkthrough runs the two-layer model, then swaps in a six-layer one from the same family, `ms-marco-MiniLM-L-6-v2`, and the twenty-five decide. Whatever wins is frozen in `rag/rerank.py`, and every later day runs on it.

**Walkthrough — the chapter, then the runner on your twenty-five**

**Step 1. Branch, log.**
Same branch as P15:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch dev/eval && git pull
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Chapter 9, lessons 5 to 7, every lesson green; the F1 column by hand first
- [ ] Step 3: rag.eval.run, the table in evidence/P16.md, P15 predictions checked
- [ ] Step 4: prediction committed, the second cross-encoder, MODEL frozen
- [ ] Extension: the judge on three questions against my labels
- [ ] Fill every slot in evidence/P16.md (check_evidence.py: all slots filled); push, sign off
```

**Step 2. Chapter 9, lessons 5 to 7.**

In `~/version_control/rag-search-engine/`. When the chapter page shows every lesson complete, save it as `evidence/p16-bootdev-ch9.png` in the rag repo.

**Lesson 5: F1.** One formula and one line. In `cli/evaluation_cli.py`, after the recall line:

```python
        f1 = 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)
```

```python
        print(f"- F1 Score: {f1:.4f}")
```

The `if` is for a query that found nothing: both fractions are `0`, and `0 / 0` is a crash, not a score. Run, then submit:

```bash
uv run cli/evaluation_cli.py --limit 4
uv run cli/evaluation_cli.py --limit 8
bootdev run 4bc96f50-0a24-4af3-b300-bad11568f3f3
bootdev run -s 4bc96f50-0a24-4af3-b300-bad11568f3f3
```

*You should see* `children's animated bear adventure` at four with `Precision@4: 0.2500`, `Recall@4: 0.0769` and `F1 Score: 0.1176`, and `friendship transformation magic with bears` at eight with `0.2500`, `0.6667` and `0.3636`; those are the checks. The first is one of thirteen relevant titles found: recall is small because the denominator is thirteen, and F1 sits near the smaller of the two. *If it broke* with `ZeroDivisionError`, the `if` is on the wrong side of the division.

**Lesson 6: error analysis.** Four debug lines in `rrf-search`, each printing what one stage produced, so a missing result can be traced to the stage that lost it. In `cli/hybrid_search_cli.py`, above `main`:

```python
# In: a label and anything.  Out: nothing; prints it only when --debug is on (chapter 9, error analysis).
def debug(enabled, label, value):
    if enabled:
        print(f"[debug] {label}: {value}")
```

A `--debug` flag on the `rrf-search` parser:

```python
    rrf_parser.add_argument("--debug", action="store_true", help="Print what each stage of the pipeline did")
```

`action="store_true"` makes a flag with no value: present means `True`. Then the four calls in the case, one after each stage:

```python
            query = args.query
            debug(args.debug, "original query", query)
            if args.enhance:
                enhanced = enhance_query(args.enhance, query)
                print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced}'\n")
                query = enhanced
            debug(args.debug, "query after enhancement", query)
            searcher = HybridSearch(load_movies())
            pool = args.limit * 5 if args.rerank_method else args.limit
            results = searcher.rrf_search(query, args.k, pool)[:pool]
            debug(args.debug, "after RRF", [e["doc"]["title"] for e in results])
            if args.rerank_method:
                ...
                results = results[:args.limit]
            debug(args.debug, "final results", [e["doc"]["title"] for e in results])
```

The `...` is the reranking block from P14, unchanged. Then the lesson's command, with the flag:

```bash
uv run cli/hybrid_search_cli.py rrf-search "dinosaur" --rerank-method cross_encoder --debug
```

*You should see* the four `[debug]` lines around the normal output. Find `Land Before Time XI` in them: if it is in the `after RRF` list and not in `final results`, the reranker dropped it; if it is in neither, hybrid search never gave it to the reranker, and the BM25 and semantic ranks tell you which engine. Answer the lesson's question from that, and write your choice and the line that decided it under `## Step 2` in `evidence/P16.md`. No submission for this lesson; the question is the check.

**Lesson 7: the model as judge.** A flag that, after the results print, asks the model to score each one from 0 to 3. In `cli/hybrid_search_cli.py`, the prompt above `main`, the import, the flag, and the block at the end of the case:

```python
from lib.llm import ask
from lib.reranking import parse_json_list
```

```python
EVALUATE = """Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{results}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
```

```python
    rrf_parser.add_argument("--evaluate", action="store_true", help="Rate the results with the LLM, 0 to 3")
```

```python
            if args.evaluate:
                formatted_results = [f"{i}. {e['doc']['title']} - {e['doc']['description']}" for i, e in enumerate(results, start=1)]
                scores = parse_json_list(ask(EVALUATE.format(query=query, results=chr(10).join(formatted_results))))
                for i, (entry, score) in enumerate(zip(results, scores), start=1):
                    print(f"{i}. {entry['doc']['title']}: {score}/3")
```

`chr(10)` is a newline character, written that way because a backslash is not allowed inside the braces of an f-string; `zip` walks the results and the scores together. Run, then submit:

```bash
uv run cli/hybrid_search_cli.py rrf-search "family movie about bears in the woods" --evaluate
bootdev run -s 1f4bdbed-99c3-434d-8e68-5a2d30448c9a
```

*You should see* the five results, then five lines ending in `/3`, with the Berenstain title among them; `/3` and that title are the check. *If it broke* with `json.decoder.JSONDecodeError`, the model wrote words around its list; `parse_json_list` cuts to the first `[` and last `]`, so the usual cause is a reply with no list at all. Retry; the lesson is marked safe.

**Step 3. The runner.**

Open `rag/eval/metrics.py`: chapter 9's three numbers for one question plus the reciprocal rank, each a few lines. Open `rag/eval/run.py`: every method on every golden question, averaged. Then:

```bash
uv run python -m rag.eval.run
```

*You should see* `n = 25`, four rows, and the precision ceiling on the last line. On Homer, with the chapter's cross-encoder:

```text
n = 25 questions; one question moves recall by 0.040
method     prec@5  recall@5   F1     MRR
bm25       0.1360    0.4633  0.2037 0.3527
semantic   0.0800    0.2600  0.1200 0.2280
hybrid     0.1360    0.4933  0.2081 0.4547
rerank     0.1120    0.3933  0.1700 0.3467
highest precision@5 this set allows: 0.3040
```

Read the rows against each other. BM25 beats semantic on this corpus by a wide margin: Butler's names are rare tokens, and the semantic model's 256-token window never got the whole passage. Hybrid is a little above both on recall and well above both on MRR, which is fusion doing what P11 said it would. And the chapter's reranker is below hybrid on every column, which is P14's five queries again, now on twenty-five. Paste the table under `## Step 3`, check your P15 predictions, and look up your three expected misses in `rag/eval/runs.json` (the per-question ranks the runner saved) to say what happened to them.

*If it broke* with a row where `MRR` is above `recall@5` or below the recall@1 you can read off the per-question ranks, the metrics are wrong; `rag/eval/metrics.py` is the template's. *If it broke* with `FileNotFoundError: golden.json`, you are on a branch from before P15's commit.

**Step 4. The second cross-encoder, predicted first.**

`ms-marco-MiniLM-L-6-v2` is the same family as the chapter's model, six layers instead of two, about 90 MB. Before you run it, write under `## Step 4` in `evidence/P16.md` whether it will raise or lower recall@5 and MRR, and by about how much, and commit the file. I check your commit timestamps. Then, in `rag/rerank.py`, change the one line:

```python
MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

and run the runner again:

```bash
uv run python -m rag.eval.run
```

*You should see* three rows unchanged and the fourth different. On Homer:

```text
rerank     0.1600    0.5933  0.2462 0.5513
```

From below hybrid on every column to above it on every column: recall@5 up from `0.49` to `0.59`, MRR from `0.45` to `0.55`, two and a half questions' worth. The six-layer model reads Butler's sentences; the two-layer one read the words. Paste the row, and freeze whichever model won on your corpus in `rag/rerank.py` with one line under `## Step 4` saying why. On Homer that is the six-layer model, and every later day runs on it.

*If it broke* with the download stalling, the model is 90 MB and school wifi is what it is; the file lands in the same cache as the others and never downloads again.

**Extension — The judge against your labels** *(assigned)*

Lesson 7's judge and your golden labels answer the same question about the same pairs: is this chunk relevant to this question? Run the judge on three of your questions, one request each, and count where it disagrees with you:

```bash
uv run python -m rag.eval.judge <three golden ids, like g01 g07 g17>
```

*You should see* for each question the top five, your label for each (`yes` if it is in `chunk_ids`), the judge's `0` to `3`, and `agree` or `DISAGREE`, then a total. On Homer:

```text
{{LLM:P16-judge}}
```

Paste it under `## Extension` and write "where it got worse": the chunk you and the judge disagree on most, with the judge's score, who you think is right, and the sentence from the chunk that settles it. Open the chunk; do not decide from its first line.

**Commit, push, sign off.** Fill `## Reflection` in `evidence/P16.md` (the questions are below) and run `uv run python scripts/check_evidence.py P16` until it says `all slots filled`. Then:

```bash
git add rag/rerank.py rag/eval/runs.json scratch/ evidence/
git commit -m "P16: four methods on the golden set, the six-layer cross-encoder chosen, the judge against my labels"
git push
```

The P15 pull request is still open and now has both days on it; leave it for me.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`evidence/P16.md` with every slot filled, the Step 4 prediction committed before the run + `rag/rerank.py` with the model you chose + `evidence/p16-bootdev-ch9.png`.

**Reflection Questions** (answer under `## Reflection` in `evidence/P16.md`)

1. Paste your table. Say, for the method with the best MRR, how MRR can be above recall@1 and below recall@5 at the same time, with one question's rank as the example.

   *How to get it:* MRR averages `1 / rank` over questions, with `0` for a miss. A question found at rank 3 adds `0.33`, which is more than the `0` it adds to recall@1 and less than the `1` it adds to recall@5. Pick one such question from `runs.json`.

2. Paste both rerank rows from Step 4. Say what changed between the two models that the five scoreboard queries in P14 could not show, and why twenty-five questions could.

   *How to get it:* on five queries both models scored 1 of 5 and looked the same; on twenty-five, one found twelve questions and the other seventeen. A difference of one question in five is noise; a difference of five in twenty-five is a fifth of the set.

3. Paste your judge output. The chapter says to have domain experts write the criteria and spot-check the judge. Say what you are in that sentence for your corpus, what one disagreement told you about either the judge or your labels, and which one you changed.

   *How to get it:* you are the domain expert; the labels are your criteria. A chunk the judge scores `3` that you did not label is either a second correct chunk you missed (fix the label) or a chunk that mentions the subject without answering the question (the judge is wrong, and that is what spot-checking is for). Either is a fine answer if you read the chunk.
