# P21 · Postmortem: Five Questions Traced to the Stage That Lost Them

**Meetings:** D44 · **Points:** 10 pts

**Watch — none**

No Boot.dev today. Chapter 9's lesson 6 said: when search fails, do not shrug, find the stage. Today is that lesson on your golden set, with a tracer that prints every stage's rank for every question the pipeline still misses.

**Notes**

**A miss has an address.** The pipeline is five stages: BM25, semantic, fusion, the reranker's pool, the reranker. `rag/eval/trace.py` prints the rank of the labeled chunk at each one. A chunk at BM25 rank 13 and semantic rank `-` (not in the top 50) lands at hybrid rank 24, outside the reranker's pool of 20, and the reranker never sees it: lost at fusion. A chunk in the pool that the reranker put ninth: lost at the reranker. A chunk in neither engine's top 50: lost before anything you can tune.

**Measure, then change one thing, predicted first.** The tracer will show most of your misses at one stage. A constant in `rag/` governs that stage, and the honest experiment is: predict how many of the five it recovers, change it, rerun the runner, and read every column, because a constant that recovers two questions can cost one elsewhere.

**Fixing a golden question is how you overfit a golden set.** Twenty-five questions you wrote are the test set. Every change made today to recover one of them makes the set a little less honest about the next question nobody has written yet. The report names each change as what it is.

**Walkthrough — every miss traced, five explained, one change**

**Step 1. Merge, branch, log.**
Open the rag repo on GitHub and merge the P19 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch main && git pull
git switch -c dev/postmortem
```

*If P19 is not approved yet:* `git switch dev/milestone-a1 && git switch -c dev/postmortem`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Step 2: rag.eval.trace on every miss, the stage table in evidence/P21.md
- [ ] Step 3: five traced with the expansion, five requests
- [ ] Step 4: one constant, predicted first, the runner before and after
- [ ] Extension: rag/POSTMORTEM.md
- [ ] Fill every slot in evidence/P21.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. The tracer on every miss.**

Open `rag/eval/trace.py` and read `trace` and `lost_at`: five ranks and the rule that names the first stage where the chunk fell out. Then, with no arguments, it finds every golden question whose labeled chunk is not in the final top five and traces each:

```bash
uv run python -m rag.eval.trace
```

*You should see* a count, then one block per miss. On Homer:

```text
8 of 25 questions have no labeled chunk in the final top 5

g03 (paraphrase): How did the enchantress on the island change the crew into animals?   labeled ['c00327']
   original  bm25 -  semantic -  hybrid -  in rerank pool no  rerank -   -> lost at: both engines (never retrieved)

g04 (ordinary): How long did Penelope fool the suitors with her weaving?   labeled ['c00052', 'c00053', 'c00775', 'c00776']
   original  bm25 4  semantic 38  hybrid 21  in rerank pool no  rerank -   -> lost at: fusion (outside the rerank pool)

g10 (paraphrase): How many sailors does the many-headed monster take from a ship at each pass?   labeled ['c00398']
   original  bm25 13  semantic -  hybrid 24  in rerank pool no  rerank -   -> lost at: fusion (outside the rerank pool)

g11 (paraphrase): Who recognized the returning master before any person in the house did, and what happened to him?   labeled ['c00571', 'c00572']
   original  bm25 24  semantic -  hybrid 46  in rerank pool no  rerank -   -> lost at: fusion (outside the rerank pool)

g12 (ordinary): Which two kings does Telemachus visit for news of his father?   labeled ['c00036']
   original  bm25 47  semantic -  hybrid -  in rerank pool no  rerank -   -> lost at: fusion (outside the rerank pool)

g15 (ordinary): Whose prize did Agamemnon threaten to take from Achilles?   labeled ['c00875', 'c00876']
   original  bm25 13  semantic -  hybrid 29  in rerank pool no  rerank -   -> lost at: fusion (outside the rerank pool)

g20 (exact): In how many layers did Vulcan make the shield of Achilles?   labeled ['c01620']
   original  bm25 10  semantic -  hybrid 26  in rerank pool no  rerank -   -> lost at: fusion (outside the rerank pool)

g22 (paraphrase): What did the old king do when he reached the tent of the man who had killed his sons?   labeled ['c01859']
   original  bm25 22  semantic -  hybrid 46  in rerank pool no  rerank -   -> lost at: fusion (outside the rerank pool)
```

Seven of Homer's eight misses have the same shape: BM25 found the chunk somewhere in its top 50, the semantic engine did not find it at all, and a chunk with one vote lands at hybrid rank 21 to 46, past the reranker's pool of 20. The reranker never read it. g04 is the sharpest case: BM25 rank 4, and still lost, because semantic rank 38 is worth almost nothing in RRF. The eighth, g03, neither engine found: `enchantress`, `island`, `crew`, `animals` are none of them Butler's words for Circe turning men into pigs. Paste the output under `## Step 2` and fill the stage table with your counts.

*If it broke* with `0 of 25 questions`, your pipeline misses nothing at five, which with twenty-five hand-written questions is a sign the golden set was labeled from search; P15's Reflection said why. Trace five questions by id instead, the five with the lowest reranker scores in `rag/eval/runs.json`.

**Step 3. Five traced, with the expansion.**

Pick five of your misses, at least one from each stage that has any. Chapter 7's expansion is the one tool from the course that acts before every stage, so trace each of the five with and without it, one request each:

```bash
uv run python -m rag.eval.trace --expand <five ids>
```

*You should see* each question's original trace, the expanded query, and the expanded trace under it. On Homer, for g04, g10, g15, g20 and g22:

```text
{{LLM:P21-expand}}
```

Paste it under `## Step 3`. An expansion that moves a chunk from semantic `-` to a semantic rank is the fusion getting its second vote; one that moves BM25 the wrong way is dilution from P13.

**Step 4. One change, predicted first.**

The stage that lost the most of your misses has a constant: `POOL` in `rag/rerank.py` for the reranker's pool (20), `POOL` in `rag/search.py` for what each engine hands to fusion (50), `K` in `rag/hybrid.py` for how top-heavy RRF is (60), `MODEL` in `rag/rerank.py`. Choose one, and before you change it, write under `## Step 4` in `evidence/P21.md` which constant, the new value, and how many of your five it will recover. Commit the file. I check your commit timestamps. Then change it and rerun both the runner and the tracer:

```bash
uv run python -m rag.eval.run
uv run python -m rag.eval.trace <the same five ids>
```

*You should see* the `rerank` row move and the other three rows stay. On Homer, with the reranker's `POOL` raised from 20 to 50 so that fusion rank 21 to 50 is no longer a death sentence:

```text
rerank     0.1600    0.5933  0.2462 0.5513     (POOL = 20)
rerank     0.1760    0.6433  0.2684 0.6013     (POOL = 50)
```

Two questions recovered, g04 and g20: the reranker, once it could see them, put g04's chunk first and g20's fourth. The other three were in the pool of 50 and the reranker still put them at 9, 27 and 28, so for those the stage that lost them moved from fusion to the reranker, which is progress of a kind: a stage with a model in it can be swapped, and a chunk outside the pool could not be helped by anything. The cost: every query now scores fifty pairs instead of twenty, two and a half times the reranker's time, and `rag.ask` is slower by that much. Paste the before-and-after rows and the five traces under `## Step 4`, and keep the change if the table says to; on Homer it stays.

*If it broke* with no row changing, the constant you changed is not on the path for those five; the trace says which stage lost them, and that stage's constant is the one.

**Extension — `rag/POSTMORTEM.md`** *(assigned)*

One page, three parts:

1. **Five questions, one line each**, in the form `id: lost at <stage> because <mechanism>`, where the mechanism names the words: which query words BM25 matched, why the semantic model did not see it, what the reranker preferred instead. Not "it ranked low"; why.
2. **The one change**, what it recovered and what it cost, from the Step 4 table, with the sentence that says whether it is a real improvement or a fit to five questions you wrote yourself.
3. **The one you decided not to fix**, and why. On Homer that is g03: the only fix is a question written in Butler's words, which is fixing the test, or a synonym list for Circe, which is a fix for one question.

Paste the five lines, the change sentence and the not-fixed sentence under `## Extension` in `evidence/P21.md`.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/P21.md` (the questions are below) and run `uv run python scripts/check_evidence.py P21` until it says `all slots filled`. Then:

```bash
git add rag/ evidence/
git commit -m "P21: every miss traced to its stage, five explained, the reranker's pool widened"
git push -u origin dev/postmortem
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body is the Step 4 before-and-after rows.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`rag/POSTMORTEM.md` + `evidence/P21.md` with every slot filled, the Step 4 prediction committed before the change.

**Reflection Questions** (answer under `## Reflection` in `evidence/P21.md`)

1. Paste your stage table. Say which stage lost the most questions and what that stage cannot know about a chunk, using one of the five as the example.

   *How to get it:* fusion knows two ranks and nothing else. It cannot know that BM25's rank 4 for g04 was a chunk with the exact answer and semantic's rank 38 was the model never having read past the first 256 tokens; it adds `1/64` and `1/98` and moves on.

2. Paste one trace with and without expansion. Say what the expansion did to the ranks at each stage, and whether the model's added words are in the labeled chunk.

   *How to get it:* open the chunk. If the added words are in it, BM25's rank improves and that is a true fix; if they are the model's general knowledge of the subject and not Butler's words, the ranks move for other reasons or not at all.

3. Paste your before-and-after rows. Say what "fixing" a golden question by changing a constant risks, in terms of the test split from chapter 9, and what you would need to know the change is real.

   *How to get it:* chapter 9 said never to tune on the test set, and the golden set is your test set. A constant chosen because it recovers g04 and g20 is tuned on them. To know it is real you would need questions you did not look at while choosing, which means writing a second set, which is the first thing to do in January.
