# P22 · Ship It: One Command, a Clean Clone, the Practical

**Meetings:** D45 · **Points:** 10 pts

**Watch — none**

No Boot.dev today. The last day of the semester's applied track. Everything from P01 is in this repo: the chapter's preprocessing, TF-IDF and BM25 on your passages, an embedding engine, a chunker, rank fusion, query enhancement, a reranker, a golden set with its runner, cited answers behind a gate, and an agent that reaches for all of it. Today it gets a front page, one command that runs a question end to end, and a final proof that it all runs from a clean clone, then it is merged to `main`. The practical runs at the side of the room while you work.

**Notes**

**A README is for the stranger.** Someone who has never seen the repo should read the top of `README.md` and, in five minutes, know what is here, ask the corpus a question, and find the eval. The template's README told you how to set up; yours tells them what you built. Keep the setup section and add yours above it.

**One command is the whole pipeline.** `rag/ask.py` is twelve lines: hybrid search over your chunks, the cross-encoder on the pool, the gate, the cited answer, the citation check. Every one of those is a module you ran on its own day, and the stranger runs them all with one question.

**The clone needs two things the repo does not carry.** Your corpus, which is git-ignored and fetched by `scripts/fetch_corpus.sh`, and the key, which is in `.env` and must never be in a commit. A clean clone that runs is a clone where you put both back by hand, and the README says so.

**The practical is ten minutes and you have the pages.** The demo from `/tmp/ship-check`, `rag/MILESTONE.md` read aloud with its delta and its regressions, and one code-review question on a stage of the pipeline I pick. P21's `POSTMORTEM.md` is the best preparation for the third part, because it already names every stage and what it cannot know.

**Walkthrough — the front page, the final clone, the merge**

**Step 1. Merge, branch, log.**
Open the rag repo on GitHub and merge the P21 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch main && git pull
git switch -c dev/ship
```

*If P21 is not approved yet:* `git switch dev/postmortem && git switch -c dev/ship`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Step 2: README.md, the front page
- [ ] Step 3: the final clean clone, the runner, one question
- [ ] Step 4: PR, merged, branch deleted; the practical when called
- [ ] Fill every slot in evidence/P22.md (check_evidence.py: all slots filled); sign off
```

**Step 2. The front page.**

Open `README.md` at the repo root. Above the template's `## First run`, add a section with your name and the semester, and under it four short parts, in this order:

- **What this is.** Three sentences: the corpus (title, source, size in words from P02), what the search engines do on it (the scoreboard's best `hit@3` from P14 and the golden set's `recall@5` and `MRR` for the reranked pipeline from P21), and what the agent does with it (the milestone's headline from P20, with its two numbers).
- **Ask it a question.** The clone commands from Step 3 and one `rag.ask` line with a question your corpus answers, as a code block a stranger can paste, with the two lines above it that say where the corpus and the key go.
- **The eval.** One line pointing at `rag/eval/golden.json` and `rag/eval/run.py`, with the command, and one at `rag/MILESTONE.md`.
- **Where everything is.** A table with one row per module in `rag/` and the assignment it came from, in your own words, not the template's.

Half a page. No screenshots.

**Step 3. The final clone.**

```bash
cd /tmp
rm -rf ship-check
git clone https://github.com/Sierra-Canyon/hse-2026-2027-rag-<your-username>.git ship-check
cd ship-check
git switch dev/ship
cp ~/version_control/hse-2026-2027-rag-<your-username>/data/corpus.txt data/
cp ~/version_control/hse-2026-2027-rag-<your-username>/.env .
uv sync
uv run python -m rag.eval.check_golden
uv run python -m rag.eval.run
uv run python -m rag.ask "<the question your README names>"
```

*You should see* the golden set check pass, the four-row table with your P21 numbers, and a cited answer. The first two commands take a minute each the first time, because the clone has no `data/emb_chunks_*.npy` and no model cache of its own: the embeddings are rebuilt from the corpus, and the two models load from your home folder's cache. On Homer:

```text
chunking 200/50, 1882 chunks, 25 questions
kinds: {'ordinary': 9, 'exact': 5, 'paraphrase': 9, 'two-chunk': 2}
mean labeled chunks per question: 1.52
golden set is consistent with the chunking
n = 25 questions; one question moves recall by 0.040
method     prec@5  recall@5   F1     MRR
bm25       0.1360    0.4633  0.2037 0.3527
semantic   0.0800    0.2600  0.1200 0.2280
hybrid     0.1360    0.4933  0.2081 0.4547
rerank     0.1760    0.6433  0.2684 0.6013
highest precision@5 this set allows: 0.3040
```

and for `uv run python -m rag.ask "How many axes did Ulysses set up for the archery contest?"`:

```text
{{LLM:P22-ask}}
```

If every number matches your working copy's, the semester's work is reproducible from the repo plus the two files the README names. Paste the table and the answer under `## Step 3` in `evidence/P22.md`. Leave `/tmp/ship-check` where it is: it is the clone you demo from when your name is called.

*If it broke* at `uv sync` on `torch`, the committed `pyproject.toml` is missing what P07's `uv add` wrote; check `git status` in your working copy for an uncommitted `pyproject.toml`. *If it broke* with `golden.json` missing, P15's commit never reached `main`; the branch it is on is still open. *If it broke* with `OPENROUTER_API_KEY is not set`, the `cp .env` line was skipped, which is the point of the README saying it. Write under `## Step 3` what broke, if anything, and the commit that fixed it.

**Step 4. The last pull request.**

Fill `## Reflection` and `## Practical prep` in `evidence/P22.md` (the questions are below) and run `uv run python scripts/check_evidence.py P22` until it says `all slots filled`. Then:

```bash
git add README.md evidence/P22.md
git commit -m "P22: README front page, final clean clone, Semester 1 shipped"
git push -u origin dev/ship
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, title it **Semester 1: shipped**, **Create pull request**. The PR body is the three-sentence "What this is" from your README. I review it during the period; when it is approved, merge it, click **Delete branch**, and then:

```bash
git switch main && git pull
git branch --show-current   # main
git log --oneline | head -3
```

Paste the three commit lines under `## Step 4`. That is the semester, on `main`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**The practical.** When your name is called, bring the laptop with `/tmp/ship-check` open. Ten minutes: run the question (two minutes), read the milestone's headline and its two regressions from `rag/MILESTONE.md` (three minutes), and answer one question from me about one stage of `rag/ask.py`'s pipeline: what it takes in, what it hands on, and what it cannot know (five minutes). `POSTMORTEM.md` already has the answer for the stages that lost questions; prepare the others.

**Deliverable**
`README.md` with the front page + `evidence/P22.md` with every slot filled + the repo on `main` with every P01 to P22 branch merged and deleted + the Semester 1 Practical, ten minutes, oral.

**Reflection Questions** (answer under `## Reflection` in `evidence/P22.md`)

1. Paste your "What this is" paragraph. For each of the three numbers in it (the scoreboard's hit@3, the golden set's recall@5, the milestone's delta), say what question about your corpus it answers.

   *How to get it:* does the search find the passage for a query (five hand-picked queries); does the whole pipeline find the labeled chunk for a question (twenty-five labeled ones); does handing the pipeline to a model make its answers better than its memory (twenty questions, two modes). Three questions, three numbers, three sizes of evidence.

2. Paste your clone outputs. Name the one file in `rag/` that, if deleted, would break `rag.ask` but not `rag.eval.run`, and the one that would break both.

   *How to get it:* `rag/llm.py` is imported by `rag/answer.py`, which `rag.ask` needs and the runner does not; `rag/search.py` (or `chunking.py` under it) is under both. Read the imports at the top of each file; the answer is there.

3. Of the twenty-two assignments, name the one whose idea you would be least able to explain to a classmate without opening the repo, and the one you could explain from memory, with one sentence on why each.

   *How to get it:* no hint; this is the question I ask at the end of the practical.
