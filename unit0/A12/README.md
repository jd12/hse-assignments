# A12 · Evaluating AI Agents 07–14 + Unit 0 Capstone & Track Election

**Meetings:** D24–D26 · **Points:** 15 pts

**Watch — 75 min**

**Day 1 — 29 min**
[Evaluating AI Agents, DeepLearning.AI / Arize](https://www.deeplearning.ai/short-courses/evaluating-ai-agents/) · lesson 7, Adding router and skill evaluations (12m) · lesson 8, Lab 3: Adding router and skill evaluations (17m, notebook)
 · A full section, over 25 minutes to finish it.

**Day 2 — 14 min**
Same course · lesson 9, Adding trajectory evaluations (5m) · lesson 10, Lab 4: Adding trajectory evaluations (9m, notebook)
 · The rest of the period is capstone build: the judge.

**Day 3 — 32 min**
Same course · lesson 11, Adding structure to your evaluations (7m) · lesson 12, Lab 5: Adding structure to your evaluations (15m, notebook) · lesson 13, Improving your LLM-as-a-judge (4m) · lesson 14, Monitoring agents (6m)
 · Lesson 15, the conclusion, is one minute and optional. The rest of the period is the measured improvement and the track election.

**During the video**

The notebooks are the code-along again: run every cell in the DLAI page as he does. Then the walkthrough runs the same evaluations against your own agent, on your own 25 questions, with the scripts you copied in A11.

**Lesson 7 and Lab 3 · find the ground truth.** When the lab builds its router eval, write in your log where its correct answers come from and who wrote them. Yours come from `expected_tool` in `evals/questions.jsonl`, which you wrote in A11. When it builds a skill eval, write down whether a person, a rule or a model decides pass and fail.

**Lesson 9 and Lab 4 · write down the division.** When the lab scores a trajectory, pause and write exactly what it divides by what, and where the "best" path length comes from. Step 5 does the same thing with your `min_steps`.

**Lesson 11 and Lab 5 · what stays fixed.** When the lab compares two experiments, write down everything it holds the same between them. The measured improvement at the end of this assignment is worth nothing if any of those move.

**Lesson 13 · list the fixes.** Write down each technique he gives for improving a judge. Step 8 uses exactly one of them, and you name which.

**Lesson 14 · one monitor.** Write one number about your agent you would want on a dashboard in production, and which span attribute in your `spans.jsonl` it would come from.

**Notes**

**Labels before the judge. The order is not negotiable.** Write a one-line pass rule, label the 25 runs by hand, commit the labels, and only then run the judge. A judge that has run first pulls your labels toward it without your meaning it to, and the agreement number becomes a measurement of nothing. `judge.py` has been in your repo since A11, so the order I check is the labels commit against the first commit that adds `evals/judge_out/`. Labels committed after the judge's first output cap Evidence at 3.

**A judge that passes everything is the default failure.** If 20 of your 25 are passes, a judge that says "pass" to everything agrees with you 80% of the time and is useless. Overall agreement hides it. `agreement.py` reports the two per-class numbers: of the runs you failed, how many did the judge pass; of the runs you passed, how many did it fail.

**The judge sees the quote.** Every question has the passage from your corpus that answers it. Without it, the judge is grading whether an answer sounds right.

**You are tuning the judge on the set you measure it on.** Twenty-five is all you have. Change the judge prompt once, not five times, and say in the evidence that the after-number is optimistic for that reason.

**One run is not a measurement.** Your agent samples, so the same question can go two ways. Before and after are three runs each, and the spread goes in the table.

**Spend.** One agent run is several model calls. Six runs of 25 is 150 agent runs, plus the judge on each. Check the remaining balance on your key before Day 3; it is a hard cap and it does not warn you.

**Walkthrough — router, skill and trajectory evals, a calibrated judge, one measured improvement**

**Step 1. Merge, branch, log.**
Open the foundations repo on GitHub and merge the pull request for A10 if I have approved it, and A11b's if you have not already; click **Delete branch** on each. A11's two branches stay open and this work continues on them:

```bash
cd ~/version_control/<your-v2-agent-repo>
git switch dev/instrumentation && git pull
grep -n 'evals/' .gitignore          # prints nothing: A11 Step 2 removed evals/runs/ and evals/traces/; if a line prints, delete it now
git branch --show-current     # dev/instrumentation, not main
cd ~/version_control/hse-2026-2027-foundations-<your-username>
git switch dev/agent-evals && git pull
ls evidence/A12.md
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist. Day 1:

```markdown
- [ ] Lesson 7 and Lab 3, ground truth written down
- [ ] Re-read evals/questions.jsonl and STEPS.md before anything else
- [ ] Step 2, router and skill numbers for base-1 in evidence/A12.md
- [ ] Steps 3–4, pass rule, 25 hand labels, committed alone
```

Day 2: lessons 9–10 and Lab 4, Step 5 (trajectory), Steps 6–7 (judge and agreement). Day 3: lessons 11–14 and Lab 5, Step 8 (one judge fix), the extension, the track election, `check_evidence.py A12`, and closing the Unit 0 log.

Three meetings, the same two branches and the same two pull requests as A11. Push again each day; one log entry per meeting.

**Step 2. Router and skill, as two numbers (Day 1).**

A week of fall break is long enough that your own questions stop being obvious. Read `evals/questions.jsonl` and `evals/STEPS.md` before you write anything. Then, in the v2 repo:

```bash
uv run --no-project python evals/evals.py base-1
```

*You should see* two lines: `base-1: router N/25  skill M/K`, then `repeats R  convergence C.CC`, which is Step 5's line; read only the first for now. Router asks whether the first tool your agent reached for is the one you said it should. Skill asks, only for the runs where it was, whether any tool call actually brought back the passage that holds the answer: the first fifty characters of your `quote`, looked for in every tool's recorded output. The no-answer questions have no passage, so they have no skill score and `K` is at most 22. The two fractions will not be the same. A low router number is fixed in tool descriptions and the system prompt. A low skill number is fixed in the tool itself: a search pattern that is too literal, a read that stops before the answer, an output cut at 20,000 characters. Paste both lines under `## Step 2` in `evidence/A12.md` and say which number is lower and what that points at.

**Step 3. Write the pass rule.**

In the v2 repo, create `evals/judge_labels.json` with the rule and nothing else yet:

```json
{"rule": "PASS if ... ; otherwise FAIL.", "labels": []}
```

One sentence, precise enough that a stranger applying it to your 25 would agree with you. It must say something about the answer and something about the path. For the no-answer questions it must say what a passing answer looks like. Copy the sentence into `## Prediction` in `evidence/A12.md`.

**Step 4. Label the 25 by hand, and commit them alone.**

For each run in `evals/runs/base-1.jsonl`, print its tree with `uv run --no-project python evals/show_trace.py <trace_id>`, read the answer against the quote, and add a label to the `labels` list:

```json
{"id": "e07", "run": "base-1", "label": "fail", "why": "right name, but searched the whole file three times first"}
```

*You should see*, counting your labels, some of each. If all 25 are passes or all are fails, your questions were too easy or too hard to calibrate anything, and the judge will say nothing. Your A11 no-answer and two-passage questions are where the fails should be. Fill the counts line under `## Prediction` in `evidence/A12.md`, then commit the labels on their own:

```bash
git add evals/judge_labels.json
git commit -m "A12: pass rule and 25 hand labels, before the judge runs"
git push
```

I will check your commit timestamps: this commit has to be older than the first one that adds `evals/judge_out/`.

**Step 5. The trajectory eval (Day 2).**

Open `evals/evals.py` and read `trajectory()`: `repeats` counts tool calls that exactly repeat an earlier call in the same run (same tool, same arguments), and `convergence` is your `min_steps` divided by the calls it took, capped at 1, so `1.00` means every run was as short as you said it could be. The second line Step 2 printed is their totals. Now the rows:

```bash
uv run --no-project python evals/evals.py base-1 --rows
```

*You should see* a 25-row table, one row per question, with `router`, `skill`, `calls`, `repeats` and `convergence`, then the same two summary lines. Find a run you labeled **pass** whose convergence is under `0.50` or whose repeats are above `0`. That is a correct answer reached by a bad path, and a final-answer eval cannot see it. Paste its row under `## Step 5` in `evidence/A12.md`, print its tree, and say what the agent did along the way.

**Step 6. The judge.**

Open `evals/judge.py` and read `PROMPT`: the judge gets your rule, the question, the passage, the expected answer, the tool calls in order, and the final answer, and returns `{"label": ..., "reason": ...}` as JSON. It reads your rule and never your labels. `MODEL` is the model from A09 and A10; change it only if that one refuses `response_format` or `temperature`.

```bash
uv run --no-project python evals/judge.py base-1
```

*You should see* 25 lines, each an id, `pass` or `fail`, and the start of a reason, and `evals/judge_out/base-1.jsonl` with 25 lines. Commit it now, so the order is on record:

```bash
git add evals/judge_out/
git commit -m "A12: judge v1 on base-1"
git log --oneline -- evals/judge_labels.json evals/judge_out/
```

*You should see* two commits, the labels one below (older than) the judge one. Paste those lines in the `git log` fence under `## Prediction` in `evidence/A12.md`.

**Step 7. Agreement, per class.**

```bash
uv run --no-project python evals/agreement.py base-1
```

*You should see* three fractions and a list of ids: overall agreement, then `you FAIL, judge PASS (false pass)` out of the runs you failed, then `you PASS, judge FAIL (false fail)` out of the runs you passed. Paste the four lines under `## Step 7` in `evidence/A12.md`. Then, for every id in the disagree list, decide who was right, you or the judge, and write one line each. Some of them will be the judge. That is not a failure of the assignment; it is the assignment.

**Step 8. Fix the judge once (Day 3).**

Pick one technique from lesson 13 and apply it to `PROMPT` or `RULE`, once. Save the old output first so the comparison survives:

```bash
cp evals/judge_out/base-1.jsonl evals/judge_out/base-1.v1.jsonl
uv run --no-project python evals/judge.py base-1
uv run --no-project python evals/agreement.py base-1
```

Paste the new four lines under `## Step 8` in `evidence/A12.md`, name the technique, and say which per-class number moved and which way. If the false-pass rate went up, the fix made the judge more lenient; say so. Do not change your labels to match the judge. If you now think a label was wrong, say which and why, and leave it.

**Extension — one improvement, measured (ASSIGNED)**

**X1. Baseline, three runs.** You have `base-1`. Make two more, unchanged:

```bash
uv run --no-project python evals/run_all.py base-2
uv run --no-project python evals/run_all.py base-3
```

**X2. Choose one change and predict.** Choose the one change your Step 2 and Step 5 numbers point at: a tool description (router), a tool's search behavior (skill), or a stopping instruction for the no-answer case (trajectory). One change. Fill the improvement line under `## Prediction` in `evidence/A12.md`: what you are changing, which of your four numbers (router, skill, convergence, judge pass rate) it should move and by how much, and which one it might make worse. Commit both repos before you touch the agent:

```bash
git add evals/
git commit -m "A12: baseline x3, judge v2, predicted effect of one change, before the change"
cd ~/version_control/hse-2026-2027-foundations-<your-username>
git add evidence/A12.md
git commit -m "A12: prediction before the change"
cd ~/version_control/<your-v2-agent-repo>
```

**X3. Make the change, then three more runs.**

```bash
git add -u && git commit -m "A12: <the change, in words>"
for i in 1 2 3; do uv run --no-project python evals/run_all.py fix-$i; done
for t in base-2 base-3 fix-1 fix-2 fix-3; do uv run --no-project python evals/judge.py $t; done
uv run --no-project python evals/evals.py base-1 base-2 base-3 fix-1 fix-2 fix-3
```

*You should see* two lines per run, six runs. The spans file keeps growing; every eval looks runs up by trace id, so old runs stay readable. Do not change `PREFIX`, the questions, the model, the judge or the rule between the first `base` run and the last `fix` run.

**X4. The table.** Fill the six-row table under `## Extension X4` in `evidence/A12.md` from those lines, with the judge-pass column counted from each run's `evals/judge_out/<tag>.jsonl` (`grep -c '"pass"' evals/judge_out/fix-1.jsonl`), all from judge v2. Then the two mean rows, each with the min and max in parentheses, and the paragraph: is the difference between the means bigger than the spread inside each group? Which number got worse? If the change made things worse, the paragraph says so, and a clean negative result gets full credit.

**X5. Commit, push.** Fill `## Reflection` in `evidence/A12.md` (the questions are below) and run `uv run python scripts/check_evidence.py A12` in the foundations repo until it says `all slots filled`. Then:

```bash
git add evals/ && git commit -m "A12: before/after x3 each, one change, judge v2"
git push
cd ~/version_control/hse-2026-2027-foundations-<your-username>
git add evidence/A12.md && git commit -m "A12: evidence" && git push
```

Both pull requests from A11 are already open; these commits land on them.

**Step 9. The track election.**

It goes in the foundations repo on its own branch, because it is course business rather than agent work:

```bash
cd ~/version_control/hse-2026-2027-foundations-<your-username>
git switch main && git pull
git switch -c dev/track-election
```

Create `TRACK_ELECTION.md`. Rank both, 1 and 2; you may not get your first choice.

**Track A, applied.** Production AI engineering: retrieval, memory, the Model Context Protocol, durable execution, orchestration, deployment. Your corpus becomes the thing your agent retrieves from.

**Track B, theory.** LLM internals: linear algebra, autograd from scratch, attention derived rather than described, and a GPT built from an empty file. Your corpus becomes what that GPT trains on.

One paragraph, based on evidence from these eight weeks rather than on which sounds better. Name the assignment from A01 through A12 you lost track of time on, with the file in your repo that shows it, and one thing about the track you did not rank first that you will miss.

```bash
git add TRACK_ELECTION.md
git commit -m "A12: track election"
git push -u origin dev/track-election
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop.

**Step 10. Close the Unit 0 log.**

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

Then, on GitHub, merge the `<your-username>-unit0` pull request and delete the branch. Eight weeks of entries go in at once. Your track branch opens at A13 or B13.

The v2 pull request, `dev/agent-evals` and `dev/track-election` merge when the work is finished and I have approved them.

**Deliverable**
In the v2 repo: `evals/judge_labels.json` (committed before the first `evals/judge_out/`) + `evals/judge.py` with its one Step 8 change + `evals/runs/` (six runs) + `evals/judge_out/` (seven files, `base-1.v1.jsonl` among them). In the foundations repo: `evidence/A12.md` with every slot filled (the Step 2 lines, the rule, the label counts, the commit order, the predicted change committed before the change, the Step 5 row, judge v1 and v2 per class, the six-row table with its two mean rows and the paragraph, the three reflection answers) + `TRACK_ELECTION.md`.

**Reflection Questions** (answer under `## Reflection` in `evidence/A12.md`)

1. Paste `git log --oneline -- evals/judge_labels.json evals/judge_out/`. Then give the three fractions from `agreement.py` for judge v1 and v2, and say which class is worse and whether your one fix helped it or only helped overall agreement.

   *How to get it:* the log is already pasted under `## Prediction`; the six fractions are under `## Step 7` and `## Step 8`. The worse class is the larger of the two per-class fractions; the fix helped it if that fraction fell from v1 to v2, and helped only overall agreement if overall rose while that fraction did not fall.

2. Pick one id where you and the judge disagreed. Paste the question, your label and `why`, and the judge's `reason`. Who was right? If the judge was, quote the part of your pass rule that did not cover the case. Then give one run you labeled **pass** that `evals.py` scores badly on repeats or convergence, and say what the agent did along the way.

   *How to get it:* the ids are the disagree list under `## Step 7`; `grep e07 evals/judge_labels.json evals/judge_out/base-1.jsonl` prints your line and the judge's for one id. The bad-path pass run is the row you pasted under `## Step 5`.

3. From your improvement table: the base mean and range and the fix mean and range for the number you predicted would move, next to the prediction in the commit before the change (give its hash). Is the difference bigger than the spread? Name the number that got worse, and the entry in your A10 catalog that this change addresses or does not touch.

   *How to get it:* the means and ranges are the last two rows of the table under `## Extension X4`; the prediction is the improvement line under `## Prediction`; the hash is `git log --oneline --grep="before the change" -- evals/ | tail -1` in the v2 repo. "Bigger than the spread" means the two ranges do not overlap. The A10 entry is in `evidence/A10.md` in the foundations repo: its number and four-word title.
