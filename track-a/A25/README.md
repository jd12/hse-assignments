# A25 · FLOAT: Catch-Up, Repair, and Retrieval Postmortem

**Meetings:** D44–D45 · **Points:** 15 pts

**Watch**
Day 1 — no video.
Day 2 — no video. Postmortem discussion in class.

Re-watch any lesson from A17–A21 that did not land, and only the part you need.

**During the video**

No video. Instead, before anything else on Day 1, build your repair list: every assignment from A13 to A22, what is still open, and what my review said. Step 2 gives you the commands and the table.

**Notes**

The unit should not end with anyone carrying invisible debt into frameworks. Nothing new is assigned. Three things are on the table.

**Finish what is unfinished.** Any Boot.dev chapter still red from A13–A16, any DeepLearning.AI lesson you did not type along with, any deliverable still missing.

**Resubmit.** Any assignment A13–A22 can be resubmitted for a regrade, once, within two weeks of its due date. Fix the thing the review named. Push the fix to the **same branch** if its pull request is still open; do not open a new one. If it has already merged, the fix gets its own branch and pull request, and Step 1 shows you both.

**Come ready for the postmortem on Wed Dec 9.** Everyone takes part, including anyone fully caught up. It is a conversation, not a document, and it is ungraded. It is not a summary of what RAG is. It is an account of *your* pipeline's real weaknesses, said out loud with a number attached, because in January frameworks start taking over parts of your architecture and you should know what you are handing over. The questions are printed below so you can think before you talk.

The 15 points are for the repair work. A student who arrives fully caught up earns them by moving one Milestone A1 number and saying honestly whether it moved.

**Walkthrough — repair**

**Step 1. Merge, branch, log.**
Open the rag repo and your v2 agent repo on GitHub and merge every pull request I have approved; click **Delete branch** on each. For each assignment you are repairing:

If its pull request is still open, work on that branch:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch main && git pull
git switch dev/<that assignment's branch> && git pull
```

If it has merged, open a repair branch from `main`:

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
git switch main && git pull
git switch -c dev/resubmit-a<NN>
```

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Repair list built (Step 2)
- [ ] Clean-clone check on the rag repo (Step 3)
- [ ] Repair 1: <assignment, what the review named>
- [ ] Repair 2: <assignment, what the review named>
- [ ] `REPAIR.md` rows for today
- [ ] Push, sign off
```

**Step 2. The repair list.**

```bash
cd ~/version_control/hse-2026-2027-rag-<your-username>
gh pr list --state all --limit 20
gh pr view <number> --comments
```

If `gh` is not set up, open **Pull requests** on the repo page and clear the `is:open` filter. Do the same in your v2 agent repo for A20. Then, in your log entry, a table:

| Assignment | Branch | PR state | What my review named | Deliverable missing | Plan |
|---|---|---|---|---|---|

One row for each of A13, A14, A15, A16, A17, A18, A19, A20, A21, A22. A row with nothing in the last three columns is a row you are done with. *You should see* at least one row with something in it; if every row is clean, skip to Step 5.

**Step 3. The clean-clone check.** A repo that only runs on your laptop is a Correctness defect, and this is the week to find out. Clone the rag repo fresh, somewhere outside your work:

```bash
cd ~
rm -rf check-rag && git clone https://github.com/Sierra-Canyon/hse-2026-2027-rag-<your-username>.git check-rag
cd check-rag
./setup.sh
bash scripts/fetch_corpus.sh
shasum -a 256 data/corpus.txt
uv run python -m rag.eval.check_golden
uv run python -m rag.eval.run
```

This clone has no `data/` except what `fetch_corpus.sh` makes, so it re-embeds your chunks once. That costs a little and it is the point: it proves someone else could reproduce your numbers.

*You should see* the sha256 from `data/SOURCE.md`, `check_golden` pass, and a `run.py` table whose `bm25` row is identical to the one in `rag/eval/RESULTS.md` and whose other rows are within one question of it. If anything fails, that failure is Repair 1: fix it in your real repo, not in `check-rag`. Delete `~/check-rag` when it passes.

*If it broke:* `ModuleNotFoundError` means a package you installed was never committed to `pyproject.toml`; `uv add` it in the real repo. `FileNotFoundError` on something in `data/` means a script depends on a file only your laptop has.

**Step 4. Repair, and say what changed.** For each row on your list, fix the thing, commit with a message that names the assignment, push, and add one line to that pull request's description:

```
Resubmission: <what the review named> → <what I changed> (<short commit hash>)
```

One line per fix. "Fixed stuff" is not a changelog. Then comment `@jd12 ready for regrade` on the pull request, so it comes back to the top of my list.

**Extension — the repair ledger (ASSIGNED)**

**Part 1. `REPAIR.md`** at the top of the rag repo, committed on whichever branch you finish on. One row per repair:

| Assignment | What the review named | What changed | Number before | Number after | Commit |
|---|---|---|---|---|---|

"Number" is whatever the repair moved: a scoreboard total, a recall@5, a count of invalid citations, a policy count out of twelve, a test that now passes. A missing deliverable that now exists is "absent → present". At least one row has a real number on both sides.

**Part 2. Where it got worse.** Fixes break things. For your biggest repair, re-run the scoreboard or `rag.eval.run` after it and compare every row to before. Name one number that went down, even by one question, or show the full before-and-after table proving none did.

**Step 5. If you are fully caught up: move one Milestone A1 number.** Pick one cell from the `both` table in `evals/grounded/REPORT.md`. In `REPAIR.md`, before you change anything, write the cell, its current mean and range, the one change you will make (a tool description, `REFUSE_BELOW`, the pool size, the chunk count per result), and the value you predict. Commit that. Then make the change, re-run `both` three times into new files (`both-r1.jsonl` to `both-r3.jsonl`, after moving the old runs into `runs/before/`), re-run `score.ts`, and write the new mean and range under your prediction. Say whether the difference is bigger than the range. "It did not move" is a full-credit answer if the numbers show it.

```bash
git add REPAIR.md
git commit -m "A25: repair ledger"
git push
```

One pull request per branch; push again each day. Then close the log: `bash scripts/sign-off.sh`, and `git add logs && git commit && git push` in the log repo. On Day 2, one more log entry: which discussion question you answered out loud, and the number you brought to it.

**Discussion · Wed Dec 9, in class**

Ungraded, nothing to hand in. Bring your evidence open on your laptop: the eval numbers, the scoreboard, the commit you are not proud of. An answer with a number attached beats an answer without one.

1. Which single component of your pipeline is weakest right now (preprocessing, chunking, retrieval, reranking, generation, or the agent's routing), and what number points there?
2. What is the most embarrassing bug you shipped between A13 and A22, and what was its root cause?
3. Which failure took you longest to find, and what made it hard to see? What would have caught it in five minutes?
4. Name one thing you built that you now think was unnecessary complexity.
5. Name one thing you skipped that you now think you should have built.
6. Your golden set is twenty questions. Would it catch a regression you introduce next month? What is missing from it?
7. How confident are you that your Milestone A1 golden-set accuracy would hold on 200 fresh questions about your corpus? Why?
8. If a stranger used your grounded agent tomorrow, what is the first thing that would go wrong for them?
9. Which decision from Milestone A1 do you most want to revisit, now that memory sits on top of it?
10. Going into frameworks: one part of your hand-built pipeline you would be relieved to hand to a library, and one you would refuse to.

**Deliverable**
Each resubmission pushed with its one-line changelog in the pull request description + `REPAIR.md` (every repair with its before and after, and the where-it-got-worse check) + anything unfinished from A13–A22 closed out. Fully caught up: `REPAIR.md` with the committed prediction and the three-run before and after for one Milestone A1 number.

**Reflection Questions**

1. Paste your Step 2 repair table as it stood on Day 1, and the rows of `REPAIR.md`. Which row took the longest, and was the review comment it answered about the code, the measurement, or the writing? Paste the comment.
2. Paste the output of your clean-clone check, or the error it stopped on and the commit that fixed it. If the `run.py` table differed from `RESULTS.md` by more than one question on any row, name the row and say what in your pipeline is not reproducible.
3. Paste your Part 2 comparison, or your Step 5 prediction with its before and after. Did the thing you fixed or changed move the number you expected it to? If a different number moved, name it and say why that one.
