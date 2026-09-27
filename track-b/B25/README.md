# B25 · Float Week (Math Clinic and Resubmission)

**Meetings:** D40 · **Points:** 15 pts



**Notes**

No video today. This meeting sits between the backward pass and Milestone B1 on purpose: whatever did not land in linear algebra or the chain rule gets fixed here, because there is no room for a gap in `.backward()` once the milestone opens tomorrow.

Being behind here is normal. Being *quietly* behind is the only real failure mode.

Everybody takes the diagnostic in Step 2, cold, and commits it before looking anything up. It takes about twenty minutes and it tells you which option in Step 5 is yours. A cold answer that is wrong is worth more to you than a looked-up answer that is right, and the commit is how I know which one I am reading. I will check your commit timestamps.

"Cold" means no notes, no repo, no video, and no AI. Write "don't know" where you don't know. That line is an answer, and it is the most useful one on the page.

Do not edit a cold answer after you commit it, even to fix a typo. The corrections go underneath, in a separate section, so both attempts are visible.

Come to a check-in today if any of these is true: your micrograd does not match PyTorch to six decimals; you cannot explain what a gradient is without reading your notes; you still have to think hard about which dimension "vanishes" in a matmul.

If you are fully caught up, do the diagnostic, choose option (b), and then rest. There is no extra credit for grinding through a holiday week.

**Walkthrough — Diagnose, then repair**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Then:
```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/clinic
```
*If B21 is not approved yet:* leave its pull request open and branch from `main` anyway. Nothing today builds on B21's new files, and tomorrow's milestone branches from whichever is newer.
```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # should print <your-username>-track, not main
bash scripts/start-entry.sh
```
Under the timestamp, write today's checklist:
```markdown
- [ ] Step 2: the diagnostic, cold, committed before I open anything
- [ ] Step 3: mark it against my own repo, question by question
- [ ] Step 4: corrections under each wrong answer, cold answers untouched
- [ ] Step 5: option (a) two resubmissions, or option (b) the corrected diagnostic
- [ ] Extension: prove one repair with a number
- [ ] Push dev/clinic and open the PR
```

**Step 2. The diagnostic, cold.**

Close everything except a terminal and your editor. Create `clinic/DIAGNOSTIC.md` with this in it, and answer under each question. Twenty minutes. No notes, no repo, no AI.

```markdown
# B25 diagnostic — cold

1. What are the columns of a transformation matrix?
2. `(4×7) @ (7×2)` produces what shape? What if you reverse the order?
3. What does a determinant of 0 mean, in two different vocabularies?
4. What is `∂f/∂y` for `f = x²y + y³`?
5. Write the gradient descent update rule from memory.
6. What happens when the learning rate is too large?
7. Why must `_backward` use `+=`?
8. In what order does `.backward()` visit nodes, and why that order?
9. Write the softmax formula, including the numerical stability fix.
10. Cross-entropy loss for a correct-class probability of 0.25: what is it?
11. Which single topic from B13–B21 would you most want re-explained?
```

Questions 9 and 10 reach back to Unit 0 and forward to B23. "Don't know" is allowed on both. Question 11 is worth more than getting the other ten right, so answer it honestly.

When the twenty minutes are up, commit immediately:

```bash
git add clinic/DIAGNOSTIC.md
git commit -m "B25: diagnostic, cold"
git push -u origin dev/clinic
```

Open the pull request now, **jd12** as reviewer.

**Step 3. Mark it against your own repo.**

Now open everything. For each question, find the place in *your own* repository that answers it, and mark your cold answer right, partly right, or wrong. The table says where to look first:

| Q | Look here first |
|---|---|
| 1 | `linear/landing.py` output: the `lands at` lines against your matrices |
| 2 | `linear/linalg.py`: the `ValueError` message in `matmul`, and `test_non_square` |
| 3 | `linear/FINDINGS-b16.md` and your B13 singular system |
| 4 | Do it by nudging, as in `calc/partials.py`, at a point you choose |
| 5 | `calc/gradient_descent.py`, the two lines inside `descend` |
| 6 | `calc/diverge.py` output and `calc/FINDINGS-b18.md` |
| 7 | `micrograd/test_engine.py::test_reuse`, and B20's `b = a + a` |
| 8 | `micrograd/engine.py`, `backward`, and `micrograd/topo_check.py` output |
| 9 | Your foundations repo, A07–A09 |
| 10 | `uv run python -c "import math; print(-math.log(0.25), -math.log2(0.25))"` |

Add a line at the top of `DIAGNOSTIC.md`, above the questions: `Cold score: N right, M partly right, K wrong.` That line is the baseline for the extension.

**Step 4. Corrections, underneath.**

Add a section at the bottom of `DIAGNOSTIC.md` headed `## Corrected`. For every question you did not get fully right, write the corrected answer, and under it, one line saying what you had wrong and where in your repo you found the right answer: a file and line, or a printed number. For question 10, give it in both nats and bits, and say which unit your cold answer was in, if it was in either.

Do not touch the cold answers above. Commit: `git commit -am "B25: diagnostic corrected"`.

**Step 5. Choose your option.**

| Option | Who it is for | What you hand in |
|---|---|---|
| **(a) Resubmit** | You scored 7 or fewer on questions 1–10, or you have an assignment from B13–B21 you know does not work | Up to two prior assignments from B13–B21, fixed, plus `clinic/RESUBMIT.md` |
| **(b) Corrected diagnostic** | You scored 8 or more, or everything B13–B21 works | Steps 2–4 are the deliverable; go straight to the extension |

For option (a): do the fixes on this branch, in the original files. In `clinic/RESUBMIT.md`, for each resubmitted assignment write one line per change: the file, what was wrong, what you changed, and the commit that changed it. Resubmissions replace the original score. A resubmission with no `RESUBMIT.md` entry is invisible to me, because I have no way to know which lines to reread.

**Extension — Prove one repair with a number** *(choose one; write which and why at the top of `clinic/REPAIR.md`)*

A corrected answer written in prose is a claim. Pick the question you got most wrong, or the resubmitted assignment you changed most, and prove the correction with a script in `clinic/repair.py` that prints a number your cold answer would have got wrong. Every option has a baseline (your cold answer, or the old version of your file) and a comparison (what the code prints now).

| Option | For a miss on | What `clinic/repair.py` does | The number |
|---|---|---|---|
| **1. Nudge it** | Q4 or Q1 | For Q4, nudge `y` in `f = x²y + y³` at three points you choose and compare with your corrected formula; for Q1, multiply your own B15 matrices by `[1, 0]` and `[0, 1]` | The largest gap between formula and nudge, or between column and landing point |
| **2. Break it** | Q6, Q7 or Q8 | Re-run the failure the answer is about on your own code: a learning rate past your B18 edge, an `=` in one `_backward`, or `topo.append` before the loop | The wrong number next to the right one, from your own files |
| **3. Measure it** | Q9 or Q10 | Write softmax without the stability fix and with it; feed both `[1000, 1001, 1002]` and then `[1, 2, 3]`; print cross-entropy for your corrected Q10 probability in nats and bits | Where the naive version fails, and by how much |
| **4. Re-run it** | Option (a) | Re-run the resubmitted assignment's main script before and after your fix (`git stash` or `git show <old-commit>:<file>` for the old version) | The before and after output, side by side |

Whichever you choose, `REPAIR.md` gets: which option and why; the script's output; your cold answer, copied exactly; and one sentence on what, specifically, you believed that was wrong. Then say where you are still not sure. A repair that leaves you with no remaining doubt was probably too easy a choice; say so if that is what happened.

Commit, push, sign off:

```bash
git add clinic/
git commit -m "B25: repair proven with <option>"
git push
```

Say in the PR body where this is weakest, and whether you want a check-in before the milestone.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`clinic/DIAGNOSTIC.md` (cold answers committed first, cold score line, `## Corrected` section) + either option (a) up to two resubmitted assignments with `clinic/RESUBMIT.md` or option (b) the corrected diagnostic alone + `clinic/repair.py` + `clinic/REPAIR.md`.

**Reflection Questions**

1. Paste your cold score line and the commit hashes and times of `B25: diagnostic, cold` and `B25: diagnostic corrected`. Name the question you were most confident about and got wrong, or if there was none, the one you were least confident about and got right. Quote your cold answer and say what in your own repo corrected it, by file and line or printed number.

2. Paste the output of `clinic/repair.py` and your cold answer to the question it proves. Say what the script's number would have been if your cold answer had been true, and how far that is from what it printed.

3. Paste your cold answer to question 11. Now that you have marked the other ten, say whether that is still the topic you would pick, and point to the assignment and the file where you first went wrong on it. Then say what you will do differently in the milestone because of it, in one concrete sentence about code you will write or check.
