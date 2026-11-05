# T20 · Ship It: a Clean Clone, a README, One Command, and the Semester 1 Practical

**Meetings:** D45 · **Points:** 15 pts

**Watch — none**
 · The gpt repo, `README.md` at its root, `evidence/T20.md`, terminal in the repo root, and a second terminal for the clone. The practical runs at the side of the room while you work.

No video. The last day of the semester's theory track. Everything from T01 is in this repo: the linear algebra scripts on your chapters, an autograd engine you typed from nothing, a network it trained, and a name model in PyTorch trained on names mined from your corpus. Today it gets a front page, a one-command demo, and a final proof that it all runs from a clean clone, then it is merged to `main` and the branch is deleted.

**Notes**

**A README is for the stranger.** Someone who has never seen the repo should be able to read the top of `README.md` and, in five minutes, know what is here, run the name model, and find the engine. The template's README told you how to set up; yours tells them what you built. Keep the setup section and add yours above it.

**Ship means `main`.** Every assignment since T01 landed on `main` through a pull request. Today's PR is the last, and after it is merged every piece of the semester is reachable from one branch at one commit, which is what the spring picks up from.

**The practical is ten minutes and you have rehearsed it.** `names/PRACTICAL.md` is on the desk, the T21 page beside it, the demo seed decided. When your name is called, bring the laptop with `/tmp/practical-check` already cloned from `main`.

**Walkthrough — the front page, the final clone, the merge**

**Step 1. Merge, branch, log.**
Open the gpt repo on GitHub and merge the T22 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/ship
```

*If T22 is not approved yet:* `git switch dev/practical && git switch -c dev/ship`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-track, not main
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Step 2: README.md, the front page
- [ ] Step 3: the final clean clone, tests, training, demo
- [ ] Step 4: PR, merged, branch deleted; the practical when called
- [ ] Fill every slot in evidence/T20.md (check_evidence.py: all slots filled); sign off
```

**Step 2. The front page.**

Open `README.md` at the repo root. Above the template's `## First run`, add a section with your name and the semester, and under it four short parts, in this order:

- **What this is.** Three sentences: the corpus (title, source, size in words from T01), what the linear algebra scripts measured on it (T06's most-alike pair in the ranks 51-250 band), and what the two models learned from it (T12's final count out of 40, T18's three losses).
- **Run the name model.** The three clean-clone commands from T22 Step 2 and the demo line with your seed, as one code block a stranger can paste.
- **The engine.** One sentence pointing at `micrograd/engine.py` and `micrograd/MILESTONE.md`, with the five tests: `uv run pytest micrograd -q`.
- **Where everything is.** A table with one row per folder, `linear/`, `micrograd/`, `names/`, `evidence/`, `scratch/`, and the assignments each covers, in your own words, not the template's.

Half a page. No screenshots; the pictures are in the folders and the table says so.

**Step 3. The final clone.**

```bash
cd /tmp
rm -rf ship-check
git clone https://github.com/Sierra-Canyon/hse-2026-2027-gpt-<your-username>.git ship-check
cd ship-check
git switch dev/ship
cp ~/version_control/hse-2026-2027-gpt-<your-username>/data/corpus.txt data/
uv sync
uv run pytest micrograd -q
uv run python micrograd/train.py | tail -8
uv run python names/demo.py 7
```

*You should see* `5 passed`, T12's final loss and misses, and T22's ten names. On Homer:

```text
5 passed in 0.03s
after training: loss 18.1400   correct 35/40
misses (chapter, start, x1, x2, label, prediction):
   chapter 1  start     0   x1   0.21  x2  -1.64   label -1   predicted +0.309
   ...
   chapter 6  start  8814   x1  -1.80  x2  -0.41   label +1   predicted -0.922
drawn: micrograd/loss.png
name model trained on 508 names mined from data/corpus.txt   seed 7
   phoconisus
   ...
```

If every number matches your working copy's, the semester's work is reproducible from the repo alone plus the corpus. Paste the `5 passed` line, the `after training` line and the first three names under `## Step 3` in `evidence/T20.md`. Leave `/tmp/ship-check` where it is: it is the clone you demo from when your name is called.

*If it broke* at any step, the thing that broke is the thing a stranger would hit first; fix it in your working copy, commit, push, and clone again. Write under `## Step 3` what broke, if anything, and the commit that fixed it.

**Step 4. The last pull request.**

Fill `## Reflection` in `evidence/T20.md` (the questions are below) and run `uv run python scripts/check_evidence.py T20` until it says `all slots filled`. Then:

```bash
git add README.md evidence/T20.md
git commit -m "T20: README front page, final clean clone, Semester 1 shipped"
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

**Deliverable**
`README.md` with the front page + `evidence/T20.md` with every slot filled + the repo on `main` with every T01 to T22 branch merged and deleted + the Semester 1 Practical, ten minutes, oral.

**Reflection Questions** (answer under `## Reflection` in `evidence/T20.md`)

1. Paste your "What this is" paragraph. Then, for each of the three models in it (the Gram matrix of T06, the micrograd net of T12, the name MLP of T18), say in one line what question it answered about your corpus and what number is the answer.

   *How to get it:* which chapters are alike (a pair and a cosine), which half of the corpus a slice is from (a count out of 40), what the next letter of a name is (a test loss). Three questions, three numbers.

2. Paste your Step 3 lines. Name the one file in the repo that, if deleted, would break the most of those three commands, and say which commands it breaks and why.

   *How to get it:* `linear/chapter_vectors.py`: `train.py` reads `slices.py`, which reads `features.py`, which reads it; `demo.py` does not, because `names/` reads only `names.txt` and `mlp.pt`. Deleting `micrograd/engine.py` breaks the tests and `train.py` but not the demo.

3. Of the twenty-two assignments, name the one whose idea you would be least able to explain to a classmate without opening the repo, and the one you could explain from memory, with one sentence on why each.

   *How to get it:* no hint; this is the question I ask at the end of the practical.
