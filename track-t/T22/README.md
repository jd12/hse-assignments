# T22 · Practical Prep: the Derivation Page, the Demo, and Five Shape Questions

**Meetings:** D44 · **Points:** 15 pts

**Watch — none**
 · `evidence/T21.md` (the derivation page), `names/demo.py`, `names/PRACTICAL.md` (new), `evidence/T22.md`, terminal in the repo root.

No video. Tomorrow is the Semester 1 Practical: ten minutes, oral, one at a time while the others ship. Three parts: the name model demonstrated from a clean clone, the T21 derivation page explained aloud, and one question about a tensor shape in your own makemore code. Today is the rehearsal, and the deliverable is the page you will have in front of you.

**Notes**

**The demo is one command.** `uv run python names/demo.py` loads `names/mlp.pt` and prints ten names. With a seed it prints the same ten every time, which is what you want in a demo: `uv run python names/demo.py 7`. If `mlp.pt` is missing it says so and stops; T18 committed it, and Step 2 checks.

**The derivation page is explained, not read.** You will be asked to walk from `o2` back to `w1` on the page you typed in T21, saying at each node which operation it is and where its local derivative comes from. The numbers are already on the page; the explanation is what is being graded. Step 3 is a rehearsal with the clock running.

**The shape question is one of five, and you know all five now.** Step 4 is the five, each answered from your own `names/mlp.py` and `train_mlp.py` with the shape and the reason. Tomorrow I pick one.

**Walkthrough — rehearse all three**

**Step 1. Merge, branch, log.**
Open the gpt repo on GitHub and merge the T19 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/practical
```

*If T19 is not approved yet:* `git switch dev/embed && git switch -c dev/practical`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-track, not main
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] Step 2: demo.py from a clean clone, with a seed
- [ ] Step 3: the derivation page explained aloud, timed, twice
- [ ] Step 4: five shape questions answered in names/PRACTICAL.md
- [ ] Fill every slot in evidence/T22.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. The demo, from a clean clone.**

In a second terminal, the T21 clean-clone check again, on today's branch, with the demo at the end:

```bash
cd /tmp
rm -rf practical-check
git clone https://github.com/Sierra-Canyon/hse-2026-2027-gpt-<your-username>.git practical-check
cd practical-check
git switch dev/practical
cp ~/version_control/hse-2026-2027-gpt-<your-username>/data/corpus.txt data/
uv sync
uv run python names/demo.py 7
```

*You should see* the model's banner and ten names, the same ten every time with seed `7`. On Homer:

```text
name model trained on 508 names mined from data/corpus.txt   seed 7
   phoconisus
   lixnymederrus
   palli
   reons
   zmimon
   melopomantill
   ayons
   esens
   areteansus
   amyucon
```

Paste the ten under `## Step 2` in `evidence/T22.md`. Then run it once without the seed and once with `uv run python names/demo.py 42`, and write under them which seed you will use tomorrow and why that one.

*If it broke* with `names/mlp.pt is missing`, the model was never committed: `git add names/mlp.pt` in your working copy, commit, push, and clone again. *If it broke* with `ModuleNotFoundError: No module named 'counts'`, `names/names.txt` is missing from the clone and `counts.py` could not build the vocabulary; commit it. *If it broke* at `uv sync` on `torch`, the CPU index from T10 is not in the committed `pyproject.toml`.

**Step 3. The derivation page, aloud, timed.**

Open `evidence/T21.md` at `## Step 3`. Set a timer for three minutes and explain the page to the wall, or to the person next to you, from `do2/do2 = 1` down to `do2/dw1`, naming at each line the operation (`tanh`, `+`, `*`) and where the local derivative comes from (`1 - o²` from the output, the other factor for a product, `1` for a sum). Stop the timer. Write under `## Step 3` in `evidence/T22.md` how long it took, which line you stumbled on, and the one sentence you will say at that line tomorrow. Then do it again and write the second time.

**Step 4. Five shape questions.**

Create `names/PRACTICAL.md` with the demo command at the top, the seed you chose, and then these five, each answered with the shape and one sentence on why, from your own `mlp.py` and `train_mlp.py`:

1. `X` for your whole name list: its shape, and why the first number is T13's bigram total.
2. `C[X]`: its shape, and what each of the three numbers in it counts.
3. `emb.view(-1, 6)`: its shape, where the `6` comes from, and what `-1` asks PyTorch to do.
4. `h`, `logits` and `probs` for a minibatch of 32: three shapes, and which one `F.cross_entropy` takes.
5. `W1.grad` after `loss.backward()`: its shape, and why it must be the same as `W1`'s.

On Homer the answers start `(4125, 3)`, `(4125, 3, 2)`, `(4125, 6)`, `(32, 100) (32, 27) (32, 27)`, `(6, 100)`; yours have your example count in the first three. Check each against the printed shapes from T17 and T18. Paste the five answers under `## Step 4` in `evidence/T22.md`.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/T22.md` (the questions are below) and run `uv run python scripts/check_evidence.py T22` until it says `all slots filled`. Then:

```bash
git add names/PRACTICAL.md evidence/T22.md
git commit -m "T22: practical prep, the demo from a clean clone, the page rehearsed, five shapes"
git push -u origin dev/practical
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body is your demo seed and your second rehearsal time.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`names/PRACTICAL.md` (the command, the seed, the five shapes answered) + `evidence/T22.md` with every slot filled. Bring the T21 page and `PRACTICAL.md` tomorrow; nothing else is allowed on the desk.

**Reflection Questions** (answer under `## Reflection` in `evidence/T22.md`)

1. Paste your two rehearsal times and the line you stumbled on. Say what the line's local derivative is and where in `engine.py` it lives, in two sentences you could say aloud.

   *How to get it:* the usual stumble is `do2/dn1 = (1 - o1²) · do2/do1`: the `tanh` closure, `self.grad += (1 - t ** 2) * out.grad`, with `t` the output of the first neuron, not the second.

2. Paste your five shape answers. Pick the one you were least sure of and write the one-line `print` that would settle it, then say what it printed.

   *How to get it:* `uv run python -c "import sys; sys.path.insert(0, 'names'); from mlp import *; X, Y = build_dataset(words); print(C.shape)"` will not work because `C` is made inside `make_parameters`; the honest line builds it: `... p = make_parameters(torch.Generator().manual_seed(SEED)); print(p[0][X].shape)`.

3. Write the three sentences you will open with tomorrow, one per part: what the demo shows, what the page proves, and what a shape question is really asking.

   *How to get it:* the demo shows a model trained on names from your corpus making new ones; the page proves you can run the chain rule by hand through two neurons and the engine agrees; a shape question asks whether you know what every number in a tensor counts.
