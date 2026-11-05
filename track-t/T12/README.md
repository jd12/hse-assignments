# T12 · Train It: a Hundred Steps Downhill on Your Corpus

**Meetings:** D35 · **Points:** 15 pts

**Watch — 25 min**
[Karpathy, building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=7272s) · 02:01:12 → 02:25:52, "doing gradient descent optimization manually, training the network", "summary of what we learned", "walkthrough of the full code of micrograd on github", "real stuff: diving into PyTorch, finding their backward pass for tanh" and "conclusion" (25m)
 · `scratch/t11-video.py` open (it becomes the training loop), `micrograd/engine.py` beside it, `evidence/T12.md`, terminal in the repo root.

**During the video**

**02:01:12 → 02:14:03 · type everything he types, and run it.** The training loop, in `scratch/t11-video.py`: forward pass, zero the gradients, `loss.backward()`, nudge every parameter against its gradient, repeat. **At the moment he says the gradients have to be reset**, pause and write under `## During the video` in `evidence/T12.md` what `+=` in every `_backward` does to a gradient on the second step if nobody sets it back to zero, and which way that pushes the parameters. Then unpause.

**02:14:03 → 02:25:52 · watch.** The summary, his finished `micrograd` on GitHub (compare its `engine.py` with yours: the ops and the `backward` are the same shape), and PyTorch's own `tanh` backward, which is `1 - t²` written in C++. Nothing to type. Write one line under `## During the video` naming one thing in his GitHub `engine.py` that yours does differently, or "nothing" if you found nothing.

**Notes**

**The loop is four lines, and the order matters.** Forward (predictions and the loss), zero every `grad`, `backward()`, update. The update is `p.data -= lr * p.grad`: minus, because the gradient points uphill and the loss should go down. T09 Step 5 went the other way on purpose, to make `o` go up.

**Zeroing the gradients is not optional.** Every `_backward` uses `+=`, which T10 showed is right when a node is used twice inside one backward pass. Between passes it is wrong: the gradient from step 1 is still sitting in `p.grad` when step 2 adds its own, so step 2 moves the parameters by the sum of two gradients, step 3 by the sum of three, and the steps grow until the network is thrown past every minimum. He mentions this as the commonest bug in the lecture, and the Extension measures it on your corpus.

**The learning rate is the T07 question again.** The gradient is right where you are; a step of `0.01` stays where it is right. `train.py` uses `0.01` for 100 steps. Larger finishes faster until it does not finish at all.

**Walkthrough — the network learns which half of your corpus it is reading**

**Step 1. Merge, branch, log.**
Open the gpt repo on GitHub and merge the T11 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/train
```

*If T11 is not approved yet:* `git switch dev/mlp && git switch -c dev/train`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-track, not main
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] micrograd 2:01:12 → 2:25:52 watched, the loop typed into scratch/t11-video.py, the zero-grad answer written before he said it
- [ ] Step 2: predictions in evidence/T12.md and committed, then train.py
- [ ] Step 3: the loss curve and the misses, read against T11's table
- [ ] Extension: train.py noreset, predicted first
- [ ] Fill every slot in evidence/T12.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. Train.**

Open `micrograd/train.py` and read the loop in the middle: it is the four lines from the video, on your forty slices, 100 steps at `0.01`, and it saves the loss at every step for a picture at the end. Before you run it, write two predictions under `## Step 2` in `evidence/T12.md` and commit the file: the loss after 100 steps (T11's untrained loss is your starting point; a perfect network's loss is `0`), and how many of the forty it will get right. Use your T11 table: the slices you named as belonging to the other half are the ones it will probably miss. I will check your commit timestamps. Then:

```bash
uv run python micrograd/train.py
```

*You should see* the loss every ten steps, the final loss and count, the slices it still gets wrong, and `drawn: micrograd/loss.png`. On Homer:

```text
parameters: 37
before training: loss 29.3493   correct 30/40
step   0   loss  29.3493   correct 30/40
step  10   loss  21.9942   correct 33/40
step  20   loss  21.0970   correct 33/40
step  30   loss  20.5266   correct 33/40
step  40   loss  19.9972   correct 34/40
step  50   loss  19.4943   correct 35/40
step  60   loss  19.0772   correct 35/40
step  70   loss  18.7580   correct 35/40
step  80   loss  18.5131   correct 35/40
step  90   loss  18.3141   correct 35/40
step  99   loss  18.1567   correct 35/40
after training: loss 18.1400   correct 35/40
misses (chapter, start, x1, x2, label, prediction):
   chapter 1  start     0   x1   0.21  x2  -1.64   label -1   predicted +0.309
   chapter 3  start 17628   x1   0.41  x2  -1.02   label -1   predicted +0.454
   chapter 4  start  8814   x1   3.47  x2  -1.64   label -1   predicted +0.382
   chapter 4  start 26442   x1   1.87  x2   2.06   label -1   predicted +0.849
   chapter 6  start  8814   x1  -1.80  x2  -0.41   label +1   predicted -0.922
drawn: micrograd/loss.png
```

The loss falls every ten steps and never rises: that is what a learning rate that is small enough looks like. It falls fast at first and slowly later, because the easy slices are learned in the first twenty steps and the rest are the ones the two features cannot settle. Paste everything from `before training` to the last miss under `## Step 2` and check your two predictions.

*If it broke* with the loss rising or printing `nan`, the update line has `+=` instead of `-=`, or `engine.py`'s `backward` is not zeroing anything it should not; `train.py` itself is the template's. *If it broke* with `ModuleNotFoundError: No module named 'matplotlib'`, `uv sync` first; it has been in `pyproject.toml` since T01.

**Step 3. Read the curve and the misses.**

Open `micrograd/loss.png`: one line, steep for twenty steps, then a slow slide. Then hold the misses against your T11 Step 3 table. On Homer, chapter 4's two late slices are there, as T11 predicted: one has an Iliad-sized `x1`, the other an Iliad-sized `x2`, and both are labeled Odyssey because chapter 4 straddles where one book ends and the other begins. Chapter 6's second slice is the other kind of miss: `x1 = -1.80`, a stretch of the Iliad that happens to say `the` as rarely as the Odyssey does. No network reading only these two numbers can get those right; it would need a third feature or a bigger slice. Write under `## Step 3` which of your misses are label noise (the slice really does belong to the other half, by where the books meet) and which are feature noise (the slice belongs where it is labeled, but its two numbers say otherwise), with one sentence each.

**Extension — What happens when nobody zeroes the gradients** *(assigned)*

`train.py noreset` runs the same loop without the zeroing lines, for 200 steps so the effect has time to show. Before you run it, write under `## Extension` in `evidence/T12.md` what the loss will do (keep falling, fall then rise, or explode) and how many of the forty will be right at the end, and commit the file. I will check your commit timestamps. Then:

```bash
uv run python micrograd/train.py noreset
```

*You should see* a loss that falls for about ten steps and then wanders, jumps and ends far above where it started falling, with a count no better than the untrained network's. On Homer the last lines:

```text
step 170   loss  35.2513   correct 30/40
step 180   loss  30.2802   correct 32/40
step 190   loss  34.6356   correct 31/40
step 199   loss  35.3653   correct 31/40
after training: loss 35.3317   correct 31/40
misses (chapter, start, x1, x2, label, prediction):
   chapter 2  start  8814   x1  -2.73  x2  -0.41   label -1   predicted +0.816
   chapter 4  start  8814   x1   3.47  x2  -1.64   label -1   predicted +1.000
   ...
   chapter 9  start 26442   x1   0.07  x2  -2.26   label +1   predicted -1.000
```

Look at the predictions: `+1.000` and `-1.000`, `tanh` saturated flat at its limits, with the sign wrong. The gradients were never cleared, so each step moved by the sum of every gradient so far, and after a hundred steps the parameters are being thrown back and forth across the minimum by a hundred stale slopes. A loss of `36` is nine slices wrong by the full `2`, squared. Open `micrograd/loss_noreset.png` next to `loss.png`. Write where it got worse under `## Extension`: the first step where the no-reset loss is above the reset loss at the same step, the final losses side by side, and the fact that the misses at the end are at `±1.000` and what that says about their gradients.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/T12.md` (the questions are below) and run `uv run python scripts/check_evidence.py T12` until it says `all slots filled`. Then:

```bash
git add micrograd/loss.png micrograd/loss_noreset.png scratch/t11-video.py evidence/T12.md
git commit -m "T12: trained the MLP on forty corpus slices, the curve and the misses, what stale gradients do"
git push -u origin dev/train
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names one miss and says whether it is the label's fault or the features'.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`micrograd/loss.png` and `micrograd/loss_noreset.png` + `scratch/t11-video.py` with the training loop + `evidence/T12.md` with every slot filled, both prediction sets committed before their runs. The milestone on D36 ships the whole of `micrograd/` from a clean clone.

**Reflection Questions** (answer under `## Reflection` in `evidence/T12.md`)

1. Paste your training lines next to your committed predictions. Say why the loss falls fast for twenty steps and slowly after, in terms of which slices are being learned when, and say what a loss of `18` on forty slices means per slice.

   *How to get it:* `18 / 40 = 0.45` per slice on average, but it is not spread evenly: thirty-five slices with a miss near `0.1` and five with a miss near `1.5` squared add up the same way. The early fall is the clean slices moving to the right side; the later slide is the network trying to bend its boundary around slices it cannot separate.

2. Paste your misses with your Step 3 sentences. For one label-noise miss, say what would fix it (the label, not the network), and for one feature-noise miss, say what a third feature would have to measure to fix it.

   *How to get it:* a label-noise miss sits where one book ends and the next begins; the fix is a label that follows the books, not the chapter cuts. A feature-noise miss has the right label and misleading rates; the fix is a feature that the two halves of your corpus differ on more reliably than one word's rate over 2,000 words.

3. Paste the last five lines of the `noreset` run and the zero-grad answer you wrote during the video. Say what `p.grad` held at step 50 without the reset, as a sum, and why a prediction of exactly `-1.000` has a gradient of exactly `0` and so can never be fixed by any further step.

   *How to get it:* without the reset, `p.grad` at step 50 is the sum of the gradients from all fifty backward passes. `do/dn = 1 - o²` is `0` at `o = ±1`, so a saturated neuron passes nothing back, and the parameters behind it stop moving however wrong it is.
