# T16 · One Layer of Neurons Learns the Matrix the Counts Gave for Free

**Meetings:** D40 · **Points:** 15 pts

**Watch — 34 min**
[Karpathy, building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo&t=5043s) · 01:24:03 → 01:57:45, "summary, preview to next steps, reference to micrograd", "vectorized loss", "backward and update, in PyTorch", "putting everything together", "note 1: one-hot encoding really just selects a row of the next Linear layer's weight matrix", "note 2: model smoothing as regularization loss", "sampling from the neural net" and "conclusion" (34m)
 · `scratch/t13-video.py` open, `evidence/T16.md`, terminal in the repo root.

**During the video**

**Type everything he types, and run it.** The loss on the whole dataset in one line, `W.grad = None`, `loss.backward()`, the update, then the loop, then sampling from the trained net.

**01:26:17 · compute before he does.** He walks the forward pass as a graph and says what micrograd did at each node. Before he does, write under `## During the video` in `evidence/T16.md` which of your `Value` operations `logits.exp()` is, which one the division by the row sums is, and which one `.log()` would be if your engine had it. Then unpause.

**01:38:36 · compute before he does.** Before the first `loss.backward()` and update, write what the loss will do after one step with his learning rate, and whether `W.grad` has the same shape as `W`. Then unpause.

**01:47:49 → 01:50:18 · note 1.** One-hot times `W` is a row of `W`. Write one sentence under `## During the video` saying what that means the 27 rows of `W` are, in terms of T14's matrix.

**Notes**

**This is micrograd with tensors.** `W` is 729 `Value`s in a grid, `requires_grad=True` is `Value`'s `grad` field, `loss.backward()` is your `backward()`, and `W.grad = None` is the zeroing T12 showed you cannot skip. The forward pass is one matrix multiply and the two lines of softmax, and there is no `tanh` because the network is one layer deep.

**The net is learning T14's matrix.** The input is a one-hot character, so `xenc @ W` picks out one row of `W`; `exp` and the row division turn that row into probabilities; the loss asks those probabilities to match what really followed. The best `W` the gradient can find is the one whose rows, after softmax, are T14's rows, and the walkthrough measures how close it gets in 100 steps. The difference between the two is the subject of the Extension.

**The learning rate is `50`.** Not a typo. The loss is an average over 4,000 bigrams, so its gradient on any one weight is tiny, and the step has to be large to move. T12's `0.01` was for a loss that was a sum over forty. The right learning rate is a property of the loss, not a universal number.

**Walkthrough — the net learns your counts**

**Step 1. Merge, branch, log.**
Open the gpt repo on GitHub and merge the T15 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/net
```

*If T15 is not approved yet:* `git switch dev/sample && git switch -c dev/net`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-track, not main
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] makemore 1:24:03 → 1:57:45 watched, typed along, the three During-the-video answers written before he said them
- [ ] Step 2: predictions in evidence/T16.md and committed, then net.py
- [ ] Step 3: the net's matrix against the counts, the biggest difference explained
- [ ] Extension: the same twenty samples, two models
- [ ] Fill every slot in evidence/T16.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. Train the one-layer net.**

Open `names/net.py` and read the loop: his loop, on your 4,000 bigrams. Before you run it, write two predictions under `## Step 2` in `evidence/T16.md` and commit the file: the loss after 100 steps compared with T15's counts loss (above it, below it, or equal), and whether the net's twenty samples with the same seed will be the same twenty as the counts gave. I will check your commit timestamps. Then:

```bash
uv run python names/net.py
```

*You should see* the bigram count, the loss every ten steps, the two losses side by side, the largest and average difference between the net's probabilities and the counts', the cell where the difference is largest, and twenty samples from each model. On Homer:

```text
4125 bigrams from 508 names
step   0   loss 3.7071
step  10   loss 2.4328
step  20   loss 2.3248
step  30   loss 2.2843
step  40   loss 2.2631
step  50   loss 2.2499
step  60   loss 2.2409
step  70   loss 2.2343
step  80   loss 2.2292
step  90   loss 2.2252
step  99   loss 2.2223
counts loss 2.1902   net loss 2.2220
largest difference between the net's probabilities and the counts': 0.9626   average 0.0154
   at row 'q' column 'u': counts say 1.0000, net says 0.0374, N there is 1
twenty names from the net:    cexze pomphus icans an eronimithaneragaus man a s axthus therygothimiclis cthus eptedoeuthempemsideustiavernioctls mhusidenstrheus das cus neslepyns istha epfdines mem des
the same twenty from counts:  cex macomphus icans an eronimithaneragaus man a s ax pus therygothimiclis cthus eptedoeuthempemes s eniavernioctlupidicidendorheus das cus neslepyns istha epidines
```

The loss starts above the uniform model's `3.30` (random weights are worse than guessing evenly) and in a hundred steps gets within `0.03` of the counts' `2.19`, which is the floor for any bigram model on these names. On average the net's probabilities are within `0.015` of the counts'. Paste from `counts loss` to the end under `## Step 2` and check your predictions.

*If it broke* with the loss going up or to `nan`, the update has `+` where it needs `-`, or the learning rate was edited; `net.py` is the template's. *If it broke* with `RuntimeError: ... requires_grad`, `W` was built without `requires_grad=True`.

**Step 3. Where the net disagrees with the counts.**

The largest difference on Homer is the `q` row: the counts say `u` follows `q` with probability `1.0`, because the one `q` in Homer's names is in a word with `qu`; the net says `0.037`. One sighting is not enough for gradient descent to push a whole row of 27 logits to put all its weight on one column in a hundred steps, so the net's `q` row is still close to where it started, nearly flat. That is smoothing, arrived at from the other direction: the counts matrix is exactly the data and nothing else, and the net is the data pulled toward a flat prior by its random start and its short training. Write under `## Step 3` your largest-difference cell, how many times that bigram appears in your names, and whether the net's version or the counts' version is the better guess for a name it has never seen.

**Extension — The same twenty, two models** *(assigned)*

Both models were sampled with the same seed, so the random walk draws the same sequence of random numbers, and a sample comes out the same from both until one draw lands where the two matrices disagree. Before you look closely, write under `## Extension` in `evidence/T16.md` how many of the counts' twenty names you expect to find somewhere in the net's twenty, and commit the file. I will check your commit timestamps. Then count: for each name in the counts' line, is it anywhere in the net's line?

*You should see* the two lists agree at the start, fall out of step, and come back into step. On Homer the first eight are nearly all shared (`icans`, `an`, `eronimithaneragaus`, `man`, `a`, `s`), then `ax` against `axthus` is the first divergence: the counts' `x` row ended the name and the net's `x` row, built from a handful of sightings, did not. A divergence like that uses up a different number of random draws, so the two walks are reading different random numbers for a while, and the lists differ (`pus` against `therygothimiclis`) until the draw counts happen to line up again, after which `therygothimiclis`, `cthus`, `das`, `cus`, `neslepyns` and `istha` appear in both. Thirteen of Homer's twenty are shared (`s` twice). Write where it got worse under `## Extension`: your shared count against your prediction, the first name that diverged with the character at which it did, and which model you would rather sample from for a corpus with few names, with a reason that uses the `q` row.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/T16.md` (the questions are below) and run `uv run python scripts/check_evidence.py T16` until it says `all slots filled`. Then:

```bash
git add scratch/t13-video.py evidence/T16.md
git commit -m "T16: one-layer net trained on my bigrams, its matrix against the counts, the same twenty samples"
git push -u origin dev/net
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body is your largest-difference cell.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`scratch/t13-video.py` complete through the first makemore video + `evidence/T16.md` with every slot filled, both prediction sets committed before their runs. Makemore's second video, the real network, starts on D41.

**Reflection Questions** (answer under `## Reflection` in `evidence/T16.md`)

1. Paste your loss lines and your During-the-video answer about `exp`, the division and `log`. Say which of micrograd's `_backward` closures PyTorch ran for each of the three operations in the forward pass, and what `W.grad = None` is in micrograd terms.

   *How to get it:* `exp` is your `exp`, the division is your `__truediv__` (so `__pow__` with `-1` and `__mul__`), the matrix multiply is `__mul__` and `__add__` many times over; `log` has no closure in your engine, which is one thing PyTorch has that micrograd does not. `W.grad = None` is the `p.grad = 0.0` loop from `train.py`.

2. Paste the largest-difference line and your Step 3 answer. Say why the net cannot reach the counts' loss exactly in finite steps when a row has one sighting, in terms of what the logits would have to be for softmax to put probability `1.0` on one column.

   *How to get it:* probability exactly `1.0` on one column needs that logit infinitely larger than the others; gradient descent only ever moves a finite step, so the net approaches and never arrives. That gap is also why the net generalizes slightly better than the counts.

3. Paste the two sample lines with the shared names marked. Say why the two lists can agree, then disagree for a stretch, then agree again, in terms of the random numbers the generator produces and how many each name uses up.

   *How to get it:* the generator produces one sequence of random numbers for both models; each pick turns the next number into a character using that model's row. The names match until a number falls where the two rows differ. From then on one walk is a different length, so the two are reading different numbers, until by chance both have used the same total and fall back into step.
