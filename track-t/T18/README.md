# T18 · Train, Dev and Test, Minibatches, and the Learning-Rate Sweep on Your Names

**Meetings:** D42 · **Points:** 15 pts

**Watch — 38 min**
[Karpathy, Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I&t=2276s) · 00:37:56 → 01:15:40, "implementing the training loop, overfitting one batch", "training on the full dataset, minibatches", "finding a good initial learning rate", "splitting up the dataset into train/val/test splits and why", "experiment: larger hidden layer", "visualizing the character embeddings", "experiment: larger embedding size", "summary of our final code, conclusion" and "sampling from the model" (38m)
 · `scratch/t17-video.py` open, `evidence/T18.md`, terminal in the repo root.

**During the video**

**Type everything he types, and run it.** The training loop, first on 32 examples until the loss is near zero, then on everything with minibatches, then the learning-rate sweep, then the three splits, then the two experiments. The embedding picture and the sampling at the end are T19's walkthrough; type them today, read them on Monday.

**00:37:56 · compute before he does.** He trains on the first 32 examples alone and the loss falls to almost zero. Before he says why it cannot reach zero exactly, pause and write under `## During the video` in `evidence/T18.md` what two examples in those 32 have the same context and different targets. Then unpause.

**00:45:40 · compute before he does.** Before he plots loss against the learning-rate exponent, write which end of the plot (`-3` or `0`) will be high because the steps are too small, which end because they are too big, and roughly where the low point will be. Then unpause.

**00:53:20 · compute before he does.** Before he explains the three splits, write what goes wrong if the learning rate, the hidden size and the embedding size are all chosen by looking at the test loss. Then unpause.

**Notes**

**A minibatch is a random 32 of the examples, and the loss on it is a noisy estimate of the loss on all of them.** The gradient of that noisy loss is a noisy gradient, and a noisy gradient many times beats an exact gradient a few times. The printed minibatch loss jumps around; the split loss, computed on everything, is the one to read.

**The sweep finds the learning rate by running it from too small to too big in one go.** 1,000 steps with the exponent climbing from `-3` to `0`: the loss falls slowly, then fast, then flattens, then blows up. The low point of the smoothed curve is the rate to train at, and `0.1` is where it lands for this network on most name lists.

**Train, dev, test.** The network is fitted to the training split. The dev split is for choosing settings: learning rate, hidden size, how long to train. The test split is looked at once, at the end, and never used to choose anything, so its loss is an honest number. On 508 names the splits are 406, 51 and 51 names, which is small, and the gap between train loss and dev loss is the subject of the Extension. His 32,033 names make a gap of about `0.03`; yours will be bigger.

**Walkthrough — the sweep, then the splits**

**Step 1. Merge, branch, log.**
Open the gpt repo on GitHub and merge the T17 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/splits
```

*If T17 is not approved yet:* `git switch dev/mlp-names && git switch -c dev/splits`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-track, not main
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] makemore 2, 37:56 → 1:15:40 watched, typed along, the three answers written before he gave them
- [ ] Step 2: prediction in evidence/T18.md and committed, then train_mlp.py sweep; the picture opened
- [ ] Step 3: predictions committed, then train_mlp.py; train, dev and test read
- [ ] Extension: train_mlp.py big, predicted first
- [ ] Fill every slot in evidence/T18.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. The sweep.**

Open `names/train_mlp.py` and read the top: the shuffle with his seed, the 80/10/10 cut, `split_loss`, and the `sweep` block. Before you run it, write under `## Step 2` in `evidence/T18.md` the learning rate you expect the sweep to pick (as a power of ten), and commit the file. I will check your commit timestamps. Then:

```bash
uv run python names/train_mlp.py sweep
```

*You should see* the split sizes, the parameter count, `drawn: names/lr_sweep.png` with the low point, and the smoothed loss at five exponents. On Homer:

```text
508 names: train 406 / dev 51 / test 51   examples 3249 / 454 / 422
parameters: 3481
drawn: names/lr_sweep.png   lowest smoothed loss 2.895 at log10(lr) = -0.80, learning rate 0.158
   log10(lr)    -3   loss around there 19.036
   log10(lr)    -2   loss around there 9.931
   log10(lr)    -1   loss around there 3.146
   log10(lr)  -0.5   loss around there 3.205
   log10(lr)     0   loss around there 6.401
```

Open `names/lr_sweep.png`: high on the left where the steps are too small to get anywhere in the first few hundred steps, a valley around `-1`, and climbing again at `0` where the steps overshoot. The low point is near `0.1`, which is what the training block uses, and then `0.01` for the last stretch, his learning-rate decay. Paste the output under `## Step 2` and check your prediction.

**Step 3. Train on the training split, score all three.**

Before you run it, write two predictions under `## Step 3` in `evidence/T18.md` and commit the file: the training loss after 15,000 steps compared with T15's bigram loss (above, below, or about the same), and whether the dev loss will be above or below the training loss, and by roughly how much. I will check your commit timestamps. Then:

```bash
uv run python names/train_mlp.py
```

*You should see* the losses every 2,500 steps, the three final losses, `saved: names/mlp.pt` and `drawn: names/train_curve.png`. On Homer:

```text
step     0   lr 0.1   minibatch loss 19.3858   train 17.3791   dev 17.9430
step  2500   lr 0.1   minibatch loss 2.0541   train 2.2729   dev 2.4170
step  5000   lr 0.1   minibatch loss 2.0514   train 2.1372   dev 2.3141
step  7500   lr 0.1   minibatch loss 2.1742   train 2.1256   dev 2.3450
step 10000   lr 0.01   minibatch loss 1.7569   train 2.0654   dev 2.3069
step 12500   lr 0.01   minibatch loss 2.0581   train 1.9736   dev 2.2169
step 14999   lr 0.01   minibatch loss 2.0823   train 1.9680   dev 2.1913
final   train 1.9680   dev 2.1913   test 2.2449
saved: names/mlp.pt   drawn: names/train_curve.png
```

Read the last line. Train `1.97` is well below T15's bigram `2.19`: three characters of context beat one. Dev `2.19` is above train by `0.22`, which is the network knowing the 406 training names a little too well; his gap on 32,000 names was a few hundredths. Test `2.24` is the honest number, looked at once. The drop at step 10,000 when the rate falls to `0.01` is the decay doing its job. Paste everything under `## Step 3` and check your predictions. `names/mlp.pt` is the trained model; T19 and the practical sample from it.

*If it broke* with `train` and `dev` both near `17` after 2,500 steps, the update has the wrong sign; `train_mlp.py` is the template's. *If it broke* with the minibatch loss at `nan`, the learning rate was edited above `1`.

**Extension — Three times the hidden layer** *(assigned)*

`train_mlp.py big` trains the same way with a hidden layer of 300 neurons, 10,281 parameters. Before you run it, write under `## Extension` in `evidence/T18.md` what the bigger network will do to the training loss and what it will do to the dev loss, and commit the file. I will check your commit timestamps. Then:

```bash
uv run python names/train_mlp.py big
```

*You should see* the same shape of output with the final line different. On Homer:

```text
final   train 1.7968   dev 2.4105   test 2.2470
```

Training loss down from `1.97` to `1.80`, dev loss up from `2.19` to `2.41`. The bigger network has room to memorize 406 names and does, and what it memorized does not carry to the 51 it never saw. On 32,000 names the same experiment helped a little; on 508 it hurts. Write where it got worse under `## Extension`: both final lines side by side, the dev gap for each network, and which of the two you would ship for your corpus and why. Then open `names/train_curve.png` next to `names/train_curve_big.png`.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/T18.md` (the questions are below) and run `uv run python scripts/check_evidence.py T18` until it says `all slots filled`. Then:

```bash
git add names/mlp.pt names/lr_sweep.png names/train_curve.png names/train_curve_big.png scratch/t17-video.py evidence/T18.md
git commit -m "T18: the learning-rate sweep, train/dev/test on my names, the bigger hidden layer overfits"
git push -u origin dev/splits
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body is your final line with the three losses.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`names/mlp.pt` (the trained model) + `names/lr_sweep.png`, `names/train_curve.png`, `names/train_curve_big.png` + `scratch/t17-video.py` complete + `evidence/T18.md` with every slot filled, all three prediction sets committed before their runs.

**Reflection Questions** (answer under `## Reflection` in `evidence/T18.md`)

1. Paste your sweep lines beside your prediction. Say what the loss does at each end of the plot and why, and say what would be wrong with picking the learning rate by training ten full runs at ten rates instead.

   *How to get it:* too small and 1,000 steps go nowhere; too big and each step overshoots. Ten full runs cost ten times the compute to learn what one climbing run shows, and the sweep's answer is only rough, which is all a starting rate needs to be.

2. Paste your three final losses beside your predictions. Say what each of the three numbers is for, in one sentence each, and why the test loss is the only one you may quote as "how good the model is".

   *How to get it:* train is what the network fitted; dev chose the learning rate and the decay step and the hidden size; test was never used to choose anything, so nothing about the model was tuned toward it.

3. Paste both final lines from the Extension beside your prediction. Say why the larger network's training loss went down and its dev loss went up, using the number of training names, and name the one change that would let the bigger network help: more of what?

   *How to get it:* 10,281 parameters against 3,249 training examples is room to memorize; more names, not more neurons, is what would make the capacity pay. That is the ratio his 32,000 names had and yours does not.
