# B22 · ★ MILESTONE B1: A Working Autograd Engine and a Trained Net

**Meetings:** D41–D43 · **Points:** 15 pts



**Watch — 13 min**
Day 1 — 13 min
[Karpathy, building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=7272s) · segment 02:01:12 → 02:14:03, "gradient descent manually, training" (13m)
 · `scratch/b22-video.py` open, with the same `sys.path` lines as before and `from nn import MLP`.

Day 2 — none. Today is the training run.

Day 3 — none. The one-step explanation, written in class, and the oral defense.

Optional, not counted, and only after your net trains: [3Blue1Brown, Neural Networks, Ch. 4: Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) (10:18). It is the chain rule you just implemented, in symbols. It lands far harder once your own engine works.

**During the video**

**Type everything he types, and run it.** His training loop goes into `scratch/b22-video.py`, on his four-example dataset from the end of B21. Run it after every change.

**Day 1 · 02:01:12 → 02:14:03.** He takes one gradient step by hand, then another, then wraps them in a loop and trains. Partway through, he finds a bug in his own loop. **When the loss starts behaving well, before he names the bug, pause and write in your scratch file, as a comment, what you think is missing from his loop.** Then unpause. The bug is the one the Notes below spend the most words on, and your net is about to have it too if you copy his first version into your own file instead of his fixed one.

**Notes**

Structure is a hierarchy you already have: `Neuron` (weights, bias, `tanh`) → `Layer` (a list of neurons) → `MLP` (a list of layers). `parameters()` at each level returns the flattened list from the level below. `MLP(3, [4, 4, 1])` has 41.

**The loop order is: forward → zero_grad → backward → update.** Every word of that line is load-bearing.

**The zero_grad bug.** `.backward()` adds into `.grad` with `+=`; that is what made B20's `b = a + a` come out right. It also means that if you never reset `.grad` between steps, every step's gradient piles on top of the last one. Your steps get bigger each time without you changing the learning rate. The first few steps look *better* than a correct loop, which is what makes it dangerous; then the loss spikes, and the net saturates and gets stuck. Step 6 makes you watch this on your own data.

Zeroing *after* `.backward()` wipes out the gradient you just computed, and the update then moves nothing. The loss prints the same number every step.

**Sign of the update:** `p.data += -lr * p.grad`. If your loss goes *up* smoothly, you have the sign backwards and you are doing gradient ascent.

**Keep it small.** 3 inputs, layers of `[4, 4, 1]`, eight training examples. Sixty steps should take a second or two. If it takes a minute, you have a bug, not a slow laptop: usually a graph that grows across steps because something outside the loop holds onto the previous loss.

The net starts from random weights, so two runs with different seeds give different loss curves. `random.seed(...)` before the `MLP` is built is what makes your run the same run twice. Without it, the oral defense cannot reproduce your number.

**Walkthrough — Train your net on your corpus, and defend it**

**Day 1 starts here.**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Then:
```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/milestone-b1
```
*If B21 is not approved yet:* `git switch dev/backward && git switch -c dev/milestone-b1`. The milestone needs B21's engine, `nn.py` and tests. If B25's resubmission changed the engine, merge that branch in too: `git merge dev/clinic`.
```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # should print <your-username>-track, not main
bash scripts/start-entry.sh
```
Under the timestamp, write the checklist for the whole milestone, and tick what you finish each day:
```markdown
- [ ] Day 1: Karpathy micrograd 02:01:12 → 02:14:03, typing his loop into scratch, predicting the bug before he names it
- [ ] Day 1: Steps 2–3, engine tests still green, predictions committed, train.py on my corpus with a loss curve
- [ ] Day 2: Steps 4–6, PyTorch gradient check to 6 decimals, new-window accuracy, the zero_grad bug on my data
- [ ] Day 2: Extension, five seeds at four learning rates
- [ ] Day 3: Step 7, the 150-word explanation, written in class
- [ ] Day 3: Step 8, oral defense
- [ ] Day 3: MILESTONE.md complete, pushed, PR described
```

Watch the Day 1 segment and type along before Step 2.

**Step 2. The engine still passes, and your predictions go in first.**

```bash
uv run pytest micrograd -q
```

*You should see* `5 passed`. If anything fails, stop and fix it; nothing below means anything on an engine that fails its own tests.

Then, before you train anything, write three predictions in `micrograd/PREDICTIONS.md`: the loss after 20 steps and after 60, with the settings in Step 3; how many of the 8 training windows the trained net will get right; and how many of 100 windows it has never seen. Your B19 hand-set neuron's score on those same 100 windows is in `micrograd/FINDINGS-b19.md`; use it. Commit it by itself:

```bash
git add micrograd/PREDICTIONS.md scratch/b22-video.py
git commit -m "B22: predictions before training"
git push -u origin dev/milestone-b1
```

Open the pull request now with **jd12** as reviewer. I will check your commit timestamps: predictions written after the run cap the Evidence score at 3.

**Step 3. Train on your corpus.**

Create `micrograd/train.py`:

```python
import random
import sys
import matplotlib.pyplot as plt
from nn import MLP
from data import load

def train(seed=1337, lr=0.02, steps=60, verbose=True):
    random.seed(seed)                              # same seed, same starting weights
    model = MLP(3, [4, 4, 1])
    xs, ys, _ = load()                             # your 8 windows: 4 real, 4 shuffled
    losses = []
    for k in range(steps):
        ypred = [model(x) for x in xs]                                     # forward
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))
        for p in model.parameters():                                       # zero_grad
            p.grad = 0.0
        loss.backward()                                                    # backward
        for p in model.parameters():                                       # update
            p.data += -lr * p.grad
        losses.append(loss.data)
        if verbose and (k < 20 or k % 10 == 9):
            print(f"step {k:3d}  loss {loss.data:.4f}")
    return model, losses

def accuracy(model, xs, ys):
    return sum((model(x).data > 0) == (y > 0) for x, y in zip(xs, ys))

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1337
    model, losses = train(seed)
    xs, ys, _ = load()
    tx, ty, _ = load(n_each=50, seed=99)           # 100 windows it never trains on
    print(f"train {accuracy(model, xs, ys)}/8   new windows {accuracy(model, tx, ty)}/100")
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("step"); plt.ylabel("loss (sum of squared errors, log scale)")
    plt.title(f"MLP(3, [4, 4, 1]) on my corpus, seed {seed}")
    plt.savefig("micrograd/loss.png", dpi=110)
```

Run it: `uv run python micrograd/train.py`.

*You should see* twenty lines of loss for steps 0 to 19, then every tenth step to 59, falling every step with this learning rate. The first loss is somewhere between about 5 and 16. It is a sum of eight squared errors, each at most 4, and random weights guess about half of them wrong. By step 59 it is well under 1, typically around 0.1. The last line shows the net getting all 8 of its training windows right and most, not all, of the 100 new ones. `micrograd/loss.png` is a curve that falls steeply, then bends and keeps falling slowly. Those numbers are yours; the shape is everyone's.

*If it broke:* a loss that prints the same number every step means `zero_grad` is after `backward`. A loss that climbs smoothly means the update has the wrong sign. `TypeError: unsupported operand type(s) for +: 'int' and 'Value'` on the `sum` line means `__radd__` is missing. A run that takes more than a few seconds means your engine's closures are keeping old graphs alive; ask me.

Commit and push. Sign off the log.

**Day 2 starts here.** Open a new entry with `bash scripts/start-entry.sh` in the log repo and copy over what is left of the checklist.

**Step 4. Every gradient in your net, against PyTorch, to six decimals.**

B21 checked one neuron. The milestone checks the whole net: all 41 gradients of the loss on your 8 windows, at the starting weights. Create `micrograd/check_torch.py`:

```python
import random
import torch
from nn import MLP
from data import load

random.seed(1337)
model = MLP(3, [4, 4, 1])
xs, ys, _ = load()

ypred = [model(x) for x in xs]
loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))
loss.backward()

X = torch.tensor(xs, dtype=torch.float64)
Y = torch.tensor(ys, dtype=torch.float64)
h = X
twins = []
for layer in model.layers:                       # copy your weights into PyTorch, layer by layer
    W = torch.tensor([[w.data for w in n.w] for n in layer.neurons],
                     dtype=torch.float64, requires_grad=True)
    b = torch.tensor([n.b.data for n in layer.neurons],
                     dtype=torch.float64, requires_grad=True)
    twins.append((layer, W, b))
    h = torch.tanh(h @ W.T + b)
tloss = ((h.squeeze(1) - Y) ** 2).sum()
tloss.backward()

print(f"loss  micrograd {loss.data:.6f}   torch {tloss.item():.6f}")
worst = 0.0
for layer, W, b in twins:
    for i, n in enumerate(layer.neurons):
        for j, w in enumerate(n.w):
            worst = max(worst, abs(w.grad - W.grad[i, j].item()))
        worst = max(worst, abs(n.b.grad - b.grad[i].item()))
print(f"compared {len(model.parameters())} gradients, largest difference {worst:.2e}")
print("match to 6 decimals:", worst < 5e-7)
```

Run it: `uv run python micrograd/check_torch.py`.

*You should see* the two losses printed to six decimals and equal, `compared 41 gradients`, a largest difference around `1e-15` or smaller, and `match to 6 decimals: True`. Look at `h @ W.T + b`: PyTorch does a whole layer as one matrix multiply, rows of windows times the transpose of a weight matrix, which is your B14 `matmul` and your B13 notation sheet's row convention. Your engine does the same arithmetic one scalar at a time.

*If it broke:* `False` with a difference near `1e-8` means one side is in 32-bit floats; both must be `float64`. A difference near `0.1` or larger means a `_backward` is wrong; B21's `vs_torch.py` on one neuron will tell you which operation, faster than this will.

**Step 5. What the training number does not tell you.**

You already have two numbers from Step 3: training windows right, and new windows right. Put your B19 hand-set neuron beside them, on the same 100 new windows, and a guess-everything-real baseline:

```bash
uv run python -c "import sys; sys.path.insert(0, 'micrograd'); from data import load; tx, ty, _ = load(n_each=50, seed=99); print('always real:', sum(y > 0 for y in ty), '/ 100')"
```

*You should see* `always real: 50 / 100`, because the new windows come in real and shuffled pairs. That is the floor. Your B19 hand neuron and your trained net both sit above it, and the trained net is not necessarily the higher of the two. A net that gets 8 of 8 by learning your eight windows has not necessarily learned the difference between real and shuffled text. Write the three numbers in your log.

**Step 6. The zero_grad bug, on your data.**

Copy `train.py` to `micrograd/no_zero_grad.py`, delete the two `zero_grad` lines, and change the plot's file name to `micrograd/loss-no-zero-grad.png`. Run it.

*You should see* the loss behave nothing like the correct run: stalling where the correct run was falling, or falling to almost zero, then a spike back up to a few units, and then a flat line. The flat number may be `0.0000` (the accumulated push happened to land every window on the right side) or an exact whole number such as `4.0000` or `8.0000`. If yours ends at `0.0000`, run `uv run python micrograd/no_zero_grad.py 5` and keep both printouts; seed 5 stalls at `4.0000` with 7 of 8. A flat `4.0000` is one window predicted with full confidence as the wrong class; `8.0000` is two of your eight windows predicted with full confidence as the wrong class: each contributes `(±1 − ∓1)² = 4`. The accumulated gradient has pushed the `tanh` units to their limits, where their slope is zero, and a unit with zero slope passes no gradient back, so the net can no longer move. Keep the printout; the oral defense asks about it. Commit and push.

**Day 3 starts here**. Open a new entry with `bash scripts/start-entry.sh`.

**Step 7. One training step, in 150 words, written in class.**

In class, without notes open, write `micrograd/ONE_STEP.md`: 150 words explaining what one single training step does, from forward pass to weight update, for a classmate in Track A who has never seen this. Use your own net: its 41 parameters, your 8 windows, your loss at step 0 and step 1 from Step 3. Commit it before you leave the room.

**Step 8. Oral defense.**

Five minutes with me, at your laptop, today. Have `train.py`, `engine.py`, `check_torch.py` and your two loss plots open. I will ask you to do some of these, not all:

| I ask you to | You |
|---|---|
| Walk me through one training step | Say forward, zero_grad, backward, update, pointing at the line in `train.py` for each |
| Show me the chain rule | Point at the line in `engine.py` where a local derivative multiplies `out.grad`, and say what each factor is |
| Run it live | `uv run python micrograd/train.py` and `uv run python micrograd/check_torch.py`, and get the numbers in your `MILESTONE.md` |
| Explain your worst number | Say where your net did worst, from the extension, and why you think it did |
| Break it | Say what the loss would do if I moved `zero_grad` below `backward`, then do it and run it |

Your numbers must reproduce on the spot. That is what `random.seed` is for.

**Extension — How much of your result is the seed and the learning rate?** *(assigned)*

One run is one sample. Before you run this, add a line to `PREDICTIONS.md` and commit it: which learning rate out of 0.02, 0.05, 0.1 and 0.2 will reach the lowest final loss, and whether it will also do best on the 100 new windows. I will check your commit timestamps.

Create `micrograd/seeds.py`:

```python
from train import train, accuracy
from data import load

xs, ys, _ = load()
tx, ty, _ = load(n_each=50, seed=99)
print(f"{'lr':>5} {'seed':>5} {'loss@0':>8} {'loss@20':>8} {'final':>8} {'rises':>6} {'train':>6} {'new':>8}")
for lr in (0.02, 0.05, 0.1, 0.2):
    for seed in range(5):
        model, losses = train(seed, lr=lr, verbose=False)
        rises = sum(b > a for a, b in zip(losses, losses[1:]))     # steps where the loss went UP
        print(f"{lr:5} {seed:5d} {losses[0]:8.3f} {losses[20]:8.4f} {losses[-1]:8.4f} {rises:6d}"
              f" {accuracy(model, xs, ys):4d}/8 {accuracy(model, tx, ty):4d}/100")
```

Run it: `uv run python micrograd/seeds.py`. It trains 20 nets and takes about ten seconds.

*You should see* 20 rows. At `lr 0.02` the loss never goes up (`rises 0`) and ends the highest of the four, though still well under 1. At `0.05` and `0.1` it ends lower and may go up a few times on the way. At `0.2`, some seeds end with the lowest losses of all, and others stall at `4.0000` or `8.0000` with 7 or 6 of 8 training windows right, the same saturation as Step 6. The column that does not follow the others is `new`: final training loss and new-window accuracy do not rank the runs in the same order, and a stuck run can score as well on new windows as a perfect one. Which rows do what is yours.

Then complete `micrograd/MILESTONE.md` with these sections, each with numbers from your runs:

| Section | What goes in it |
|---|---|
| Engine | `pytest` output; the `check_torch.py` output |
| Data | What your 8 windows are, how they were made from your corpus, and the invariant from B19 |
| Training | The Step 3 loss lines, `loss.png`, and your predictions beside them |
| Baselines | Always-real, your B19 hand neuron, and your trained net, all on the same 100 windows |
| Reliability | The `seeds.py` table, and the spread of `new` at your chosen learning rate |
| Where it got worse | At least two, with numbers: the zero_grad run from Step 6, the stalled seeds, the drop from 8/8 to your new-window score, or a window your net gets wrong (print it and paste it) |
| One step | A link to `ONE_STEP.md` |

Commit, push, sign off:

```bash
git add micrograd/ scratch/
git commit -m "B22: milestone — trained on my corpus, 41 gradients match PyTorch, seed/lr reliability"
git push
```

In the PR body, name the weakest number in `MILESTONE.md`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

Push again each day; one PR, not two; one log entry per meeting.

**Deliverable**
`micrograd/engine.py` + `micrograd/nn.py` (the complete engine and `Neuron`/`Layer`/`MLP`, about 150 lines together) + `micrograd/train.py` with `loss.png` (loss printed and falling over the first 20 steps) + `micrograd/check_torch.py` (all 41 gradients match to 6 decimals) + `micrograd/no_zero_grad.py` with its plot + `micrograd/seeds.py` + `micrograd/PREDICTIONS.md` (committed before training) + `micrograd/ONE_STEP.md` (150 words, written in class) + `micrograd/MILESTONE.md` + the oral defense on D43.

**Reflection Questions**

1. Paste your step 0 and step 1 loss lines from `train.py`. Pick one weight in your first layer (say which neuron and which input), print its `.grad` at step 0 and its `.data` before and after step 0's update, and paste them. Say in plain English what that gradient meant about your 8 windows, and check with your own numbers that the change in `.data` was exactly `-lr` times the gradient.

2. Paste the first twelve losses from `no_zero_grad.py` beside the first twelve from `train.py`. Say at which step the broken run first looked different (better or worse), at which step it first went wrong, and what the flat number it ended on means about specific windows. Then say why moving `zero_grad` *below* `backward` would have produced a different symptom from deleting it, and which one you would find faster.

3. Paste your `seeds.py` table and the prediction you committed about it. Name the run with the lowest final loss and the run with the best new-window score, and say whether they are the same run. Then say which number in `MILESTONE.md` you would stake the milestone on, and why that one and not the lowest loss.
