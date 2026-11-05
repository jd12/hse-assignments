# T11 · A Network Out of Neurons, and Forty Slices of Your Corpus

**Meetings:** D34 · **Points:** 15 pts

**Watch — 17 min**
[Karpathy, building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=6235s) · 01:43:55 → 02:01:12, "building out a neural net library (multi-layer perceptron) in micrograd", "creating a tiny dataset, writing the loss function" and "collecting all of the parameters of the neural net" (17m)
 · A new file, `micrograd/nn.py`, open next to `engine.py`; `scratch/t11-video.py` for his dataset and loss; `evidence/T11.md`; terminal in the repo root.

**During the video**

**Type everything he types, and run it.** His `Neuron`, `Layer` and `MLP` go into `micrograd/nn.py`, starting with `import random` and `from engine import Value`. His four-example dataset and the loss go into `scratch/t11-video.py`, which starts with the same `sys.path` line as T08's and `from nn import MLP`.

**01:43:55 · compute before he does.** When he builds `MLP(3, [4, 4, 1])`, pause and work out on paper how many parameters it has, weights and biases, layer by layer, before he prints the count. Write the layer-by-layer arithmetic and the total under `## During the video` in `evidence/T11.md`. Then unpause and check.

**01:51:04 → 02:01:12.** Four inputs, four targets, a loss that is the sum of the squared misses, `loss.backward()`, and `parameters()` collecting every weight and bias into one list. Type it, run it, and leave the scratch file there: T12 starts from it.

**Notes**

**A layer is a list of neurons that all read the same inputs; a network is a list of layers, each reading the one before.** That is the whole of `nn.py`. `MLP(2, [4, 4, 1])` is two inputs, a layer of four neurons, another layer of four, and one output neuron: `2×4 + 4` plus `4×4 + 4` plus `4×1 + 1` is `37` parameters. His `MLP(3, [4, 4, 1])` has 41. Write the arithmetic out; it is the During-the-video question and it comes up again in makemore.

**`sum(...)` inside `Neuron.__call__` starts from `self.b`.** `sum(iterable, start)` adds every term onto the start value, so the bias is the first thing in the chain and the products are added to it. Without a start it would begin from the integer `0`, which is the reason `__radd__` exists.

**Today's dataset is forty slices of your corpus.** `micrograd/slices.py` came with the template. It cuts four 2,000-word slices out of each of T01's ten chapters, measures each one with T09's two features, and labels it `-1` if it came from the first five chapters and `+1` if from the last five. The question the network will learn to answer in T12 is "which half of my corpus is this slice from," from two numbers. Today the network is untrained, and the point is to see what an untrained network's answers look like and what its loss is.

**Walkthrough — the network reads your corpus, badly**

**Step 1. Merge, branch, log.**
Open the gpt repo on GitHub and merge the T10 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/mlp
```

*If T10 is not approved yet:* `git switch dev/backward && git switch -c dev/mlp`.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-track, not main
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] micrograd 1:43:55 → 2:01:12 watched, nn.py typed, the parameter count worked out before he printed it
- [ ] Step 2: nn.py checked, 41 and 37
- [ ] Step 3: slices.py read and run, the forty slices looked at
- [ ] Step 4: prediction in evidence/T11.md and committed, then train.py before
- [ ] Fill every slot in evidence/T11.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. Check `nn.py`.**

At 02:01:12, `micrograd/nn.py` has three classes. `Neuron` holds `nin` random weights and a bias, `__call__` computes `sum(w·x) + b` and squashes it with `tanh`, and `parameters()` returns the weights and the bias as one list. `Layer` is a list of `nout` neurons, `__call__` runs each one on the same input and returns the single output unwrapped when there is only one, and `parameters()` joins the neurons' lists. `MLP` builds a `Layer` for each pair of sizes in `[nin] + nouts`, `__call__` runs the layers in order, and `parameters()` joins the layers' lists. Check it:

```bash
uv run python -c "import sys; sys.path.insert(0, 'micrograd'); from nn import MLP; print(len(MLP(3, [4, 4, 1]).parameters()), len(MLP(2, [4, 4, 1]).parameters()))"
```

*You should see* `41 37`. The first is his network and your During-the-video number; the second is the one today's walkthrough uses, two inputs instead of three. If your paper said something else for 41, write the layer-by-layer arithmetic again under `## During the video` and find the layer you miscounted.

*If it broke* with `TypeError: unsupported operand type(s) for +: 'int' and 'Value'`, `__radd__` is missing from `engine.py`, or `sum` in `Neuron.__call__` has no `self.b` start value. *If it broke* with a list where a number should be, `Layer.__call__` is not unwrapping the single output.

**Step 3. Forty slices of your corpus.**

Open `micrograd/slices.py` and read the top, then:

```bash
uv run python micrograd/slices.py
```

*You should see* forty rows, four per chapter, each with the slice's start position in the chapter, its two features and its label. On Homer the first and last few:

```text
40 slices of 2000 words, 4 per chapter   label -1: chapters 0-4   label +1: chapters 5-9
chapter  start      x1      x2    y
      0      0    2.74   -2.87   -1
      0   8814   -2.86   -2.26   -1
      0  17628   -2.86   -1.02   -1
      0  26442    0.01   -1.64   -1
...
      9      0    1.47   2.67   +1
      9   8814    0.27  -1.64   +1
      9  17628    0.27  -2.26   +1
      9  26442    0.07  -2.26   +1
```

The features are wider than T09's chapter features, some past `±3`, because a 2,000-word slice is a noisier sample than a 28,000-word chapter, and the spreads are still the chapter spreads. Read your forty rows and find the slices that look like they belong to the other half: on Homer, chapter 0's first slice has `x1 = 2.74`, an Iliad-sized rate of `the` in the Odyssey's first 2,000 words, and chapter 4's second slice has `3.47`. Those are the slices a network trained on two numbers cannot get right, and T12 finds them. Paste the first four rows and the last four under `## Step 3` in `evidence/T11.md`, and name under it the two slices in your table that look most like they belong to the other half.

**Step 4. One forward pass, untrained.**

`micrograd/train.py` builds `MLP(2, [4, 4, 1])` from `random.seed(0)`, so everyone starts from the same weights, runs every slice through it, and computes his loss: the sum over slices of `(prediction - label)²`. Before you run it, write two predictions under `## Step 4` in `evidence/T11.md` and commit the file: what the loss will be, roughly, for an untrained network on forty slices whose labels are `±1` (think about what `tanh` outputs look like when the weights are random, and what `(p - y)²` is when `p` is near `0`), and how many of the forty it will get right by the sign of its output. I will check your commit timestamps. Then:

```bash
uv run python micrograd/train.py before
```

*You should see* the parameter count, the loss, how many slices land on the right side of zero, and the first five predictions against their labels. On Homer:

```text
parameters: 37
before training: loss 29.3493   correct 30/40
first five predictions: [-0.091, -0.558, -0.542, -0.476, -0.449] labels [-1.0, -1.0, -1.0, -1.0, -1.0]
```

Thirty of forty is better than a coin, with no training at all. That is not the network knowing anything: with these particular random weights the output happens to lean negative, and half the labels are negative. A different seed could give 10 of 40. The loss is the number to watch: forty misses of about `0.5` to `1.5` each, squared and summed. Paste the three lines under `## Step 4` and check your two predictions. Then look at `scratch/t11-video.py` next to `train.py`: the forward pass, the loss and `parameters()` are his four lines, on forty examples instead of four.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/T11.md` (the questions are below) and run `uv run python scripts/check_evidence.py T11` until it says `all slots filled`. Then:

```bash
git add micrograd/nn.py scratch/t11-video.py evidence/T11.md
git commit -m "T11: Neuron, Layer and MLP; forty corpus slices through an untrained net"
git push -u origin dev/mlp
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the slice of your corpus you expect the network never to get right, and why.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`micrograd/nn.py` (`Neuron`, `Layer`, `MLP`, with `parameters()` on each) + `scratch/t11-video.py` + `evidence/T11.md` with every slot filled, the Step 4 predictions committed before `train.py before` ran.

**Reflection Questions** (answer under `## Reflection` in `evidence/T11.md`)

1. Paste your parameter arithmetic for `MLP(3, [4, 4, 1])` and for `MLP(2, [4, 4, 1])`. Then say how many parameters `MLP(2, [100, 1])` would have, and which of the two numbers that multiply in each layer the input count affects.

   *How to get it:* each layer is `inputs × neurons + neurons`. `2×100 + 100 = 300`, then `100×1 + 1 = 101`, so `401`. The input count only multiplies in the first layer; after that each layer's input count is the previous layer's neuron count.

2. Paste your Step 4 lines next to your committed predictions. Say what an untrained network's `30/40` means and does not mean, and what loss a network that output exactly `0` for every slice would have.

   *How to get it:* every prediction near zero means every miss is near `1`, so the loss is near `40`; your untrained loss is below that because the random weights happen to push some outputs the right way. `30/40` is one seed's luck on balanced labels, not knowledge.

3. From your Step 3 table, name the slice you expect the trained network never to get right, with its two numbers and its label, and say in one sentence what about your corpus put those numbers in that chapter.

   *How to get it:* a slice whose `x1` looks like the other half's. On Homer, chapter 4's second slice has `x1 = 3.47` with label `-1`: the Odyssey ends 43% of the way through the corpus, so chapter 4 is already partly the Iliad and its label is wrong for that slice before the network ever sees it.
