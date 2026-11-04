# T09 · One Neuron Reads Chapter 2, and Every Operation Learns to Pass the Gradient Back

**Meetings:** D32 · **Points:** 15 pts

**Watch — 26 min**
[Karpathy, building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=3070s) · 00:51:10 → 01:17:32, "preview of a single optimization step", "manual backpropagation example #2: a neuron" and "implementing the backward function for each operation" (26m)
 · `micrograd/engine.py`, `scratch/t09-video.py` (the same four import lines as T08's scratch file) and `evidence/T09.md` open, terminal in the repo root.

**During the video**

**Type everything he types, and run it.** Changes to the `Value` class go into `micrograd/engine.py`; the expressions he builds go into `scratch/t09-video.py`. Run the scratch file after every change he evaluates.

**00:51:10 → 00:52:52 · the preview.** He nudges every leaf of `L` a little in the direction of its gradient and `L` goes up. Before he re-runs, write under `## During the video` in `evidence/T09.md` whether `L` will go up or down, and why the gradient's sign decides it. Step 5 does this step on your neuron.

**00:52:52 → 01:09:02 · compute before he does.** He builds a neuron with two inputs, adds `tanh` to `Value`, and backpropagates through it by hand. Type `tanh` into the engine as he writes it. Then, for every node, **write the local derivative before he says it**, under `## During the video`: for a `+`, what each input gets; for a `*`, what each input gets; for `tanh`, what `do/dn` is in terms of the output `o`. Unpause and check. That page is practice for Step 3, where you do the same on your own neuron.

**01:09:02 → 01:17:32.** He moves each local derivative into the operation that creates it, as a small function called `_backward` stored on the output `Value`, and then calls those functions one at a time, by hand, in the right order from the output back. Type every `_backward` into `engine.py`. The order he calls them in is the point of Step 4; the engine learns to find the order itself in T10.

**Notes**

**A neuron is a weighted sum squashed.** `n = x1·w1 + x2·w2 + b`, then `o = tanh(n)`. `tanh` turns any number into one between -1 and 1 and keeps its sign. The two inputs today are two features of chapter 2 of your corpus from T01's counts: how unusual its rate of `WORD1` is, and how unusual its rate of `WORD2` is, each measured against the other nine chapters. A feature of `0` is an average chapter; `+1` is one standard deviation above average. The weights are fixed numbers today, the same for everyone, so your hand page and the engine see the same neuron.

**`tanh` has the derivative `1 - tanh(n)²`.** The `_backward` for `tanh` uses the output it already computed, `t`, so it reads `1 - t**2`. Write `1 - n**2` and you get plausible wrong numbers with no error.

**`+=` not `=`.** His `_backward` functions add to the gradient instead of setting it. Today's neuron uses every node once, so `=` would give the same numbers; the reason for `+=` arrives at 01:22:28 in T10, when a node is used twice. Type `+=` now anyway, because that is what he types, and because changing six closures later is how bugs get in.

**The order matters.** `o._backward()` fills in `n.grad`; only then can `n._backward()` fill in `s.grad` and `b.grad`; and so on down. Call them in the wrong order and a node passes back a gradient of `0.0`, because its own has not arrived yet. Step 4 calls them in the right order by hand; T10 replaces that with a sort.

**Walkthrough — your neuron, by hand and then by `_backward`**

**Step 1. Merge, branch, log.**
Open the gpt repo on GitHub and merge the T08 pull request if I have approved it; click **Delete branch**. Then:

```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/neuron
```

*If T08 is not approved yet:* `git switch dev/value && git switch -c dev/neuron`. The engine grows in every assignment from here to the milestone, so this branch has to start from the newest one.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # <your-username>-track, not main
bash scripts/start-entry.sh
```

Under the timestamp, write today's checklist:

```markdown
- [ ] micrograd 51:10 → 1:17:32 watched, tanh and every _backward typed into engine.py, local derivatives written before he said them
- [ ] Step 2: features.py read, neuron.py run
- [ ] Step 3: my neuron's gradients by hand, in evidence/T09.md and committed
- [ ] Step 4: neuron.py grads checks my page
- [ ] Step 5: one optimization step, o moves the way the gradient said
- [ ] Fill every slot in evidence/T09.md (check_evidence.py: all slots filled); push, open the PR, sign off
```

**Step 2. Two features of chapter 2, and the neuron that reads them.**

`micrograd/features.py` came with the template; open it and read the top. It takes T01's two counts per chapter, turns them into rates per 1,000 words, and measures each chapter against the other nine: `(rate - mean) / std`. Run it:

```bash
uv run python micrograd/features.py
```

*You should see* ten rows, one per chapter, with the two rates and the two features, then the means and spreads. On Homer, `WORD1 = the`, `WORD2 = jove`:

```text
chapter   the per 1000  jove per 1000      x1      x2
      0           47.5            1.5   -1.59   -1.01
      1           54.8            1.2   -0.62   -1.36
      2           49.9            2.0   -1.28   -0.36
      3           50.6            1.3   -1.18   -1.31
      4           66.3            2.5    0.91    0.20
      5           66.2            3.0    0.90    0.81
      6           63.8            3.3    0.59    1.16
      7           68.0            3.7    1.14    1.68
      8           65.7            2.8    0.83    0.55
      9           61.7            2.0    0.30   -0.36
mean 59.5 and 2.3 per 1,000 words; std 7.5 and 0.8
```

Read down the `x1` column. On Homer the first four chapters are below average and the last six above, with no overlap: chapters 0 to 3 are the Odyssey and 4 to 9 are the Iliad, and the Iliad says `the` more. A single word's rate separates the two books. Your corpus may or may not split this cleanly; say what you see under `## Step 2` in `evidence/T09.md`.

Then the neuron. `micrograd/neuron.py` builds it from chapter 2's row with `w1 = 0.5`, `w2 = -0.3`, `b = 0.1`, out of your `Value`s, node by node: `p1 = x1 * w1`, `p2 = x2 * w2`, `s = p1 + p2`, `n = s + b`, `o = n.tanh()`.

```bash
uv run python micrograd/neuron.py
```

*You should see* chapter 2's two features and the five node values. On Homer:

```text
chapter 2: x1 (the) = -1.28   x2 (jove) = -0.36   w1 = 0.5  w2 = -0.3  b = 0.1
p1 = -0.6400   p2 = 0.1080   s = -0.5320   n = -0.4320   o = tanh(n) = -0.4070
```

`o` is strictly between -1 and 1 with the same sign as `n`. Paste both lines under `## Step 2`. *If it broke* with `AttributeError: 'Value' object has no attribute 'tanh'`, the 00:52:52 segment's `tanh` is not in `engine.py` yet.

**Step 3. The hand page, committed before the engine can check it.**

Under `## Step 3` in `evidence/T09.md`, work backwards from `o`, one line per node, each with the local derivative you used and the number, in this order: `do/do = 1`; `do/dn = 1 - o²`, with your `o`; `do/ds` and `do/db` (a `+`); `do/dp1` and `do/dp2` (a `+`); then `do/dx1`, `do/dw1`, `do/dx2`, `do/dw2` (each a `*`: the other factor times the gradient arriving). Ten numbers to four decimals, from the two lines Step 2 printed. Then commit the file by itself:

```bash
git add evidence/T09.md
git commit -m "T09: my neuron's gradients by hand"
```

I will check your commit timestamps: the page has to exist before `neuron.py grads` runs.

**Step 4. Every `_backward`, called by hand, in order.**

This is the 01:09:02 segment on your neuron. `neuron.py grads` sets `o.grad = 1.0`, then calls `o._backward()`, `n._backward()`, `s._backward()`, `p1._backward()`, `p2._backward()`, in that order, and prints every gradient:

```bash
uv run python micrograd/neuron.py grads
```

*You should see* your two Step 2 lines, then ten gradients. On Homer:

```text
do/do  =  1.0000
do/dn  =  0.8344
do/ds  =  0.8344
do/db  =  0.8344
do/dp1 =  0.8344
do/dp2 =  0.8344
do/dx1 =  0.4172
do/dw1 = -1.0680
do/dx2 = -0.2503
do/dw2 = -0.3004
```

Hold your page next to it. `do/dn` is `1 - (-0.4070)² = 0.8344`, and every `+` on the way down passes it through unchanged, so `b`, `s`, `p1` and `p2` all get the same number. Then each `*` hands it on multiplied by the other factor: `do/dw1 = x1 · 0.8344 = -1.0680`, `do/dx1 = w1 · 0.8344 = 0.4172`. Where your page and the engine disagree, do not change the page; write the correct number next to the wrong one under `## Step 3` with the rule you misapplied. Paste the ten lines under `## Step 4`.

*If it broke* with every gradient below `o` printing `0.0000`, one `_backward` is not stored: `out._backward = _backward` is missing from an operation, so the default `lambda: None` ran instead. *If it broke* with `do/dn` right and everything below it `0.0000`, `n._backward` ran before `o._backward` filled `n.grad`; the order in the script is right, so the closure in `tanh` is reading the wrong variable.

**Step 5. One optimization step.**

The preview at 00:51:10, on your neuron. Before you run it, write under `## Step 5` which way `o` will move when every weight is nudged by `0.01` times its own gradient, and whether `w1` will go up or down (its gradient is negative on Homer; read yours). Then:

```bash
uv run python micrograd/neuron.py step
```

*You should see* `o` before and after, the change, and the new weights. On Homer:

```text
o before -0.406991   after -0.387543   change +0.019449
weights now: w1 = 0.4893  w2 = -0.3030  b = 0.1083
```

`o` went up, as moving along the gradient always does, and `w1` went down because its gradient was negative: making `w1` smaller raises `o` when `x1` is negative. Check the change against the gradients: `0.01 × (do/dw1² + do/dw2² + do/db²)` is `0.01 × (1.1406 + 0.0902 + 0.6962) = 0.0193`, close to the `0.0194` printed; the small gap is `tanh` curving over the step. Paste both lines under `## Step 5`. T10 moves the other way, to make a loss go down.

**Commit, push, PR, sign off.** Fill `## Reflection` in `evidence/T09.md` (the questions are below) and run `uv run python scripts/check_evidence.py T09` until it says `all slots filled`. Then:

```bash
git add micrograd/engine.py scratch/t09-video.py evidence/T09.md
git commit -m "T09: tanh and _backward per operation, my neuron's gradients by hand and by the engine, one step"
git push -u origin dev/neuron
```

Open the pull request: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**, stop. The PR body names the node whose local derivative you got wrong first, or the one you were least sure of.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`micrograd/engine.py` with `tanh` and a `_backward` on `+`, `*` and `tanh` + `scratch/t09-video.py` + `evidence/T09.md` with every slot filled, the Step 3 page committed before Step 4 ran.

**Reflection Questions** (answer under `## Reflection` in `evidence/T09.md`)

1. Paste the ten lines from Step 4 next to your Step 3 page. Pick `do/dw2` and write out the chain of factors the engine multiplied to get it, from `o` back to `w2`, naming the operation each factor came from. Then say which single factor would change if `o` had been `0.9` instead of your value, and what that does to every gradient below it.

   *How to get it:* `do/dw2 = (1 - o²) × 1 × 1 × x2`: the `tanh`, then two `+`s, then the `*` whose other factor is `x2`. On Homer `0.8344 × (-0.36) = -0.3004`. At `o = 0.9` the first factor is `1 - 0.81 = 0.19`, so every gradient below `n` shrinks to less than a quarter: a neuron near `±1` barely learns.

2. Paste the `features.py` table and say, from the `x1` column, whether a single word's rate separates parts of your corpus, and which parts. Then explain why the features are `(rate - mean) / std` instead of the raw rates, using the fact that `tanh` squashes everything past about `±2`.

   *How to get it:* the raw rate of `the` is about `50` per 1,000 words; `50 × 0.5` would put `n` at `25`, where `tanh` is `1.0000` and its derivative is `0.0000`. Features near `0` keep `n` where `tanh` still has a slope.

3. Paste your Step 5 lines next to your prediction. Say why `w1` moved down while `o` moved up, in one sentence that uses the sign of `x1`. Then say what the `0.01` is, what would happen with `1.0` in its place, and which T07 table tells you so.

   *How to get it:* `do/dw1 = x1 × 0.8344` is negative because `x1` is negative, so raising `o` means lowering `w1`. The `0.01` is the step size; T07's `far` table showed the slope is only right nearby, and a step of `1.0` would land where the slope no longer holds.
