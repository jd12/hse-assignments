# B17 · From One Variable to Many: Derivatives and Optimization

**Meetings:** D31 · **Points:** 15 pts

**Watch — 27 min**
[3Blue1Brown, Neural Networks, Ch. 2: Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) · whole video (20:33, 21m)
[Karpathy, The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=488s) · segment 00:08:08 → 00:14:12, "derivative of a simple function with one input" (6m)
 · Paper for 3Blue1Brown. `scratch/b17-video.py` open for Karpathy, and a terminal in the repo root.

**During the video**

**3Blue1Brown Ch. 2 · draw.** When he pictures the cost as a curve over a single input and rolls a ball down it, stop and draw that yourself: a bowl-shaped curve, a ball on its right-hand wall, and the tangent line at the ball. Write the sign of the slope at the ball, and draw an arrow showing which way the ball should move. Then put a second ball on the left-hand wall and do the same. The rule you just drew, "step against the sign of the slope", is Step 6 below, in code, on your corpus. Later in the chapter he counts the weights and biases in his network. Write that number down next to your drawing; your model today has one.

**Karpathy 00:08:08 → 00:14:12 · type and run.** He is in a notebook; you are in `scratch/b17-video.py`. **Type everything he types, and run it** with `uv run python scratch/b17-video.py`. He defines a quadratic, evaluates it, plots it over a range with NumPy and matplotlib, and then estimates its slope by nudging the input by a small `h` and dividing. Where he shows a plot inline, you add `plt.savefig("scratch/b17-f.png")`. When he computes the slope at a point, also compute the exact derivative of his function by the power rule, at the same point, and print both on one line. Do it at every point he tries. They should agree to about as many digits as his `h` is small, and that agreement is the whole idea of today's walkthrough.

**Notes**

The pivot of this week: in AP Calc, you find a minimum by setting the derivative to zero and solving. With one parameter you still can, and Step 6 does it once as a check. In ML the function has millions of inputs and there is no closed form. So you walk downhill instead. Everything after B17–B18 is a consequence of that.

Notation warning: the curly `∂f/∂x` is how partial derivatives are written. AP Calc never showed you the curly ∂. It means: differentiate with respect to `x` while pretending every other variable is a constant. That is the entire trick, and Step 7 is three problems of it.

**The numerical derivative has two ways to go wrong.** `h` too large and you are measuring the slope of a chord, not the tangent: at `h = 0.1` your numerical slope is visibly off. `h` far too small and floating point runs out of digits: at `h = 1e-12` it gets worse again, not better. `1e-6` is a safe middle for today. B21 measures exactly where this breaks.

**The learning-rate trap.** Too small and you crawl. Too large and every step overshoots the minimum by more than it started, so `w` flips sign, grows, and the loss goes to `inf`. On today's data everyone's limit is close to the same number, and Step 6 has you find it.

**PyTorch is installed today and not used until B21.** It is large. On a Mac, `uv add torch` gets the CPU build, which is the right one; nothing this semester uses a GPU. If the download on any machine is over a gigabyte, it is pulling the CUDA build: stop and tell me.

**Walkthrough — Slope of a loss, by nudging and by rule**

**Step 1. Merge, branch, log.**
Open the repo on GitHub and merge the pull request for the last assignment if I have approved it; click **Delete branch**. Then:
```bash
cd ~/version_control/hse-2026-2027-gpt-<your-username>
git switch main && git pull
git switch -c dev/derivative
```
```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
git branch --show-current   # should print <your-username>-track, not main
bash scripts/start-entry.sh
```
Under the timestamp, write today's checklist:
```markdown
- [ ] Watch 3Blue1Brown Deep Learning Ch. 2 and draw the two balls and their tangents
- [ ] Watch Karpathy micrograd 00:08:08 → 00:14:12, typing along in scratch/b17-video.py
- [ ] Step 2: uv add torch, CPU build, and check it imports
- [ ] Steps 3–6: windows.py, derivative.py, the plot, walking downhill at three learning rates
- [ ] Step 7: three partial-derivative problems by hand, committed before partials.py checks them
- [ ] Extension: how much my slope depends on which windows I drew
- [ ] Push dev/derivative and open the PR
```

**Step 2. Install PyTorch.**

```bash
uv add torch
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

*You should see* a version number and `False`. `False` is correct: CUDA is NVIDIA's GPU platform, and your laptop is not running it. The version ends in `+cpu` on some machines and in nothing on a Mac; both are the CPU build. If it ends in `+cu` and a number, you got the CUDA build: stop and use the CPU index line from A07.

*If it broke:* if `uv add` sits downloading for minutes with a size in the gigabytes, press Ctrl-C and tell me; your `pyproject.toml` is missing the CPU index. Commit `pyproject.toml` and `uv.lock` with today's work so your installation is reproducible.

**Step 3. A dataset from your corpus.**

The data today is a question about your corpus: if you read a stretch of `L` words, how many different words have you seen? Create `calc/windows.py`:

```python
import re
import numpy as np

def load_windows(n=200, seed=0):
    text = open("data/corpus.txt", encoding="utf-8").read()
    words = re.findall(r"[a-z']+", text.lower())
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(n):
        length = int(rng.integers(50, 1001))            # a stretch of 50 to 1000 words
        start = int(rng.integers(0, len(words) - length))
        window = words[start:start + length]
        xs.append(length / 100)                          # length, in hundreds of words
        ys.append(len(set(window)) / 100)                # distinct words seen, in hundreds
    return np.array(xs), np.array(ys)

if __name__ == "__main__":
    x, y = load_windows()
    print(x.shape, y.shape)
    print(f"x from {x.min():.2f} to {x.max():.2f}   y from {y.min():.2f} to {y.max():.2f}")
```

Run it: `uv run python calc/windows.py`.

*You should see* `(200,) (200,)`, `x` from about 0.5 to about 10 for everyone, since everyone's lengths are drawn the same way, and `y` from well under 1 to somewhere between 2 and 6, depending on how varied your corpus's vocabulary is. Measuring in hundreds keeps the numbers near 1, which keeps the learning rate from having to be absurdly small.

**Step 4. The loss, and its slope two ways.**

The model is a line through the origin, `y ≈ w · x`: distinct words are some fraction `w` of words read. The loss is the mean squared miss. Create `calc/derivative.py`:

```python
import numpy as np
import matplotlib.pyplot as plt
from windows import load_windows

x, y = load_windows()

def loss(w):
    return np.mean((w * x - y) ** 2)

def dloss(w):                      # by the chain rule: d/dw of (w x - y)^2 is 2 (w x - y) x
    return np.mean(2 * (w * x - y) * x)

h = 1e-6
print(f"{'w':>6} {'loss':>10} {'nudge slope':>12} {'rule slope':>12}")
for w in (0.0, 0.25, 0.5, 1.0):
    nudge = (loss(w + h) - loss(w)) / h
    print(f"{w:6.2f} {loss(w):10.4f} {nudge:12.5f} {dloss(w):12.5f}")
```

*You should see* four rows where the two slope columns agree to about five significant figures. The slope is negative at `w = 0` and positive at `w = 1`, so the minimum is between them: the same two balls you drew during 3Blue1Brown, with your data under them.

*If it broke:* if the two columns disagree in the first digit, the rule is missing the factor of 2 or the trailing `* x`. That is the chain rule's inner derivative, and it is the one people drop.

**Step 5. See the curve.**

Add:

```python
ws = np.linspace(-0.2, 1.0, 200)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
a1.plot(ws, [loss(w) for w in ws]); a1.set_xlabel("w"); a1.set_ylabel("loss")
a2.scatter(x, y, s=8); a2.set_xlabel("words read (hundreds)"); a2.set_ylabel("distinct words (hundreds)")
```

Do not save the figure yet; Step 6 adds the fitted line and saves it.

**Step 6. Walk downhill, three times.**

Add:

```python
def descend(lr, steps=30, w=0.0):
    for k in range(steps):
        w = w - lr * dloss(w)
    return w

w_star = np.sum(x * y) / np.sum(x * x)     # the AP Calc answer: dloss(w) = 0, solved
print(f"closed form      w = {w_star:.5f}")
for lr in (0.01, 0.025, 0.03):
    w = descend(lr)
    print(f"lr {lr:<6}  30 steps  w = {w:.5f}   loss {loss(w):.5f}")

w = descend(0.01)
a2.plot([0, 10], [0, 10 * w], color="red", label=f"w = {w:.3f}")
a2.legend()
plt.savefig("calc/fit_1d.png", dpi=110)
```

Run it: `uv run python calc/derivative.py`.

*You should see* the closed form and `lr 0.01` agree to five decimals: your `w`, somewhere between about 0.2 and 0.6. `lr 0.025` also lands, but if you print `w` every step you will see it jump back and forth across the answer, because it overshoots and comes back less each time. `lr 0.03` is nowhere near: `w` in the tens or worse, with a loss to match. The limit is where one step multiplies the distance to the minimum by more than 1 in size. It depends only on the `x` values, which everyone drew the same way, so everyone's limit is close to 0.027. The picture on the right shows your line through the cloud of windows, and it misses at both ends, which is a question for the extension, not a bug.

**Step 7. Many variables: three problems by hand, then one check.**

By hand, on paper, for `f(x, y) = x² + 3xy + y²`: compute `∂f/∂x` and `∂f/∂y`; evaluate both at `(1, 2)`; and write one sentence for each number explaining what it physically means about the surface at that point: which way you are facing, and how steep. Photograph or type it as `calc/PARTIALS.md`, and commit it before the next part: `git commit -m "B17: partials by hand"`. I will check your commit timestamps.

Then create `calc/partials.py`:

```python
def f(x, y):
    return x ** 2 + 3 * x * y + y ** 2

h = 1e-6
x, y = 1.0, 2.0
print("df/dx at (1,2) by nudging x, y held still:", (f(x + h, y) - f(x, y)) / h)
print("df/dy at (1,2) by nudging y, x held still:", (f(x, y + h) - f(x, y)) / h)
```

*You should see* two numbers within about `1e-5` of what you wrote by hand. "Holding `y` still" is not a figure of speech here: it is the `y` in `f(x + h, y)` that does not move. If a number disagrees with your page, find the mistake on paper and write the correction under the original. Do not erase the original.

**Extension — How much does your slope depend on which windows you drew?** *(assigned)*

Your loss is an average over 200 windows, so its slope is an average of 200 small slopes, and a different 200 windows would give a slightly different `w`. Before you run anything, predict in your log how far apart the five `w` values will be (largest minus smallest) with 10 windows each, and with 1,000 each. Commit the log. I will check your commit timestamps.

Create `calc/spread.py`:

```python
import numpy as np
from windows import load_windows

print(f"{'windows':>8}  {'w for seeds 0-4':>44}  {'spread':>7}")
for n in (10, 50, 200, 1000):
    ws = []
    for seed in range(5):
        x, y = load_windows(n, seed)
        ws.append(np.sum(x * y) / np.sum(x * x))
    print(f"{n:8d}  {np.round(ws, 4)!s:>44}  {max(ws) - min(ws):7.4f}")
```

It uses the closed form because with one parameter you are allowed to, and because descending five times at each size would say the same thing more slowly.

*You should see* the spread shrink as the number of windows grows, by roughly a factor of two for every four times as many windows, though five seeds is few enough that the pattern is rough. Your row for 200 should include the `w` from Step 6, since seed 0 with 200 windows is exactly the walkthrough's data.

Write `calc/FINDINGS-b17.md`: the table; your baseline, which is the Step 6 `w`; your two predictions, copied unchanged, with a sentence each on whether they held; and where it got worse, with the number: the 10-window spread against the 1,000-window spread, and what that means for a model that looks at only a handful of examples before each step. Then look at `calc/fit_1d.png` and name where your line misses worst: short windows or long ones, and in which direction.

Commit, push, open the pull request:

```bash
git add calc/ scratch/b17-video.py pyproject.toml uv.lock
git commit -m "B17: numerical vs rule derivative of a corpus loss, 1-D descent, slope spread"
git push -u origin dev/derivative
```

On GitHub: **Compare & pull request**, **jd12** under **Reviewers**, **Create pull request**. Say in the PR body where this is weakest.

```bash
cd ~/version_control/hse-2026-2027-student-log-<your-username>
bash scripts/sign-off.sh
git add logs && git commit && git push
```

**Deliverable**
`calc/windows.py` + `calc/derivative.py` + `calc/fit_1d.png` + `calc/PARTIALS.md` (three worked problems, committed before the check) + `calc/partials.py` + `calc/spread.py` + `calc/FINDINGS-b17.md` + `scratch/b17-video.py`, with `torch` in `pyproject.toml`.

**Reflection Questions**

1. Paste the four-row table from Step 4 and the three `lr` lines from Step 6. Using your own `dloss(0.0)` and your `lr 0.03` line, say what the very first step at `lr = 0.03` does to `w`, and why every step after it makes things worse rather than better. Give the number you think is your exact limit and how you would check it.

2. Paste the two lines `partials.py` printed and the matching lines from your committed `PARTIALS.md`. Did they agree the first time? If not, paste the original line and the correction. Then say what `∂f/∂x = 8` at `(1, 2)` means for someone standing on that surface and facing along `+x`, and what it says nothing about.

3. Paste your `spread.py` table beside the predictions you committed. Where was your prediction furthest off, and in which direction? Then use `calc/fit_1d.png` to say where a straight line through the origin is the wrong model for your corpus, and whether more windows would fix that or only make the wrong line more precise.
